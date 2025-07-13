import os
import json
from datetime import datetime
from typing import List, Optional

import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from markdown import markdown
from weasyprint import HTML

# Load environment variables from a .env file (e.g., for GEMINI_API_KEY)
# Ensure you have a .env file in the same directory as this script
# with a line like: GEMINI_API_KEY='YOUR_ACTUAL_GEMINI_API_KEY_HERE'
load_dotenv()

def load_file(path: str) -> str:
    """
    Loads text content from a PDF file.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The extracted text content from the PDF.
    """
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            # Extract text from each page and concatenate
            return "".join(page.extract_text() or "" for page in reader.pages)
    except FileNotFoundError:
        print(f"Error: PDF file not found at {path}")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return ""

# Define the Pydantic model for the Annual Report data structure
class AnnualReport(BaseModel):
    company_name: str = Field(..., description="Name of the company as reported in the 10-K")
    cik: str = Field(..., description="Central Index Key (CIK) identifier assigned by the SEC")
    fiscal_year_end: datetime = Field(..., description="Fiscal year end date")
    filing_date: datetime = Field(..., description="Date when the 10-K was filed with the SEC")
    total_revenue: Optional[float] = Field(None, description="Total revenue for the fiscal year (in USD)")
    net_income: Optional[float] = Field(None, description="Net income (profit) for the fiscal year (in USD)")
    total_assets: Optional[float] = Field(None, description="Total assets at fiscal year end (in USD)")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities at fiscal year end (in USD)")
    operating_cash_flow: Optional[float] = Field(None, description="Net cash provided by operating activities (in USD)")
    cash_and_equivalents: Optional[float] = Field(None, description="Cash and cash equivalents at fiscal year end (in USD)")
    num_employees: Optional[int] = Field(None, description="Number of employees reported")
    auditor: Optional[str] = Field(None, description="Name of the external auditor")
    business_description: Optional[str] = Field(None, description="Company’s business overview (Item 1)")
    risk_factors: Optional[List[str]] = Field(None, description="Key risk factors (Item 1A)")
    management_discussion: Optional[str] = Field(None, description="Management’s Discussion & Analysis (Item 7)")

# Helper function to recursively clean the schema for API compatibility
def clean_schema(schema: dict) -> dict:
    """
    Recursively processes a JSON schema to remove unsupported keywords
    and flatten 'anyOf' for Optional types into 'nullable: true'.
    """
    if isinstance(schema, dict):
        cleaned = {}
        if 'anyOf' in schema:
            # Handle Optional types: flatten anyOf to type + nullable
            for item in schema['anyOf']:
                if item.get('type') == 'null':
                    cleaned['nullable'] = True
                else:
                    # Take the first non-null type definition
                    for k, v in item.items():
                        if k not in ['title', 'description']: # Still remove these from nested types
                            cleaned[k] = clean_schema(v)
        else:
            for k, v in schema.items():
                if k not in ['title', 'description', '$ref', '$defs']: # Remove common problematic keywords
                    cleaned[k] = clean_schema(v)
        return cleaned
    elif isinstance(schema, list):
        return [clean_schema(elem) for elem in schema]
    else:
        return schema

# Custom JSON serializer for datetime objects
def json_serial(obj):
    """Serializes datetime objects to ISO 8601 format."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# --- Main execution flow ---
if __name__ == "__main__":
    # Load the text from your PDF file
    # Make sure 'meta_10k.pdf' exists in the same directory as this script
    text = load_file('meta_10k.pdf')

    if not text:
        print("Exiting due to empty or unreadable PDF content.")
        exit()

    # Get the API key from environment variables
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please create a .env file in the script's directory with GEMINI_API_KEY='YOUR_API_KEY'.")
        exit()
    
    # Configure the google.generativeai library with the API key
    genai.configure(api_key=api_key)

    # Get the JSON schema from the Pydantic model
    raw_schema = AnnualReport.model_json_schema()
    
    # Construct the base schema for the API to be very specific and minimal.
    api_schema_base = {
        "type": raw_schema.get("type", "object"),
        "properties": raw_schema.get("properties", {}),
    }
    if "required" in raw_schema:
        api_schema_base["required"] = raw_schema["required"]
    
    # Recursively clean the schema to remove all problematic fields and flatten 'anyOf'
    api_schema = clean_schema(api_schema_base)

    # Construct the prompt for the Gemini model
    prompt = f'Analyze the annual report (10-K) and fill the data model based on it:\n\n{text}\n\n'
    prompt += 'The output needs to be in a JSON format matching the provided schema. No extra fields allowed!'

    try:
        # Instantiate the GenerativeModel
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Call the Gemini model to generate structured content
        response = model.generate_content(
            contents=prompt,
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': api_schema, # Pass the specifically constructed and cleaned minimal schema here
            }
        )

        # Validate the response against the Pydantic model
        ar = AnnualReport.model_validate_json(response.text)
        print("Successfully extracted Annual Report data:")
        # Use the custom json_serial function to handle datetime objects
        print(json.dumps(ar.model_dump(), indent=2, default=json_serial)) # Fix applied here

        # Convert the extracted data to Markdown format
        md_lines = [
            f"# {ar.company_name} Annual Report {ar.fiscal_year_end.year}",
            f"**CIK:** {ar.cik}",
            f"**Fiscal Year End:** {ar.fiscal_year_end.strftime('%Y-%m-%d')}",
            f"**Filing Date:** {ar.filing_date.strftime('%Y-%m-%d')}",
            "\n## Financials" # Added newline for better Markdown rendering
        ]

        if ar.total_revenue is not None:
            md_lines.append(f"- **Total Revenue:** ${ar.total_revenue:,.2f}")
        if ar.net_income is not None:
            md_lines.append(f"- **Net Income:** ${ar.net_income:,.2f}")
        if ar.total_assets is not None:
            md_lines.append(f"- **Total Assets:** ${ar.total_assets:,.2f}")
        if ar.total_liabilities is not None:
            md_lines.append(f"- **Total Liabilities:** ${ar.total_liabilities:,.2f}")
        if ar.operating_cash_flow is not None:
            md_lines.append(f"- **Operating Cash Flow:** ${ar.operating_cash_flow:,.2f}")
        if ar.cash_and_equivalents is not None:
            md_lines.append(f"- **Cash & Equivalents:** ${ar.cash_and_equivalents:,.2f}")
        if ar.num_employees is not None:
            md_lines.append(f"- **Number of Employees:** {ar.num_employees}")
        if ar.auditor:
            md_lines.append(f"- **Auditor:** {ar.auditor}")

        if ar.business_description:
            md_lines += ["\n## Business Description", ar.business_description]
        if ar.risk_factors:
            md_lines += ["\n## Risk Factors"] + [f"- {rf}" for rf in ar.risk_factors]
        if ar.management_discussion:
            md_lines += ["\n## Management Discussion & Analysis", ar.management_discussion]

        # Join Markdown lines with two line breaks for proper paragraph separation
        md = "\n\n".join(md_lines)
        
        # Convert Markdown to HTML
        html = markdown(md)
        
        # Prepare filename
        company = ar.company_name.replace(" ", "_").replace("/", "_") # Replace spaces and slashes for valid filename
        filename = f"annual_report_{company}_{ar.fiscal_year_end.year}.pdf"
        
        # Convert HTML to PDF using WeasyPrint
        print(f"\nGenerating PDF: {filename}...")
        HTML(string=html).write_pdf(filename)
        print(f"PDF generated successfully at {os.path.abspath(filename)}")

    except genai.types.BlockedPromptException as e:
        print(f"The prompt was blocked by the safety system: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call or data processing: {e}")

