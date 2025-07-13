import tiktoken
import PyPDF2


def load_file(path):
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return "".join(page.extract_text() or "" for page in reader.pages)
    

text = load_file('meta_10k.pdf')
encoded_text=tiktoken.encoding_for_model('gpt-4o')

print(len(encoded_text.encode(text)))
