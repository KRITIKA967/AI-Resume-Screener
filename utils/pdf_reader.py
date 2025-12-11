import fitz  # PyMuPDF

def extract_text_from_pdf(path):
    text = ""
    try:
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception:
        text = ""
    return text
