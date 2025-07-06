from pathlib import Path
import pdfplumber
from docx import Document as DocxDocument

def extract_text(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _read_pdf(file_path)
    elif ext == ".docx":
        return _read_docx(file_path)
    elif ext == ".txt":
        return _read_txt(file_path)
    return ""

def _read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def _read_docx(path):
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])

def _read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()