import os, zipfile, tempfile
from pathlib import Path
from backend.extractors import extract_text
from utils.constants import SUPPORTED_EXTENSIONS

def extract_zip_and_read_documents(zip_file):
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = os.path.join(temp_dir.name, "upload.zip")
    zip_file.save(zip_path) if hasattr(zip_file, "save") else open(zip_path, "wb").write(zip_file.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir.name)

    documents = {}
    for root, _, files in os.walk(temp_dir.name):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(root, fname)
                content = extract_text(fpath)
                if content.strip():
                    documents[fname] = content
    return documents