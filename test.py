import json
from pathlib import Path
import requests
from PyPDF2 import PdfReader

url = "https://peterhofmuseum.ru/special/print?subobj"
pdf_path = "data.pdf"
response = requests.get(url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
reader = PdfReader("data.pdf")
pdf_text = [page.extract_text() for page in reader.pages]
pdf_text = [i for i in pdf_text if i]
Path("tickets.json").write_text(json.dumps({"data": pdf_text}, ensure_ascii=False, sort_keys=False, indent=4), encoding='utf-8')
print(pdf_text)