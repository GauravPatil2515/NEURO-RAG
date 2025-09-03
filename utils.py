
import fitz  

def pdf_to_text(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Text extracted and saved to {output_path}")
