import sys
import os

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        print("Error: Neither pypdf nor PyPDF2 is installed.")
        sys.exit(1)

def extract_text(pdf_path, output_path):
    try:
        print(f"Processing {pdf_path}...")
        reader = pypdf.PdfReader(pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    header = f"--- Page {i+1} ---\n"
                    f.write(header)
                    f.write(text)
                    f.write("\n\n")
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf.py <pdf_path> <output_path>")
        sys.exit(1)
    
    extract_text(sys.argv[1], sys.argv[2])