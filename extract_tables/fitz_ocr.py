import ocrmypdf
import os
import fitz 

# 1. OCR the scanned PDF and save it as searchable PDF
def ocr_pdf(input_path, output_path):
    print("Running OCR on the scanned PDF...")
    ocrmypdf.ocr(
        input_path,
        output_path,
        use_threads=True,
        skip_text=True,
        # deskew=True,
        clean=True,
        optimize=1,  # Lower = faster
        jobs=8       # Adjust based on your CPU core count
    )
    print("OCR completed.")


def extract_text_from_pdf(pdf_path):
    """
    Extracts and prints text from each page of a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        print(f"\n--- Page {page_number} ---\n{text}")

    doc.close()
# Paths
input_scanned_pdf = "Shell.pdf"
output_searchable_pdf = "searchable_pdf.pdf"

# Run OCR and extract tables
ocr_pdf(input_scanned_pdf, output_searchable_pdf)
extract_text_from_pdf(output_searchable_pdf)
