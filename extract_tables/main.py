import fitz  # PyMuPDF
from gmft.auto import AutoTableDetector, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document
import os
from tqdm import tqdm

DPI = 150
input_pdf = "sample_pdf\Statement of Financial Performance Template.pdf"
ocr_output_pdf = "input-ocr.pdf"

detector = AutoTableDetector()
formatter = AutoTableFormatter()


def is_page_searchable(page, min_chars=10):
    text = page.get_text().strip()
    return len(text) >= min_chars

def has_tables(pdf_path, page_number):
    doc = PyPDFium2Document(pdf_path)
    try:
        page = doc[page_number]
        table_regions = list(detector.extract(page))
        return len(table_regions) > 0
    finally:                     
        doc.close()

def ocr_page(page, page_number):
    pix = page.get_pixmap(dpi=DPI)
    imgpdf = fitz.open("pdf", pix.pdfocr_tobytes())
    temp_path = f"temp_page_{page_number+1}.pdf"
    imgpdf.save(temp_path)
    imgpdf.close()
    return temp_path

def extract_tables_from_page(pdf_path, page_number):
    doc = PyPDFium2Document(pdf_path)
    page_tables = []
    try:
        page = doc[page_number]
        for cropped in detector.extract(page):
            formatted = formatter.extract(cropped, margin='auto', padding=None)
            df = formatted.df()
            page_tables.append(df)
    finally:
        doc.close()
    return page_tables

def smart_pdf_processing(input_pdf, ocr_output_pdf):
    src = fitz.open(input_pdf)
    ocr_doc = fitz.open()
    all_tables = []

    for i, page in enumerate(src, start=0):
        print(f"\n📄 Page {i + 1}")

        searchable = is_page_searchable(page)
        table_exists = has_tables(input_pdf, i)

        print(f"   🧠 Searchable: {'✅ Yes' if searchable else '❌ No'}")
        print(f"   📊 Tables: {'✅ Yes' if table_exists else '❌ No'}")

        if not table_exists:
                continue

        if searchable:
            page_tables = extract_tables_from_page(input_pdf, i)
        else:
            # Perform OCR and save temp file
            temp_path = ocr_page(page, i)
            page_tables = extract_tables_from_page(temp_path, 0)
            # Save OCR result to final output for optional future use
            temp_ocr_pdf = fitz.open(temp_path)
            ocr_doc.insert_pdf(temp_ocr_pdf)
            temp_ocr_pdf.close()
            os.remove(temp_path)

        all_tables.extend(page_tables)

        # Optionally save each table
        # for j, df in enumerate(page_tables, 1):
        #     filename = f"table_page{i+1}_{j}.csv"
        #     df.to_csv(filename, index=False)
        #     print(f"   ✅ Saved: {filename}")

    # Save merged OCR output (optional)
    if len(ocr_doc) > 0:
        ocr_doc.save(ocr_output_pdf)
        print(f"\n💾 OCR-enhanced pages saved to: {ocr_output_pdf}")
    ocr_doc.close()
    src.close()
    return all_tables

# Run the pipeline
tables = smart_pdf_processing(input_pdf, ocr_output_pdf)
print(f"\n✅ Total tables extracted: {len(tables)}")
print(tables)