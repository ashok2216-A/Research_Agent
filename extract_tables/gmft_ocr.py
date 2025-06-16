# # 1. Install dependencies:
# from gmft.auto import AutoTableDetector, AutoTableFormatter
# from gmft.pdf_bindings import PyPDFium2Document
# import pandas as pd
# import ocrmypdf
# import tqdm

# def ocr_pdf(input_path, output_path):
#     print("Running OCR on the scanned PDF...")
#     # ocrmypdf.ocr(input_path, output_path, use_threads=True, skip_text=True, deskew=True, clean=True, optimize=3, jobs=4)
#     ocrmypdf.ocr(input_path, output_path, use_threads=True, skip_text=True, clean=True, optimize=3, jobs=4)
#     print("OCR completed.")

# # 2. Initialize detector + formatter
# detector = AutoTableDetector()
# formatter = AutoTableFormatter()

# def extract_tables(pdf_path: str):
#     tables = []
#     doc = PyPDFium2Document(pdf_path)
#     try:
#         for page in tqdm.tqdm(doc, desc="Extracting tables from pages"):
#             for cropped in detector.extract(page):
#                 formatted = formatter.extract(cropped)
#                 df = formatted.df()
#                 tables.append(df)
#     finally:
#         doc.close()
#     return tables

# # Usage
# input_scanned_pdf = "Shell.pdf"
# pdf_path = "searchable_pdf.pdf"
# ocr_pdf(input_scanned_pdf, pdf_path)
# dfs = extract_tables(pdf_path)

# # Output results
# for i, df in enumerate(dfs, 1):
#     print(f"\n📋 Table {i}:")
#     print(df.head())  # or use print(df.to_string(index=False))

#     # Optional: save to CSV/JSON
#     # df.to_csv(f"table_{i}.csv", index=False)
#     # df.to_json(f"table_{i}.json", orient="records")



# # !pip install torch torchvision transformers gmft pypdfium2 ocrmypdf tqdm

# from gmft.auto import AutoTableDetector, AutoTableFormatter
# from gmft.pdf_bindings import PyPDFium2Document
# import pandas as pd
# import ocrmypdf
# from concurrent.futures import ThreadPoolExecutor
# import tqdm
# import os

# # Function to perform OCR using ocrmypdf
# def ocr_pdf(input_path, output_path):
#     if os.path.exists(output_path):
#         print("✅ Skipping OCR - searchable PDF already exists.")
#         return
#     print("Running OCR on the scanned PDF...")
#     ocrmypdf.ocr(
#         input_path,
#         output_path,
#         use_threads=True,
#         skip_text=True,
#         # deskew=True,
#         clean=True,
#         optimize=1,  # Lower = faster
#         jobs=8       # Adjust based on your CPU core count
#     )
#     print("✅ OCR completed.")

# # Initialize GMFT detector and formatter
# detector = AutoTableDetector()
# formatter = AutoTableFormatter()

# # Multiprocessing table extraction per page
# def process_page(page):
#     page_tables = []
#     for cropped in detector.extract(page):
#         formatted = formatter.extract(cropped)
#         df = formatted.df()
#         page_tables.append(df)
#     return page_tables

# def extract_tables_parallel(pdf_path: str):
#     tables = []
#     doc = PyPDFium2Document(pdf_path)
#     try:
#         with ThreadPoolExecutor(max_workers=8) as executor:
#             results = list(tqdm.tqdm(executor.map(process_page, doc), total=len(doc)))
#             for page_number, page_tables in enumerate(results, 1):
#                 tables.extend(page_tables)
#                 print(f"Extracted {len(page_tables)} tables from page {page_number}")
#     finally:
#         doc.close()
#     return tables

# # ==== Main Usage ====
# input_scanned_pdf = "mistral.pdf"
# searchable_pdf_path = "searchable_pdf.pdf"

# ocr_pdf(input_scanned_pdf, searchable_pdf_path)
# dfs = extract_tables_parallel(searchable_pdf_path)

# # Display extracted tables
# for i, df in enumerate(dfs, 1):
#     print(f"\n📋 Table {i}:")
#     print(df.head())  # or use print(df.to_string(index=False))

#     # Optionally save
#     # df.to_csv(f"/content/table_{i}.csv", index=False)
#     # df.to_json(f"/content/table_{i}.json", orient="records")


# !pip install torch torchvision transformers gmft pypdfium2 ocrmypdf tqdm

from gmft.auto import AutoTableDetector, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document
import pandas as pd
import ocrmypdf
from concurrent.futures import ThreadPoolExecutor
import tqdm
import os

# Function to perform OCR using ocrmypdf
def ocr_pdf(input_path, output_path):
    if os.path.exists(output_path):
        print("✅ Skipping OCR - searchable PDF already exists.")
        return
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
    print("✅ OCR completed.")

# Initialize GMFT detector and formatter
detector = AutoTableDetector()
formatter = AutoTableFormatter()

def extract_tables(pdf_path: str):
    tables = []
    doc = PyPDFium2Document(pdf_path)
    try:
        for page_number, page in enumerate(doc, 1):
            page_tables = []
            for cropped in detector.extract(page):
                formatted = formatter.extract(cropped, margin='auto', padding=None)
                df = formatted.df()
                page_tables.append(df)
            tables.extend(page_tables)
            print(f"Extracted {len(page_tables)} tables from page {page_number}")
    finally:
        doc.close()
    return tables

# ==== Main Usage ====
input_scanned_pdf = "Shell.pdf"
searchable_pdf_path = "searchable_pdf.pdf"

ocr_pdf(input_scanned_pdf, searchable_pdf_path)
dfs = extract_tables(searchable_pdf_path)  # Changed to use sequential processing

# # Display extracted tables
# for i, df in enumerate(dfs, 1):
#     print(f"\n📋 Table {i}:")
#     print(df.head())  # or use print(df.to_string(index=False))

#     # Optionally save
#     df.to_csv(f"table_{i}.csv", index=False)
#     # df.to_json(f"/content/table_{i}.json", orient="records")



