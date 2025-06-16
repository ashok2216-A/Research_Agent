import ocrmypdf
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re

input_path = "Shell.pdf"
output_path = "searchable_pdf.pdf"
# Step 1: OCR the scanned PDF
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

# Step 2: Convert pages to images
images = convert_from_path("searchable_pdf.pdf")

# Step 3: OCR each image and extract structured table
custom_config = r'--psm 6'

for i, image in enumerate(images):
    print(f"\n--- 📄 Page {i+1} ---")

    raw_text = pytesseract.image_to_string(image, config=custom_config)
    print(f"\n### Raw OCR Text:\n{raw_text}")

    # Step 4: Convert raw lines into table-like format
    lines = [line for line in raw_text.split("\n") if line.strip()]
    rows = [re.split(r'\s{2,}|\t', line.strip()) for line in lines]  # Split by 2+ spaces or tabs
    rows = [row for row in rows if len(row) > 1]  # Keep only potential rows

    if not rows:
        print("No table-like rows found.")
        continue

    # Normalize column count
    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + ['']*(max_cols - len(row)) for row in rows]

    # Heuristically use first row as header
    df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])

    # Display as markdown table
    print("\n### Markdown Table:\n")
    print(df.to_markdown(index=False))
