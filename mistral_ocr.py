import base64
import os
import pandas as pd
from mistralai import Mistral
import json

def encode_pdf_to_base64(pdf_path):
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

pdf_path = "shell.pdf"
api_key = "08nQ6qoukkoLaW0XZ1RX3RiEANjjUUjK"

base64_pdf = encode_pdf_to_base64(pdf_path)
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest", document={"type": "document_url", 
    "document_url": f"data:application/pdf;base64,{base64_pdf}"}, include_image_base64=False)

ocr_dict = ocr_response.model_dump()["pages"]
# Store each page as a chunk
page_chunks = []
i = 0
for i, page in enumerate(ocr_dict):
    i += 1
    page_num = page.get('page_num', '?')
    page_text = f"Page {i}:\n"

    if "elements" in page:
        for element in page["elements"]:
            if "markdown" in element:
                page_text += element["markdown"] + "\n"
    if "markdown" in page:
        page_text += page["markdown"] + "\n"

    page_chunks.append(page_text.strip())

# Print each page chunk
for i, chunk in enumerate(page_chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print("\n" + "="*200 + "\n")

