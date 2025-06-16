import ocrmypdf
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: OCR the scanned PDF
input_path = "/content/Shell.pdf"
output_path = "searchable_pdf.pdf"
ocrmypdf.ocr(
    input_path,
    output_path,
    use_threads=True,
    skip_text=True,
    clean=True,
    optimize=1,
    jobs=8
)
print("✅ OCR completed.")

# Step 2: Convert pages to images
images = convert_from_path(output_path)

# Step 3: OCR each image and collect sentences with page info
custom_config = r'--psm 6'
page_wise_sentences = []  # List of dicts: [{'text': ..., 'page': ...}, ...]

for i, image in enumerate(images):
    raw_text = pytesseract.image_to_string(image, config=custom_config)
    sentences = re.split(r'\n+', raw_text)
    for s in sentences:
        s = s.strip()
        if s:
            page_wise_sentences.append({
                "text": s,
                "page": i + 1  # Page numbers start from 1
            })

# Step 4: Embed the sentences using SentenceTransformers
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [entry["text"] for entry in page_wise_sentences]
embeddings = model.encode(texts)

# Step 5: Store embeddings in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print(f"✅ FAISS index built with {len(texts)} sentences across {len(images)} pages.")

# Step 6: Function to search and show page-wise result
def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    print("\n🔍 Top relevant results:")
    for rank, idx in enumerate(I[0]):
        result = page_wise_sentences[idx]
        print(f"{rank+1}. 📄 Page {result['page']}: {result['text']}")

# 🔍 Example query:
semantic_search("what is the total revenue")
