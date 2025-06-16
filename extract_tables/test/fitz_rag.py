import ocrmypdf
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import re
import requests
from sentence_transformers import SentenceTransformer

# 🔐 Set your Mistral API Key
MISTRAL_API_KEY = "08nQ6qoukkoLaW0XZ1RX3RiEANjjUUjK"  # ← Replace this

# Step 1: OCR
def ocr_pdf(input_path, output_path):
    print("Running OCR on the scanned PDF...")
    if os.path.exists(output_path):
        print("✅ Skipping OCR - searchable PDF already exists.")
        return 
    ocrmypdf.ocr(
        input_path,
        output_path,
        use_threads=True,
        skip_text=True,
        clean=True,
        optimize=1,
        jobs=8
    )
    print("OCR completed.")

# Step 2: Extract text and chunk by sentences per page
def extract_and_chunk_text(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        # Improved sentence splitting
        sentences = re.split(r'(?<=[.?!])[\s\n]+', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                chunks.append({
                    "page": page_number,
                    "chunk_id": f"{page_number}_{i}",
                    "text": sentence
                })
    doc.close()
    return chunks

# Step 3: Build FAISS vector DB
def build_vector_index(chunks, embedder):
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.encode(texts, show_progress_bar=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    return index, vectors

# Step 4: Query from FAISS
def query_text(chunks, embedder, index, query, top_k=1):
    query_vector = embedder.encode([query]).astype('float32')
    D, I = index.search(query_vector, top_k)
    results = []
    for idx in I[0]:
        chunk = chunks[idx]
        results.append(chunk)
    return results

# Step 5: Mistral API call
def call_mistral_chat(context, question):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are an assistant that answers questions based on provided text."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    payload = {
        "model": "mistral-large-latest",
        "messages": messages,
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------- Usage ----------

input_pdf = "/content/Shell.pdf"  # Replace with your PDF path
output_pdf = "searchable_pdf.pdf"

# 1. OCR
ocr_pdf(input_pdf, output_pdf)

# 2. Extract and chunk
chunked_texts = extract_and_chunk_text(output_pdf)

# 3. Vector DB
model = SentenceTransformer("all-MiniLM-L6-v2")
index, vectors = build_vector_index(chunked_texts, model)

# 4. Query
query = "revenue"
results = query_text(chunked_texts, model, index, query, top_k=1)

# 5. Show matching sentences
retrieved_context = " ".join([r["text"] for r in results])
# print("🔍 Retrieved Sentences:")
# for r in results:
#     print(f"\n📄 Page {r['page']} | Chunk {r['chunk_id']}\n{r['text']}\n{'-'*60}")

# 6. Ask Mistral for answer
print("\n💬 Mistral Final Answer:")
try:
    final_answer = call_mistral_chat(retrieved_context, query)
    print(final_answer)
except Exception as e:
    print("❌ Error calling Mistral API:", e)
