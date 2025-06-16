from gmft.auto import AutoTableDetector, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document
import pandas as pd
import ocrmypdf
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import pickle

#  Configuration 
input_scanned_pdf = "Shell.pdf"
searchable_pdf_path = "searchable_pdf.pdf"
tables_cache_path = "extracted_tables.pkl"
MISTRAL_API_KEY = "08nQ6qoukkoLaW0XZ1RX3RiEANjjUUjK"  
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

#  Step 1: OCR scanned PDF 
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
        clean=True,
        optimize=1,
        jobs=8
    )
    print("✅ OCR completed.")

#  Step 2: Extract tables from PDF 
detector = AutoTableDetector()
formatter = AutoTableFormatter()

def extract_tables(pdf_path: str):
    # Check if cached tables exist
    if os.path.exists(tables_cache_path):
        print("✅ Loading cached tables...")
        with open(tables_cache_path, 'rb') as f:
            return pickle.load(f)
            
    print("Extracting tables from PDF...")
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
    
    # Cache the extracted tables
    print("✅ Caching extracted tables...")
    with open(tables_cache_path, 'wb') as f:
        pickle.dump(tables, f)
    
    return tables

#  Step 3: Convert tables to chunks 
def tables_to_chunks(tables):
    chunks = []
    for idx, df in enumerate(tables):
        for _, row in df.iterrows():
            chunk = f"Table {idx+1} Row: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
            chunks.append(chunk)
    return chunks

#  Step 4: Embed and build FAISS index 
def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

#  Step 5: Query using Mistral 
def query_table_rag(question, chunks, index, embed_model, top_k=1):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    payload = {
        "model": "mistral-small",
        "messages": [
            {"role": "system", "content": "You are a helpful table QA assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "top_p": 0.9
    }

    response = requests.post(MISTRAL_URL, headers=headers, json=payload)
    if response.status_code == 200:
        answer = response.json()['choices'][0]['message']['content'].strip()
    else:
        answer = f"❌ Error: {response.status_code} - {response.text}"
    
    return answer, retrieved_chunks

# Main Execution
if __name__ == "__main__":
    # Step 1
    ocr_pdf(input_scanned_pdf, searchable_pdf_path)

    # Step 2
    tables = extract_tables(searchable_pdf_path)

    # Step 3
    chunks = tables_to_chunks(tables)

    # Step 4
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, embeddings = build_faiss_index(chunks, embed_model)

    # Step 5
    question = "give Registered name: "
    answer, retrieved = query_table_rag(question, chunks, index, embed_model)

    print("\n🔍 Retrieved Context:")
    for r in retrieved:
        print("-", r)

    print("\n🧠 Answer:\n", answer)
