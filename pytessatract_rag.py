# !pip install pdf2image pytesseract Pillow sentence-transformers faiss-cpu requests
# !sudo apt-get install poppler-utils tesseract-ocr -y

from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json

# 📄 Step 1: Convert PDF to images
pdf_path = "/content/Shell - Financial Statement-Page1-5 1.pdf"  # Update to your file path
images = convert_from_path(pdf_path)
image_paths = []
for i, img in enumerate(images):
    img_path = f"page_{i+1}.png"
    img.save(img_path)
    image_paths.append(img_path)

# 🤖 Step 2: Extract table rows as plain sentences
def extract_table_rows(img_path):
    image = Image.open(img_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    lines = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:
            line_num = data['line_num'][i]
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(data['text'][i])
    rows = []
    for line in lines.values():
        row = [cell for cell in line if cell.strip()]
        if row:
            sentence = ", ".join(row)
            rows.append(sentence)
    return rows

all_rows = []
for path in image_paths:
    all_rows.extend(extract_table_rows(path))

# 🧠 Step 3: Embed rows with SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_rows)

# 💾 Step 4: Store in FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
faiss.write_index(index, "row_sentences_index.faiss")

# 🔍 Step 5: Semantic Search Query
query = "What is the equity instruments remeasurements?"
query_vec = model.encode([query])
D, I = index.search(np.array(query_vec), k=1)
retrieved_row = all_rows[I[0][0]]

# 🤖 Step 6: Use Mistral AI API directly with requests
def query_mistral_api(api_key, messages, model="mistral-small"):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Replace with your actual Mistral API key
api_key = "08nQ6qoukkoLaW0XZ1RX3RiEANjjUUjK"  # Replace with your actual API key

messages = [
    {"role": "system", "content": "You are a helpful financial assistant. Provide concise and clear answers."},
    {"role": "user", "content": f"Context: {retrieved_row}\n\nQuestion: {query}"}
]

response = query_mistral_api(api_key, messages)

# ✅ Step 7: Output the result
print("\n🔍 Most relevant sentence:\n")
print(retrieved_row)
print("\n🤖 Mistral's answer:\n")
if response and "choices" in response:
    print(response["choices"][0]["message"]["content"])
else:
    print("Failed to get response from Mistral API")