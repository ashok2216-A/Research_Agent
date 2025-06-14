# !pip install git+https://github.com/gsarti/gmft.git
# !pip install sentence-transformers faiss-cpu

from gmft.auto import CroppedTable, TableDetector, AutoTableFormatter, AutoTableDetector
from gmft.pdf_bindings import PyPDFium2Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 🚀 Step 1: Load and extract tables from PDF
pdf_path = "your_file.pdf"  # Replace with your actual PDF path
detector = AutoTableDetector()
formatter = AutoTableFormatter()

doc = PyPDFium2Document(pdf_path)
tables = []
for page in doc:
    tables += detector.extract(page)
doc.close()

# 📄 Step 2: Format tables (Markdown)
formatted_tables = [formatter.format(table, format="markdown") for table in tables]

# 🧠 Step 3: Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(formatted_tables)

# 💾 Step 4: Store in FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
faiss.write_index(index, "gmft_tables_index.faiss")

# 🔍 Step 5: Query
query = "What is the total amount mentioned in the invoice?"
query_vec = model.encode([query])
D, I = index.search(np.array([query_vec]), k=1)

# 🖨️ Step 6: Print matched table
print("\n🔎 Most relevant table:\n")
print(formatted_tables[I[0][0]])
