import json
from pathlib import Path
import faiss
import numpy as np
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# --- CONFIG ---
VECTOR_DIR = Path("vectors")
FAISS_INDEX_FILE = VECTOR_DIR / "multi_modal_index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.json"
TOP_K = 5

# --- Load FAISS + metadata ---
if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
    raise FileNotFoundError("Run embed.py first to generate embeddings!")

faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --- Embedding Model ---
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utility Functions ---
def embed_text(text: str) -> np.ndarray:
    """Convert text into vector for FAISS search."""
    vec = text_model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)

def extract_text_from_pdf(file_path):
    """Extract text from PDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_image(file_path):
    """Extract text from image using OCR."""
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)

def create_temp_index(uploaded_files):
    """Create a temporary FAISS index for uploaded PDFs/images."""
    vectors = []
    temp_metadata = []

    for file in uploaded_files:
        if file.type == "application/pdf":
            content = extract_text_from_pdf(file)
        elif file.type.startswith("image/"):
            content = extract_text_from_image(file)
        else:
            continue

        vec = embed_text(content)
        vectors.append(vec)
        temp_metadata.append({"type": "text", "source": file.name, "content": content})

    if vectors:
        temp_index = faiss.IndexFlatL2(len(vectors[0]))
        temp_index.add(np.array(vectors))
        return temp_index, temp_metadata
    else:
        return None, []

def query_index(query_vec, top_k=TOP_K):
    """Query prebuilt FAISS index."""
    D, I = faiss_index.search(np.array([query_vec]), top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({"distance": float(dist), "metadata": metadata[idx]})
    return results

def query_combined_index(query_vec, top_k=TOP_K, temp_index=None, temp_metadata=[]):
    """Query both prebuilt FAISS index and temporary index (uploads)."""
    results = query_index(query_vec, top_k)

    if temp_index:
        D, I = temp_index.search(np.array([query_vec]), top_k)
        for idx, dist in zip(I[0], D[0]):
            results.append({"distance": float(dist), "metadata": temp_metadata[idx]})

    # Sort by distance (closest first)
    results = sorted(results, key=lambda x: x["distance"])
    return results[:top_k]

def generate_rag_answer(results, user_query):
    """Generate answer from Groq LLaMA API based on retrieved context."""
    context_chunks = []
    for r in results:
        meta = r["metadata"]
        if meta["type"] == "text":
            context_chunks.append(meta["content"])

    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant in a Retrieval-Augmented Generation (RAG) system.
Use ONLY the context below to answer the question.

Context:
{context}

Question: {user_query}
Answer clearly and concisely:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# --- CLI for testing ---
if __name__ == "__main__":
    print("=== ✅ Multi-Modal RAG Query (Groq LLaMA 3.1) ===")

    uploaded_files = []
    while True:
        file_input = input("\nEnter file path to upload (or 'done' to finish): ").strip()
        if file_input.lower() == "done":
            break
        if not Path(file_input).exists():
            print("File not found. Try again.")
            continue
        uploaded_files.append(Path(file_input))

    temp_index, temp_metadata = create_temp_index(uploaded_files)
    print(f"\n✅ Uploaded {len(uploaded_files)} file(s) and created temporary index.\n")

    while True:
        user_query = input("\nAsk your question (or type 'exit'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        query_vec = embed_text(user_query)
        retrieved = query_combined_index(query_vec, TOP_K, temp_index, temp_metadata)

        print("\n📌 Retrieved Sources:")
        for i, r in enumerate(retrieved, 1):
            m = r["metadata"]
            print(f"{i}. {m['type'].upper()} | {m.get('source')}")

        print("\n💡 Generating Answer...")
        answer = generate_rag_answer(retrieved, user_query)

        print("\n🧠 Answer:")
        print("-----------------------------------")
        print(answer)
        print("-----------------------------------\n")
