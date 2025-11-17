# ui.py
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import numpy as np
import faiss
import sys
from pathlib import Path

# Ensure parent folder is in sys.path so Python can find query.py
sys.path.append(str(Path(__file__).parent.parent))

from query import embed_text, query_index, generate_rag_answer, metadata as prebuilt_metadata, faiss_index as prebuilt_faiss

st.set_page_config(page_title="Multi-Modal RAG Demo", layout="wide")
st.title("Multi-Modal RAG - Upload your document")

st.markdown("""
Upload PDFs and images, and ask questions about their content. 
The system will also search the existing document database.
""")

# --- File upload ---
uploaded_files = st.file_uploader(
    "Upload PDFs or Images (multiple allowed)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# --- User query ---
question = st.text_input("Ask a question about your documents:")

# --- Process query when button is clicked ---
if st.button("Get Answer") and question:
    uploaded_texts = []
    uploaded_metadata = []

    # --- Extract text from uploaded files ---
    for file in uploaded_files:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    uploaded_texts.append(text)
                    uploaded_metadata.append({
                        "source": file.name,
                        "page": page_num,
                        "type": "text",
                        "content": text  # add content key
                    })
        elif file.type.startswith("image/"):
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            if text.strip():
                uploaded_texts.append(text)
                uploaded_metadata.append({
                    "source": file.name,
                    "type": "image",
                    "ocr_text": text,
                    "content": text  # add content key
                })

    # --- Build temporary FAISS for uploaded files ---
    if uploaded_texts:
        dim = embed_text("test").shape[0]  # embedding dimension
        temp_index = faiss.IndexFlatL2(dim)
        embeddings = np.array([embed_text(txt) for txt in uploaded_texts], dtype=np.float32)
        temp_index.add(embeddings)
    else:
        temp_index = None

    # --- Embed user query ---
    query_vec = embed_text(question).astype(np.float32)

    # --- Retrieve from uploaded content ---
    uploaded_results = []
    if temp_index:
        D, I = temp_index.search(np.array([query_vec]), min(5, len(uploaded_texts)))
        for idx, dist in zip(I[0], D[0]):
            uploaded_results.append({
                "distance": float(dist),
                "metadata": uploaded_metadata[idx]
            })

    # --- Retrieve from prebuilt FAISS ---
    prebuilt_results = query_index(query_vec)

    # --- Merge results (uploaded first) ---
    all_results = uploaded_results + prebuilt_results

    # --- Generate RAG answer ---
    with st.spinner("Generating answer..."):
        answer = generate_rag_answer(all_results, question)

    # --- Display answer ---
    st.subheader("Answer:")
    st.write(answer)

    # --- Display sources ---
    st.subheader("Sources:")
    for i, res in enumerate(all_results, 1):
        m = res["metadata"]
        if m["type"] == "text":
            st.write(f"{i}. TEXT | {m.get('source')} (page {m.get('page','?')})")
        else:
            st.write(f"{i}. IMAGE | {m.get('source')} → OCR text preview: {m.get('ocr_text','')[:100]}...")
