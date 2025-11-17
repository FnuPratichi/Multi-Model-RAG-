# Write an embed.py script that:
# Loads the JSON output from /outputs
# Generates text embeddings for text + ocr_text
# Generates image embeddings for image_path
# Saves all embeddings (either in .pkl or a vector DB like FAISS)

import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import numpy as np

# --- CONFIG ---
EXTRACTED_DIR = Path("outputs")  # JSON files from extract.py
VECTOR_DIR = Path("vectors")     # Where vector store will be saved
VECTOR_DIR.mkdir(exist_ok=True)

# Text model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Image model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# FAISS store config
EMBED_DIM = 384  # Text embedding dimension for all-MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(EMBED_DIM)

all_texts = []  # Keep track of metadata for retrieval

def embed_text(text):
    return text_model.encode(text, convert_to_numpy=True)

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)  # normalize
    return features.cpu().numpy()[0]

def process_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    for page in pages:
        # Combine main text + OCR text
        texts_to_embed = [page.get("text", ""), page.get("ocr_text", "")]
        for t in texts_to_embed:
            if t.strip():
                vec = embed_text(t)
                faiss_index.add(np.array([vec], dtype=np.float32))
                all_texts.append({
                    "type": "text",
                    "content": t,
                    "source": json_path.name,
                    "page": page.get("page", 0)
                })

        # Embed images
        for img in page.get("images", []):
            img_path = Path(img["path"])
            if img_path.exists():
                vec = embed_image(img_path)
                faiss_index.add(np.array([vec], dtype=np.float32))
                all_texts.append({
                    "type": "image",
                    "content": None,
                    "source": json_path.name,
                    "page": page.get("page", 0),
                    "image_path": str(img_path)
                })

if __name__ == "__main__":
    json_files = list(EXTRACTED_DIR.glob("*.json"))
    if not json_files:
        print("No extracted JSON files found!")
        exit(0)

    for jf in json_files:
        process_file(jf)

    # Save FAISS index and metadata
    faiss.write_index(faiss_index, str(VECTOR_DIR / "multi_modal_index.faiss"))
    with open(VECTOR_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, indent=2, ensure_ascii=False)

    print("✅ Multi-modal embeddings created and saved!")
