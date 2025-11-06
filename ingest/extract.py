# # the PDF + image ingestion and OCR extractor.
# # This script will:
# # Take a PDF file from the /data folder.
# # Extract all text from each page.
# # Detect and save images from the PDF (like screenshots, diagrams).
# # Apply OCR (Optical Character Recognition) on those images to extract any text inside.
# # Save everything neatly into a JSON file inside /outputs.


# import pdfplumber
# import pytesseract
# from PIL import Image
# import json
# from pathlib import Path

# # --- CONFIG ---
# DATA_DIR = Path("data")
# OUTPUT_DIR = Path("outputs")

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# def extract_pdf(pdf_path: str):
#     pdf_path = Path(pdf_path)
#     print(f"🔍 Extracting from {pdf_path.name}...")

#     all_pages = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages, start=1):
#             print(f"Processing page {i}/{len(pdf.pages)}")

#             # Extract text from the page
#             text = page.extract_text() or ""

#             # Extract images from the page
#             images_data = []
#             if page.images:
#                 page_image = page.to_image(resolution=300)
#                 for j, img in enumerate(page.images):
#                     bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
#                     cropped = page_image.original.crop(bbox)

#                     img_filename = f"{pdf_path.stem}_page{i}_img{j}.png"
#                     img_path = OUTPUT_DIR / img_filename
#                     cropped.save(img_path)

#                     # OCR for text inside image
#                     ocr_text = pytesseract.image_to_string(cropped)

#                     images_data.append({
#                         "path": str(img_path),
#                         "ocr_text": ocr_text.strip()
#                     })

#             # Combine data
#             page_data = {
#                 "page": i,
#                 "text": text.strip(),
#                 "images": images_data
#             }
#             all_pages.append(page_data)

#     # Save output JSON
#     out_file = OUTPUT_DIR / f"{pdf_path.stem}_extracted.json"
#     with open(out_file, "w", encoding="utf-8") as f:
#         json.dump(all_pages, f, indent=2, ensure_ascii=False)

#     print(f"\n✅ Extraction complete! Saved to {out_file}")
#     return out_file


# if __name__ == "__main__":
#     pdf_files = list(DATA_DIR.glob("*.pdf"))  # find all PDFs in /data

#     if not pdf_files:
#         print("No PDF files found in /data/")
#     else:
#         for pdf_file in pdf_files:
#             extract_pdf(pdf_file)
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
import json
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ocr_image(img: Image.Image) -> str:
    """Preprocess image and run OCR."""
    # Convert to grayscale
    img_gray = img.convert("L")
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(2.0)
    # Optional: binarize
    img_bw = img_enhanced.point(lambda x: 0 if x < 128 else 255, "1")
    # OCR
    return pytesseract.image_to_string(img_bw, lang="eng", config="--psm 6").strip()


def extract_pdf(pdf_path: Path) -> Path:
    print(f"🔍 Extracting from {pdf_path.name}...")

    all_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract selectable text
            text = page.extract_text() or ""

            # Render page as image for OCR
            page_image = page.to_image(resolution=300).original
            ocr_text = ocr_image(page_image)

            # Save page image for embedding
            img_filename = OUTPUT_DIR / f"{pdf_path.stem}_page{i}.png"
            page_image.save(img_filename)

            all_pages.append({
                "page": i,
                "text": text.strip(),
                "ocr_text": ocr_text,
                "image_path": str(img_filename)
            })

    # Save JSON output
    out_file = OUTPUT_DIR / f"{pdf_path.stem}_extracted.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    print(f"✅ Extraction complete! Saved to {out_file}\n")
    return out_file


if __name__ == "__main__":
    # Process all PDFs in /data
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in /data/")
    else:
        for pdf_file in pdf_files:
            extract_pdf(pdf_file)
