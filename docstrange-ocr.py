import os
import torch
from pdf2image import convert_from_path
from docstrange import DocumentExtractor


def process_pdf_via_images(
    pdf_path: str,
    output_dir: str = "./ocr_results",
    dpi: int = 300,
):
    """
    Convert PDF to images, then OCR each image with DocStrange.
    """

    # ----------------------------
    # Validate input
    # ----------------------------
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "pages")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[INFO] Processing PDF: {pdf_path}")
    print(f"[INFO] Output directory: {output_dir}")

    # ----------------------------
    # Convert PDF → Images
    # ----------------------------
    print("[INFO] Converting PDF to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(images_dir, f"page_{i+1:04d}.png")
        page.save(img_path, "PNG")
        image_paths.append(img_path)

    print(f"[INFO] Converted {len(image_paths)} pages to images.")

    # ----------------------------
    # Initialize DocStrange
    # ----------------------------
    extractor = DocumentExtractor()

    # ----------------------------
    # OCR each image
    # ----------------------------
    all_markdown = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[INFO] OCR page {idx}/{len(image_paths)}: {img_path}")
        result = extractor.extract(img_path)
        all_markdown.append(result.extract_markdown())

    # ----------------------------
    # Save combined Markdown
    # ----------------------------
    md_path = os.path.join(output_dir, "output.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdown))

    print(f"[OUTPUT] Markdown saved: {md_path}")

    return {
        "markdown": md_path,
        "images_dir": images_dir,
        "pages": len(image_paths),
    }


# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":

    # ----------------------------
    # Device Info
    # ----------------------------
    if torch.backends.mps.is_available():
        print("[INFO] Apple MPS GPU available — DocStrange will use it automatically.")
    else:
        print("[INFO] MPS not available — running on CPU.")

    # ----------------------------
    # Input PDF
    # ----------------------------
    pdf_file = "scanned_v2.pdf"  # ← your PDF

    # ----------------------------
    # Run OCR
    # ----------------------------
    outputs = process_pdf_via_images(pdf_file)

    print("\n🎉 Done! Outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
