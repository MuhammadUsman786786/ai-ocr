import os
from pdf2image import convert_from_path
from docstrange import DocumentExtractor


def run_ocr_pipeline(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
):
    """
    PDF → Images → Markdown
    Returns markdown file path
    """

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "pages")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[OCR] Processing PDF: {pdf_path}")

    # PDF → Images
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, page in enumerate(pages):
        img_path = os.path.join(images_dir, f"page_{i+1:04d}.png")
        page.save(img_path, "PNG")
        image_paths.append(img_path)

    print(f"[OCR] Converted {len(image_paths)} pages")

    # OCR
    extractor = DocumentExtractor()
    all_markdown = []

    for idx, img in enumerate(image_paths, 1):
        print(f"[OCR] OCR page {idx}/{len(image_paths)}")
        result = extractor.extract(img)
        all_markdown.append(result.extract_markdown())

    # Save markdown
    md_path = os.path.join(output_dir, "output.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdown))

    print(f"[OCR] Markdown saved: {md_path}")

    return md_path
