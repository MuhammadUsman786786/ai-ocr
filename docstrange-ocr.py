import os
import json
import torch
from docstrange import DocumentExtractor


def process_pdf(
    pdf_path: str,
    output_dir: str = "./ocr_results",
):
    """
    Process a PDF with DocStrange and save outputs.

    Args:
        pdf_path (str): Path to the input PDF
        output_dir (str): Directory to save results
    """

    # ----------------------------
    # Validate input
    # ----------------------------
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Processing PDF: {pdf_path}")
    print(f"[INFO] Output directory: {output_dir}")

    # ----------------------------
    # Initialize DocStrange
    # ----------------------------
    extractor = DocumentExtractor()

    # ----------------------------
    # Extract document
    # ----------------------------
    result = extractor.extract(pdf_path)
    print("[INFO] Extraction complete.")

    # ----------------------------
    # Save Markdown
    # ----------------------------
    md_path = os.path.join(output_dir, "output.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.extract_markdown())
    print(f"[OUTPUT] Markdown saved: {md_path}")

    # ----------------------------
    # Save JSON
    # ----------------------------
    json_path = os.path.join(output_dir, "output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.extract_data(), f, indent=2, ensure_ascii=False)
    print(f"[OUTPUT] JSON saved: {json_path}")

    # ----------------------------
    # Save CSV (tables if present)
    # ----------------------------
    csv_path = os.path.join(output_dir, "output_tables.csv")
    try:
        csv_text = result.extract_csv()
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        print(f"[OUTPUT] CSV saved: {csv_path}")
    except Exception:
        print("[WARN] No tables found — CSV not generated.")
        csv_path = None

    return {
        "markdown": md_path,
        "json": json_path,
        "csv": csv_path,
    }


# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":

    # ----------------------------
    # Device Info (informational only)
    # ----------------------------
    if torch.backends.mps.is_available():
        print("[INFO] Apple MPS GPU available — DocStrange will use it automatically.")
    else:
        print("[INFO] MPS not available — running on CPU.")

    # ----------------------------
    # Input PDF
    # ----------------------------
    pdf_file = "scanned_document.pdf"  # ← replace with your file

    # ----------------------------
    # Run OCR
    # ----------------------------
    outputs = process_pdf(pdf_file)

    print("\n🎉 Done! Outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
