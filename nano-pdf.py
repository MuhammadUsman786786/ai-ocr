import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import time
import os

# -----------------------------
# Model Setup
# -----------------------------
MODEL_ID = "nanonets/Nanonets-OCR-s"

# Load the model (FP16, MPS-compatible)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,   # FP16 saves memory
    device_map="auto"      # will use Apple MPS if available
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# -----------------------------
# OCR Function for single image
# -----------------------------
def ocr_image(image: Image.Image, max_new_tokens=2048) -> str:
    """
    OCR a single PIL image and return Markdown-friendly text.
    Logs the processing time.
    """
    start_time = time.time()

    # Preprocessing: resize for memory efficiency
    image.thumbnail((1200, 1600))  # smaller, keeps readability

    # Prompt for OCR
    prompt = """
Extract the document text naturally.

Rules:
- Preserve headings using Markdown (#, ##, ###)
- Return tables in GitHub-flavored Markdown
- Return equations in LaTeX
- Use ☐ and ☑ for checkboxes
- Wrap watermarks in <watermark></watermark>
- Wrap page numbers in <page_number></page_number>
- If an image has no caption, add a short description in <img></img>
"""

    messages = [
        {"role": "system", "content": "You are a document OCR assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Prepare model input
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Generate OCR output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode output
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    elapsed = time.time() - start_time
    print(f"OCR completed in {elapsed:.2f} seconds for this page")
    return output_text[0]

# -----------------------------
# OCR Function for PDF
# -----------------------------
def ocr_pdf(pdf_path: str, output_dir: str = "ocr_output", dpi=200, combined_md: bool = True):
    """
    Convert PDF → Images → OCR → Save Markdown
    combined_md: if True, saves one single Markdown file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    print(f"Converting PDF to images: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi)
    print(f"PDF has {len(pages)} pages")

    full_text = ""

    for i, page in enumerate(pages, start=1):
        print(f"\nProcessing page {i}/{len(pages)}")
        page_text = ocr_image(page)
        page_text = f"\n\n<!-- PAGE {i} -->\n\n{page_text}"

        # Save each page individually
        page_file = os.path.join(output_dir, f"page_{i}.md")
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(page_text)
        print(f"Saved OCR of page {i} → {page_file}")

        # Add to combined text
        full_text += page_text

    # Save combined Markdown if requested
    if combined_md:
        combined_file = os.path.join(output_dir, "combined_output.md")
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\nSaved combined Markdown → {combined_file}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # pdf_path = "scanned_document.pdf"  # Input PDF path
    pdf_path = "text-pdf.pdf"  # Input PDF path
    ocr_pdf(pdf_path, output_dir="ocr_results", dpi=200, combined_md=True)
