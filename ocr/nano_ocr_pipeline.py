import os
import time
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText


MODEL_ID = "nanonets/Nanonets-OCR-s"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def ocr_image(image: Image.Image, max_new_tokens=2048) -> str:
    start_time = time.time()

    # Resize image for memory efficiency
    image.thumbnail((1200, 1600))

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
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    elapsed = time.time() - start_time
    print(f"[OCR] Completed in {elapsed:.2f}s for this page")
    return output_text[0]

# -----------------------------
# Rewritten pipeline to match old signature
# -----------------------------
def run_ocr_pipeline(pdf_path: str, output_dir: str, dpi: int = 300):
    """
    PDF → Images → Markdown
    Returns markdown file path
    Uses HuggingFace/Nanonets OCR instead of DocStrange.
    Signature matches old pipeline exactly.
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
    all_markdown = []

    for idx, img_path in enumerate(image_paths, 1):
        print(f"[OCR] OCR page {idx}/{len(image_paths)}")
        img = Image.open(img_path)
        page_text = ocr_image(img)
        all_markdown.append(page_text)

    # Save combined markdown
    md_path = os.path.join(output_dir, "output.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdown))

    print(f"[OCR] Markdown saved: {md_path}")
    return md_path
