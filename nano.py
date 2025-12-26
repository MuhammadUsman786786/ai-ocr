import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# -----------------------------
# Model Setup
# -----------------------------
MODEL_ID = "nanonets/Nanonets-OCR-s"

# Use dtype instead of deprecated torch_dtype
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,   # FP16 to save memory
    device_map="auto"      # uses MPS (Apple GPU) automatically
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Optional: force MPS device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# -----------------------------
# OCR Function (single image)
# -----------------------------
def ocr_image(image: Image.Image, max_new_tokens=4096) -> str:
    """
    Run Nanonets-OCR-s on a single image and return Markdown-friendly text.
    """
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

    # Prepare input for the model
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

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Extract generated text
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return output_text[0]

# -----------------------------
# PDF → OCR Pipeline
# -----------------------------
def ocr_pdf(pdf_path: str, dpi=200) -> str:
    """
    Convert a scanned PDF to Markdown using Nanonets-OCR-s page by page.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    full_text = ""

    for i, page in enumerate(pages, start=1):
        print(f"OCR page {i}/{len(pages)}")
        # Resize for memory efficiency
        page.thumbnail((1650, 2200))
        page_text = ocr_image(page)
        full_text += f"\n\n<!-- PAGE {i} -->\n\n{page_text}"

    return full_text

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    pdf_path = "scanned_document.pdf"  # path to your PDF
    output_path = "output.md"

    result = ocr_pdf(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"OCR complete → {output_path}")
