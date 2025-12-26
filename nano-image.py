import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

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
# OCR Function
# -----------------------------
def ocr_image(image_path: str, max_new_tokens=2048) -> str:
    """
    OCR a single image and return Markdown-friendly text.
    """
    # Open image
    image = Image.open(image_path).convert("RGB")

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

    return output_text[0]

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    image_path = "pdf_pages/page_1.png"  # input image
    output_path = "page_1_output.md"

    result = ocr_image(image_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"OCR complete → {output_path}")
