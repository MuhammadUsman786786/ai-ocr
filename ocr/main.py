import os
os.environ['CURL_CA_BUNDLE'] = ''

import torch

# from docstrange_ocr_pipeline import run_ocr_pipeline
from nano_ocr_pipeline import run_ocr_pipeline
from inference_pipeline import load_llm, run_inference_pipeline


def run_full_pipeline(
    pdf_path: str,
    output_dir: str,
    model_path: str,
    questions: list[str],
):
    # Device info
    if torch.backends.mps.is_available():
        print("[INFO] Apple MPS available")
    else:
        print("[INFO] CPU mode")

    # OCR
    markdown_path = run_ocr_pipeline(
        pdf_path=pdf_path,
        output_dir=output_dir
    )

    # LLM
    llm = load_llm(model_path)

    # Inference
    answer = run_inference_pipeline(
        markdown_path=markdown_path,
        questions=questions,
        llm=llm
    )

    return answer


# ----------------------------
# CLI Entry
# ----------------------------
if __name__ == "__main__":
    PDF_PATH = "scanned_3_pages.pdf"
    OUTPUT_DIR = "./ocr_results_v3"
    MODEL_PATH = "../models/llama-2-7b-chat.Q4_K_M.gguf"

    questions = [
        "for manufacturer York, what is the value of Refrigerant"
    ]

    answers = run_full_pipeline(
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        questions=questions
    )

    print("\n\n")
    print("\n".join(answers))


