import os
import torch

from ocr_pipeline import run_ocr_pipeline
from inference_pipeline import load_llm, run_inference_pipeline


def run_full_pipeline(
    pdf_path: str,
    output_dir: str,
    model_path: str,
    question: str,
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
        question=question,
        llm=llm
    )

    return answer


# ----------------------------
# CLI Entry
# ----------------------------
if __name__ == "__main__":
    PDF_PATH = "scanned_v2.pdf"
    OUTPUT_DIR = "./ocr_results"
    MODEL_PATH = "../models/llama-2-7b-chat.Q4_K_M.gguf"

    QUESTION = "schedule of water pump for SCP-CP-BF-01 TO 02"

    answer = run_full_pipeline(
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        question=QUESTION
    )

    print("\n✅ Final Answer:\n", answer)
