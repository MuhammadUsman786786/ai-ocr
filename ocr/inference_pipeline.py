import os
import time
import markdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama


# ============================================================
# Markdown → Text
# ============================================================
def md_to_text(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    html = markdown.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ============================================================
# Chunking
# ============================================================
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


# ============================================================
# Load LLaMA ONCE
# ============================================================
def load_llm(model_path: str):
    print("[LLM] Loading model...")
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count(),
        n_gpu_layers=-1,
        verbose=False
    )


# ============================================================
# Inference pipeline (MULTI-QUESTION)
# ============================================================
def run_inference_pipeline(
    markdown_path: str,
    questions: list[str],
    llm: Llama,
    top_k: int = 3,
):
    start_total = time.time()

    # Read + chunk
    text = md_to_text(markdown_path)
    chunks = chunk_text(text)

    # Embeddings (computed ONCE)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embed_model.encode(chunks, convert_to_tensor=True)

    # Encode all questions at once
    query_embeddings = embed_model.encode(questions, convert_to_tensor=True)

    # Semantic search
    hits = util.semantic_search(
        query_embeddings,
        corpus_embeddings,
        top_k=top_k
    )

    answers = []

    # One prompt per question
    for idx, question in enumerate(questions):
        context_chunks = [chunks[h["corpus_id"]] for h in hits[idx]]
        context = "\n\n".join(context_chunks)[:5000]

        prompt = f"""[INST]
Answer ONLY from the context.

Return the answer in EXACTLY this format:
Answer is: {{answer}}

Context:
{context}

Question:
{question}
[/INST]
"""

        output = llm(
            prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>"]
        )

        answers.append(output["choices"][0]["text"].strip())

    print(f"[LLM] Inference completed in {time.time() - start_total:.2f}s")

    return answers
