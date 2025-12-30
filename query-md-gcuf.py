import os
import time
import markdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

# ============================================================
# 1️⃣ Load & preprocess Markdown
# ============================================================
def md_to_text(md_path: str) -> str:
    print(f"[INFO] Reading {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    html = markdown.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ============================================================
# ⭐ 2️⃣ Chunking logic
# ============================================================
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_and_chunk_md(folder_path: str):
    start_time = time.time()
    all_chunks = []
    metadata = []

    for fname in os.listdir(folder_path):
        if fname.endswith(".md"):
            path = os.path.join(folder_path, fname)
            text = md_to_text(path)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "file": fname,
                    "chunk_id": i
                })

    elapsed = time.time() - start_time
    print(f"[INFO] Created {len(all_chunks)} chunks in {elapsed:.2f}s")
    return all_chunks, metadata


# ============================================================
# 3️⃣ Embeddings & Semantic Search
# ============================================================
def create_embeddings(texts):
    start_time = time.time()
    print("[INFO] Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[INFO] Creating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    elapsed = time.time() - start_time
    print(f"[INFO] Embeddings created in {elapsed:.2f}s")
    return model, embeddings


def search(query, texts, model, embeddings, top_k=3):
    start_time = time.time()
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)

    results = []
    for hit in hits[0]:
        idx = hit["corpus_id"]
        results.append(texts[idx])

    elapsed = time.time() - start_time
    print(f"[INFO] Semantic search completed in {elapsed:.2f}s")
    return results


# ============================================================
# 4️⃣ Load LLaMA-2 (LOCAL)
# ============================================================
MODEL_PATH = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf")
assert os.path.exists(MODEL_PATH)

print("[INFO] Loading LLaMA-2 model...")
start_time = time.time()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count(),
    n_gpu_layers=-1,
    verbose=False
)

elapsed = time.time() - start_time
print(f"[INFO] LLaMA model loaded in {elapsed:.2f}s")


# ============================================================
# 5️⃣ Question Answering
# ============================================================
def answer_question(context_chunks, question):
    context = "\n\n".join(context_chunks)
    context = context[:3000]

    prompt = f"""[INST]
You are a helpful assistant.
Answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}
[/INST]
"""

    start_time = time.time()
    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>"]
    )
    elapsed = time.time() - start_time
    print(f"[INFO] LLaMA answered question in {elapsed:.2f}s")

    return output["choices"][0]["text"].strip()


# ============================================================
# 6️⃣ Main
# ============================================================
if __name__ == "__main__":
    total_start = time.time()
    MARKDOWN_FOLDER = "./ocr_results"

    chunks, metadata = load_and_chunk_md(MARKDOWN_FOLDER)
    embed_model, embeddings = create_embeddings(chunks)

    query = "what is the purpose of BSB number"
    top_chunks = search(query, chunks, embed_model, embeddings, top_k=3)

    answer = answer_question(top_chunks, query)
    print("\n✅ Answer:\n", answer)

    total_elapsed = time.time() - total_start
    print(f"\n[INFO] Total script execution time: {total_elapsed:.2f}s")
