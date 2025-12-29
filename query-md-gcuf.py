import os
import markdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

# ----------------------------
# 1️⃣ Load & preprocess Markdown
# ----------------------------
def md_to_text(md_path: str) -> str:
    print(f"[INFO] Reading {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    html = markdown.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def load_md_files(folder_path: str):
    texts, filenames = [], []
    for fname in os.listdir(folder_path):
        if fname.endswith(".md"):
            path = os.path.join(folder_path, fname)
            texts.append(md_to_text(path))
            filenames.append(fname)

    print(f"[INFO] Loaded {len(texts)} markdown files")
    return texts, filenames


# ----------------------------
# 2️⃣ Embeddings (Semantic Search)
# ----------------------------
def create_embeddings(texts):
    print("[INFO] Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[INFO] Creating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return model, embeddings


def search(query, texts, model, embeddings, top_k=1):
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)

    results = []
    for hit in hits[0]:
        idx = hit["corpus_id"]
        results.append(texts[idx])

    return results


# ----------------------------
# 3️⃣ Load LLaMA-2 GGUF (ONCE)
# ----------------------------
print("[INFO] Loading LLaMA-2 GGUF model...")

llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=os.cpu_count(),
    n_gpu_layers=-1,   # ✅ use Metal GPU fully
    verbose=False
)

print("[INFO] LLaMA model loaded")


# ----------------------------
# 4️⃣ Q&A with LLaMA-2
# ----------------------------
def answer_question(context: str, question: str) -> str:
    # Trim context for speed & stability
    context = context[:3000]

    prompt = f"""[INST]
You are a helpful assistant. Answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}
[/INST]
"""

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>"]
    )

    return output["choices"][0]["text"].strip()


# ----------------------------
# 5️⃣ Main Flow
# ----------------------------
if __name__ == "__main__":
    folder = "./ocr_results"  # folder with .md files

    texts, filenames = load_md_files(folder)

    embed_model, embeddings = create_embeddings(texts)

    query = (
        "For Facilities Building how many minimum number of ropes "
        "are allowed? and give me the lift number for it also"
    )

    results = search(query, texts, embed_model, embeddings, top_k=1)

    if results:
        answer = answer_question(results[0], query)
        print("\n✅ Answer:\n", answer)
    else:
        print("❌ No relevant context found.")
