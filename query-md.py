import os
import markdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ----------------------------
# 1️⃣ Load and preprocess Markdown
# ----------------------------
def md_to_text(md_path):
    print(f"[INFO] Reading Markdown file: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return text


def load_md_files(folder_path):
    print(f"[INFO] Loading Markdown files from folder: {folder_path}")
    texts = []
    filenames = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".md"):
            full_path = os.path.join(folder_path, fname)
            text = md_to_text(full_path)
            texts.append(text)
            filenames.append(fname)
    print(f"[INFO] Loaded {len(texts)} Markdown files")
    return texts, filenames

# ----------------------------
# 2️⃣ Embed the Markdown for semantic search
# ----------------------------
def create_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[INFO] Creating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    print("[INFO] Embeddings created")
    return model, embeddings


def search(query, texts, model, embeddings, top_k=1):
    print(f"[INFO] Searching for query: {query}")
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)
    results = []
    for hit in hits[0]:
        idx = hit['corpus_id']
        score = hit['score']
        results.append((texts[idx], score))
    print(f"[INFO] Found {len(results)} relevant results")
    return results

# ----------------------------
# 3️⃣ Local LLM for Q&A (MPS compatible)
# ----------------------------
def answer_question(context, question,
                    model_name="NousResearch/Llama-2-7b-chat-hf"):
    print(f"[INFO] Loading LLM model: {model_name}")

    # Use MPS (Apple GPU) and float32 for stability
    device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32  # ✅ use float32 instead of float16
    ).to(device)

    print("[INFO] Generating answer from LLM...")
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ✅ disable sampling to avoid NaNs
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    print("[INFO] Answer generation complete")
    return answer

# ----------------------------
# 4️⃣ Main flow
# ----------------------------
if __name__ == "__main__":
    folder = "./ocr_results"  # folder with your .md files
    texts, filenames = load_md_files(folder)

    # create embeddings
    embed_model, embeddings = create_embeddings(texts)

    # Example query
    query = "For Facilities Building how many minimum number of ropes are allowed? and give me the lift number for it also"
    results = search(query, texts, embed_model, embeddings, top_k=1)

    if results:
        context, score = results[0]
        print(f"[INFO] Context found (score {score:.2f}):\n{context}\n")
        answer = answer_question(context, query)
        print("Answer:", answer)
    else:
        print("[INFO] No relevant context found.")
