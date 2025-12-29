from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
)
