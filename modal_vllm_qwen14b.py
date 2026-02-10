import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.13.0", "huggingface-hub==0.36.0")
)

app = modal.App("estebanf-llm-14b")

@app.function(
    image=image,
    gpu="L40S",
    timeout=30 * 60,
    scaledown_window=3 * 60,
    min_containers=0
)
@modal.web_server(port=PORT, startup_timeout=30 * 60, requires_proxy_auth=True)
def serve():
    import os, subprocess

    # Helps prevent allocator fragmentation on tight VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),

        # L40S reliability knobs for 14B model
        "--dtype", "half",
        "--max-model-len", "32768",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.80",
        "--enforce-eager",

        # ✅ REQUIRED for OpenCode (tool_choice="auto")
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",

        "--served-model-name", MODEL,
    ]

    subprocess.Popen(cmd)
