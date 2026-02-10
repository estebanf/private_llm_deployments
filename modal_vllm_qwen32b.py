import modal

MODEL = "Qwen/Qwen2.5-32B-Instruct"
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.13.0", "huggingface-hub==0.36.0")
)

app = modal.App("estebanf-llm-32b")

@app.function(
    image=image,
    gpu="L40S:2",  # 2x L40S with tensor parallelism
    timeout=30 * 60,
    scaledown_window=3 * 60,
    min_containers=0
)
@modal.web_server(port=PORT, startup_timeout=30 * 60, requires_proxy_auth=True)
def serve():
    import os, subprocess

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),

        # 2x L40S settings (96GB total)
        "--tensor-parallel-size", "2",
        "--dtype", "half",
        "--max-model-len", "32768",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",

        # Tool calling (Hermes format - works with Qwen)
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",

        "--served-model-name", MODEL,
    ]

    subprocess.Popen(cmd)
