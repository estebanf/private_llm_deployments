import modal

MODEL = "zai-org/GLM-4.7-Flash"
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly",
        "pip install git+https://github.com/huggingface/transformers.git",
    )
    .pip_install("huggingface-hub")
)

app = modal.App("estebanf-llm-glm47-flash")

@app.function(
    image=image,
    gpu="L40S:4",
    timeout=30 * 60,
    scaledown_window=3 * 60,
    min_containers=0,
    max_containers=1,
)
@modal.web_server(port=PORT, startup_timeout=30 * 60, requires_proxy_auth=True)
def serve():
    import os, subprocess

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),

        # 4x L40S settings (192GB total)
        "--tensor-parallel-size", "4",
        "--dtype", "half",
        "--max-model-len", "65536",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.85",

        # Tool calling
        "--enable-auto-tool-choice",
        "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",

        "--disable-frontend-multiprocessing",
        "--served-model-name", MODEL,
    ]

    subprocess.Popen(cmd)
