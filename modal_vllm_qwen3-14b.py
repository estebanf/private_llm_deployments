import modal

MODEL = "Qwen/Qwen3-14B"
MODEL_DIR = "/models/Qwen3-14B"
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.8.5", "huggingface-hub==0.36.0", "flashinfer-python<0.2.3")
)

volume = modal.Volume.from_name("qwen3-14b-weights", create_if_missing=True)

app = modal.App("estebanf-llm-qwen3-14b")


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=60 * 60,
)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    volume.commit()


@app.function(
    image=image,
    volumes={"/models": volume},
)
def check_volume():
    import os
    for root, dirs, files in os.walk("/models"):
        level = root.replace("/models", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            for f in files:
                print(f"{indent}  {f}")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=30 * 60,
    scaledown_window=3 * 60,
    min_containers=0,
    max_containers=1,
)
@modal.web_server(port=PORT, startup_timeout=30 * 60, requires_proxy_auth=True)
def serve():
    import os, subprocess

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Persist FlashInfer JIT cache in the volume (avoids 29s recompile on every cold start)
    jit_cache = "/models/.torch_extensions"
    os.makedirs(jit_cache, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = jit_cache

    # Compile only for A100 (SM 8.0) instead of all CUDA archs — speeds up first JIT build
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

    cmd = [
        "vllm", "serve", MODEL_DIR,
        "--host", "0.0.0.0",
        "--port", str(PORT),

        # A100-80GB settings for 14.8B model (~30GB in BF16, ~50GB headroom for KV cache)
        "--dtype", "bfloat16",
        "--max-model-len", "32768",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",  # skip torch.compile + CUDAGraph (~270s startup savings)

        # Reasoning parser enabled but thinking off by default (too slow for most requests)
        # Enable per-request with: "chat_template_kwargs": {"enable_thinking": true}
        "--enable-reasoning",
        "--reasoning-parser", "deepseek_r1",
        "--override-generation-config", '{"enable_thinking": false}',

        # Tool calling
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",

        "--served-model-name", MODEL,
    ]

    subprocess.Popen(cmd)
