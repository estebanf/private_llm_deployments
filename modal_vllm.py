import json
import subprocess
import asyncio
import time
from typing import Any
import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"}) # faster model transfers
)

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
MODEL_REVISION = "953532f942706930ec4bb870569932ef63038fdf" # avoid nasty surprises when repos update!
TOOL_CALL_PARSER = "qwen3_xml"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = True

app = modal.App("example-vllm-inference")
N_GPU = 1
GPU_TYPE = "L4"
MAX_MODEL_LEN = 32768
MINUTES = 60 # seconds
VLLM_PORT = 8000

@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES, # how long should we stay up with no requests?
    timeout=10 * MINUTES, # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent( # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level", "info",

        "--enable-auto-tool-choice",
        "--tool-call-parser", TOOL_CALL_PARSER,
        "--max-model-len", str(MAX_MODEL_LEN),

        "--tensor-parallel-size", str(N_GPU),
    ]

    if FAST_BOOT:
        cmd += ["--enforce-eager"]

    print("Starting vLLM:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = serve.get_web_url()
    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."
    messages = [
        # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]
    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        # Poll /health endpoint
        start_time = time.time()
        while time.time() - start_time < test_timeout:
            try:
                async with session.get("/health", timeout=5) as resp:
                    if resp.status == 200:
                        print(f"Successful health check for server at {url}")
                        break
            except Exception:
                pass
            print("Waiting for server to be healthy...")
            await asyncio.sleep(5)
        else:
             print(f"Failed health check for server at {url}")
             return

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, MODEL_NAME, messages) # Use the constant MODEL_NAME
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, MODEL_NAME, messages)

async def _send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        # OpenCode sends tool_choice="auto"; test the same path here.
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    async with session.post(
        "/v1/chat/completions",
        json=payload,
        headers=headers
    ) as resp:
        try:
            resp.raise_for_status()
            async for raw in resp.content:
                # extract new content and stream it
                line = raw.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    # SSE prefix
                    line = line[len("data: ") :]
                try:
                    chunk = json.loads(line)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                             print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass
            print()
        except Exception as e:
            print(f"Request failed: {e}")
            print(await resp.text())
