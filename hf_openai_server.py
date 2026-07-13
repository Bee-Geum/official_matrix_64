
#!/usr/bin/env python3
"""
Minimal OpenAI-compatible HuggingFace local server.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/completions

Designed for unified_bench experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = None
TOKENIZER = None
MODEL_ID = ""
MODEL_ALIAS = ""
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_LOCK = asyncio.Lock()

CODE_ONLY_SYSTEM_PROMPT = (
    "Return only valid Python source code. "
    "The code must define class ModelNew(torch.nn.Module). "
    "Do not include markdown fences, explanations, or example usage."
)


def choose_dtype(dtype_name: str):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16

    # auto
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text" and "content" in item:
                    parts.append(str(item["content"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        cleaned.append({"role": role, "content": normalize_content(msg.get("content", ""))})

    # Optional compatibility knob used by earlier unified_bench scripts.
    if os.environ.get("FORCE_CODE_ONLY_SYSTEM_PROMPT", "0") == "1":
        cleaned = [{"role": "system", "content": CODE_ONLY_SYSTEM_PROMPT}] + cleaned

    if not cleaned:
        cleaned = [{"role": "user", "content": ""}]

    return cleaned


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def apply_chat_template(messages: List[Dict[str, str]]) -> str:
    assert TOKENIZER is not None

    try:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Generic fallback for models without chat template.
        chunks = []
        for message in messages:
            chunks.append(f"{message['role'].upper()}:\n{message['content']}")
        chunks.append("ASSISTANT:\n")
        return "\n\n".join(chunks)


def trim_stop(text: str, stop: Optional[Any]) -> str:
    if stop is None:
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = len(text)
    for s in stops:
        if not s:
            continue
        idx = text.find(str(s))
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[Any] = None,
) -> Dict[str, Any]:
    assert TOKENIZER is not None
    assert MODEL is not None

    inputs = TOKENIZER(prompt, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    do_sample = temperature is not None and float(temperature) > 0

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        pad_token_id=TOKENIZER.eos_token_id,
        eos_token_id=TOKENIZER.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    start = time.time()
    with torch.inference_mode():
        output_ids = MODEL.generate(**inputs, **gen_kwargs)
    latency = time.time() - start

    new_ids = output_ids[0, input_len:]
    text = TOKENIZER.decode(new_ids, skip_special_tokens=True)
    text = trim_stop(text, stop)

    completion_tokens = int(new_ids.shape[-1])
    return {
        "text": text,
        "prompt_tokens": input_len,
        "completion_tokens": completion_tokens,
        "total_tokens": input_len + completion_tokens,
        "latency_s": latency,
    }


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Any = ""
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    stream: Optional[bool] = False


app = FastAPI(title="HF OpenAI-compatible server")


@app.get("/health")
def health():
    gpu = {}
    if torch.cuda.is_available():
        try:
            gpu = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
            }
        except Exception as exc:
            gpu = {"error": repr(exc)}

    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_alias": MODEL_ALIAS,
        "device": DEVICE,
        "gpu": gpu,
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ALIAS,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            },
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            },
        ],
    }


async def one_shot_sse(payload: Dict[str, Any], is_chat: bool = True):
    # Lightweight stream compatibility: send one chunk and [DONE].
    import json

    if is_chat:
        content = payload["choices"][0]["message"]["content"]
        chunk = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        final = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    else:
        content = payload["choices"][0]["text"]
        chunk = {
            "id": payload["id"],
            "object": "text_completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [{"index": 0, "text": content, "finish_reason": None}],
        }
        final = {
            "id": payload["id"],
            "object": "text_completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }

    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    max_new = req.max_new_tokens or req.max_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature
    top_p = DEFAULT_TOP_P if req.top_p is None else req.top_p

    messages = sanitize_messages(req.messages)
    prompt = apply_chat_template(messages)

    async with GEN_LOCK:
        result = await asyncio.to_thread(
            generate_text,
            prompt,
            int(max_new),
            float(temperature),
            float(top_p),
            req.stop,
        )

    model_name = req.model or MODEL_ALIAS
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
    }

    if req.stream:
        return StreamingResponse(one_shot_sse(payload, True), media_type="text/event-stream")
    return payload


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    max_new = req.max_new_tokens or req.max_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature
    top_p = DEFAULT_TOP_P if req.top_p is None else req.top_p

    prompt = normalize_content(req.prompt)

    async with GEN_LOCK:
        result = await asyncio.to_thread(
            generate_text,
            prompt,
            int(max_new),
            float(temperature),
            float(top_p),
            req.stop,
        )

    model_name = req.model or MODEL_ALIAS
    payload = {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "text": result["text"], "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
    }

    if req.stream:
        return StreamingResponse(one_shot_sse(payload, False), media_type="text/event-stream")
    return payload


def load_model(args: argparse.Namespace):
    global MODEL, TOKENIZER, MODEL_ID, MODEL_ALIAS
    global DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEVICE

    MODEL_ID = args.model_id
    MODEL_ALIAS = args.model_alias or args.model_name or os.path.basename(os.path.abspath(args.model_id.rstrip("/"))) or "local-model"
    DEFAULT_MAX_NEW_TOKENS = int(args.max_new_tokens)
    DEFAULT_TEMPERATURE = float(args.temperature)
    DEFAULT_TOP_P = float(args.top_p)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = choose_dtype(args.dtype)

    print("==========================================", flush=True)
    print("Loading model", flush=True)
    print("MODEL_ID:", MODEL_ID, flush=True)
    print("MODEL_ALIAS:", MODEL_ALIAS, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("DTYPE:", dtype, flush=True)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)
    print("==========================================", flush=True)

    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if TOKENIZER.pad_token_id is None and TOKENIZER.eos_token_id is not None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as exc:
            print(f"[WARN] --use_4bit requested but unavailable: {exc}", flush=True)

    if DEVICE == "cuda":
        kwargs["device_map"] = args.device_map
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = torch.float32

    try:
        MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    except TypeError as exc:
        # Newer/older transformers compatibility.
        message = str(exc)
        if "torch_dtype" in message:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
            MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
        else:
            raise

    MODEL.eval()
    print("==========================================", flush=True)
    print("Model loaded", flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("mem_allocated:", torch.cuda.memory_allocated(), flush=True)
        print("mem_reserved:", torch.cuda.memory_reserved(), flush=True)
    print("==========================================", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", "--model-id", dest="model_id", required=True)
    parser.add_argument("--model_alias", "--model-alias", dest="model_alias", default=None)
    parser.add_argument("--model_name", "--model-name", dest="model_name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", "--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device_map", "--device-map", dest="device_map", default="auto")
    parser.add_argument("--trust_remote_code", "--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--use_4bit", "--use-4bit", dest="use_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    load_model(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
