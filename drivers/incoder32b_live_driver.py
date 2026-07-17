#!/usr/bin/env python3
"""Run the real IndustrialCoder model (Multilingual-Multimodal-NLP/IndustrialCoder, 32B).

Replaces the shim that routed incoder32b into the generic LLM driver against
Qwen -- the run dirs said "incoder32b" while a 14B Qwen did the work. The
Industrial-Coder repo ships only SFT training code (no inference), so the agent
IS the model: serve it and prompt it.

Contract:
    --task <task.py> --out <dir> [--rounds N] [--seed S] [--temperature T]

The model uses a custom architecture (IQuestCoderForCausalLM) and is served
separately; point INCODER_BASE_URL / INCODER_MODEL at it. Sampling defaults
(temp 0.2 / top_p 0.95 / 4096 tokens) are the model card's "industrial/precise"
recommendation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8003/v1"
DEFAULT_MODEL = "incoder32b"


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        for block in blocks:
            if "class ModelNew" in block:
                return block.strip()
        return max(blocks, key=len).strip()
    if "class ModelNew" in text:
        starts = [text.find(m) for m in ("import torch", "class ModelNew") if text.find(m) != -1]
        if starts:
            return text[min(starts):].strip()
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.out or args.cand_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not task.exists():
        print(f"[incoder32b] ERROR: task not found: {task}", file=sys.stderr)
        return 2

    base_url = os.environ.get("INCODER_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("INCODER_MODEL", DEFAULT_MODEL)

    ref = task.read_text(errors="ignore")
    prompt = f"""Write an optimized GPU kernel implementation for the following PyTorch model.
Produce a COMPLETE, self-contained Python module defining `ModelNew(torch.nn.Module)` whose
output numerically matches the reference `Model` (atol=rtol=1e-2) and is faster than it.
Implement the compute with a custom CUDA kernel via torch.utils.cpp_extension.load_inline,
or with Triton @triton.jit kernels. Target GPU: NVIDIA H100 (sm_90).
Return one Python code block containing the full module.

Reference model:
```python
{ref}
```"""

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    print(f"[incoder32b] LIVE model {model} @ {base_url}", flush=True)

    produced: list[Path] = []
    for index in range(max(1, args.rounds)):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )
        raw = resp.choices[0].message.content or ""
        io_dir = out_dir / "incoder32b_io"
        io_dir.mkdir(parents=True, exist_ok=True)
        (io_dir / f"round{index:03d}_raw_reply.txt").write_text(raw, errors="ignore")

        code = extract_code(raw)
        path = out_dir / f"candidate_{index:04d}.py"
        path.write_text(code, errors="ignore")
        produced.append(path)
        print(f"[incoder32b] round {index}: wrote {path.name} "
              f"(ModelNew present: {'class ModelNew' in code}, {len(code)} chars)", flush=True)

    (out_dir / "incoder32b_io" / "provenance.json").write_text(json.dumps({
        "agent": "incoder32b",
        "mode": "model_only",
        "model_repo": "Multilingual-Multimodal-NLP/IndustrialCoder",
        "served_as": model,
        "base_url": base_url,
        "note": "upstream repo ships SFT training code only; the artifact is the 32B model",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "candidates": [str(p) for p in produced],
    }, indent=2, ensure_ascii=False))

    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
