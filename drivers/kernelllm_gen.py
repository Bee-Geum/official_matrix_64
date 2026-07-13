#!/usr/bin/env python3
"""
kernelllm_gen.py -- FAITHFUL reproduction driver for KernelLLM (facebook/KernelLLM, 8B).

KernelLLM is model-only (like AutoTriton). It is a COMPLETION model trained with a specific
PROMPT_TEMPLATE (shipped in the model repo's kernelllm.py) that wraps a torch Model and asks
for a Triton ModelNew. We use that exact template verbatim against a vLLM /v1/completions
endpoint (temperature 0.6, top_p 0.95, top_k 0 -- the model's own defaults), then extract the
ModelNew for official scoring.

Usage:
  python3 kernelllm_gen.py --task <KernelBench_task.py> --out-cand <dir> [--rounds 1]
"""
from __future__ import annotations
import argparse, importlib.util, os, re, glob
from pathlib import Path
from openai import OpenAI

BASE_URL = os.environ.get("KERNELLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("KERNELLLM_MODEL", "kernelllm")
MODEL_DIR_GLOB = "/home/bi_geum/.cache/huggingface/hub/models--facebook--KernelLLM/snapshots/*/kernelllm.py"


def load_prompt_template() -> str:
    path = glob.glob(MODEL_DIR_GLOB)[0]
    spec = importlib.util.spec_from_file_location("_kernelllm_card", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines PROMPT_TEMPLATE (no model load at import)
    return mod.PROMPT_TEMPLATE


COMPAT_SHIM = '''# [official_matrix_88] torch>=2.11 compat shim: restore classic inductor `grid`
# (KernelLLM emits torch-inductor-style code; `grid` was refactored out of
#  torch._inductor.runtime.triton_heuristics in newer torch. Kernel logic below is verbatim.)
import triton as _triton
import torch._inductor.runtime.triton_heuristics as _th
if not hasattr(_th, "grid"):
    def grid(*numels):
        def grid_fn(meta):
            return tuple(_triton.cdiv(n, meta[b]) for n, b in zip(numels, ("XBLOCK", "YBLOCK", "ZBLOCK")))
        return grid_fn
    _th.grid = grid
# ---- KernelLLM output (verbatim) ----
'''


def maybe_shim(code: str) -> str:
    # Only prepend when the code relies on the removed inductor `grid` import.
    if "import grid" in code and "triton_heuristics" in code:
        return COMPAT_SHIM + code
    return code


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "class ModelNew" in b:
                return b.strip()
        return max(blocks, key=len).strip()
    # KernelLLM often emits raw code; if ModelNew present, take from there
    if "class ModelNew" in text:
        return text.strip()
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-cand", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    template = load_prompt_template()
    code = Path(args.task).read_text(encoding="utf-8")
    prompt = template.format(code)
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    out = Path(args.out_cand); out.mkdir(parents=True, exist_ok=True)

    for r in range(max(1, args.rounds)):
        resp = client.completions.create(
            model=MODEL, prompt=prompt,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=0.95, extra_body={"top_k": 0},
        )
        raw = resp.choices[0].text or ""
        (out / f"round{r}_raw.txt").write_text(raw, encoding="utf-8")
        kern = extract_code(raw)
        (out / f"round{r}_kernel.py").write_text(kern, encoding="utf-8")
        print(f"[round {r}] wrote round{r}_kernel.py (ModelNew present: {'class ModelNew' in kern}, {len(kern)} chars)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
