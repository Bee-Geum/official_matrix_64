#!/usr/bin/env python3
"""
autotriton_gen.py -- FAITHFUL reproduction driver for AutoTriton.

AutoTriton (AI9Stars, arXiv:2507.05687) is NOT an agent framework -- its GitHub repo ships
only a README; the artifact IS the RL-trained 8B model (ai9stars/AutoTriton, based on
Seed-Coder-8B-Reasoning). So "running AutoTriton" = prompting that model to emit a Triton
kernel. This driver serves that exact model (via its own vLLM endpoint) and prompts it with a
KernelBench task to produce a Triton `ModelNew`, then extracts the code for official scoring.

Usage:
  python3 autotriton_gen.py --task <KernelBench_task.py> --out-cand <dir> [--rounds 1]
"""
from __future__ import annotations
import argparse, os, re
from pathlib import Path
from openai import OpenAI

BASE_URL = os.environ.get("AUTOTRITON_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("AUTOTRITON_MODEL", "autotriton")


def extract_code(text: str) -> str:
    # reasoning models emit <think>...</think> then the answer; strip think blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "class ModelNew" in b:
                return b.strip()
        return max(blocks, key=len).strip()
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-cand", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    ref = Path(args.task).read_text(encoding="utf-8")
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    out = Path(args.out_cand); out.mkdir(parents=True, exist_ok=True)

    prompt = f"""Use the Triton language to write an optimized GPU kernel for the following PyTorch model.
Produce a COMPLETE, self-contained Python module that defines a class `ModelNew(torch.nn.Module)`
whose output numerically matches the reference `Model` (atol=rtol=1e-2). Implement the compute with
`@triton.jit` kernels. Return one Python code block containing the full module.

Reference model:
```python
{ref}
```"""

    for r in range(max(1, args.rounds)):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.95,
        )
        raw = resp.choices[0].message.content or ""
        (out / f"round{r}_raw.txt").write_text(raw, encoding="utf-8")
        code = extract_code(raw)
        path = out / f"round{r}_kernel.py"
        path.write_text(code, encoding="utf-8")
        has = "class ModelNew" in code
        print(f"[round {r}] wrote {path} (ModelNew present: {has}, {len(code)} chars)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
