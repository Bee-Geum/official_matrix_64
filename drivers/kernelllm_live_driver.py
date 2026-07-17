#!/usr/bin/env python3
"""Run the real KernelLLM model (drivers/kernelllm_gen.py).

Replaces the shim that routed kernelllm into the generic LLM driver against
Qwen -- the run dirs said "kernelllm8b" while Qwen did the work. KernelLLM has
no agent repo; the artifact is facebook/KernelLLM (8B), a *completion* model
trained with its own PROMPT_TEMPLATE, which kernelllm_gen.py uses verbatim.

Maps the matrix runner's contract onto kernelllm_gen.py:
    --task <task.py> --out <dir> [--rounds N] [--seed S] [--temperature T]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "drivers" / "kernelllm_gen.py"

DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_MODEL = "kernelllm8b"
CARD_GLOB = str(Path.home() / ".cache/huggingface/hub/models--facebook--KernelLLM/snapshots/*/kernelllm.py")


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # 0.6 / top_p 0.95 / top_k 0 are the model card's own defaults.
    ap.add_argument("--temperature", type=float, default=0.6)
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.out or args.cand_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not task.exists():
        print(f"[kernelllm] ERROR: task not found: {task}", file=sys.stderr)
        return 2
    # kernelllm_gen.py imports PROMPT_TEMPLATE out of the model repo's kernelllm.py,
    # so the weights alone are not enough -- fail loudly rather than IndexError.
    if not glob.glob(CARD_GLOB):
        print(f"[kernelllm] ERROR: kernelllm.py not found in the HF snapshot "
              f"({CARD_GLOB}); the prompt template ships with the model repo",
              file=sys.stderr)
        return 2

    env = dict(os.environ)
    base_url = env.setdefault("KERNELLLM_BASE_URL", DEFAULT_BASE_URL)
    model = env.setdefault("KERNELLLM_MODEL", DEFAULT_MODEL)

    cmd = [
        sys.executable, str(GEN),
        "--task", str(task),
        "--out-cand", str(out_dir),
        "--rounds", str(max(1, args.rounds)),
        "--temperature", str(args.temperature),
    ]
    print(f"[kernelllm] LIVE model {model} @ {base_url}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)

    produced = sorted(out_dir.glob("round*_kernel.py"))
    if not produced:
        print("[kernelllm] ERROR: model produced no kernel", file=sys.stderr)
        return 1

    # The registry globs candidate_*.py for this agent.
    for index, src in enumerate(produced):
        (out_dir / f"candidate_{index:04d}.py").write_text(
            src.read_text(errors="ignore"), errors="ignore"
        )

    io_dir = out_dir / "kernelllm_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    (io_dir / "provenance.json").write_text(json.dumps({
        "agent": "kernelllm",
        "mode": "model_only",
        "model_repo": "facebook/KernelLLM",
        "served_as": model,
        "base_url": base_url,
        "api": "/v1/completions (completion model, not chat)",
        "prompt_template": "model repo kernelllm.py PROMPT_TEMPLATE (verbatim)",
        "generator": str(GEN),
        "gen_rc": proc.returncode,
        "kernels": [str(p) for p in produced],
    }, indent=2, ensure_ascii=False))

    print(f"[kernelllm] wrote {len(produced)} kernel(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
