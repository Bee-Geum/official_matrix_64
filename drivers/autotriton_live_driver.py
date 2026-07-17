#!/usr/bin/env python3
"""Run the real AutoTriton model (drivers/autotriton_gen.py).

Replaces the shim (drivers/trained_model.py) that routed autotriton into the
generic LLM driver against Qwen -- i.e. the run dirs said "autotriton8b" while
Qwen did the work. AutoTriton ships no code (its repo is a README + figures);
the artifact IS the RL-trained 8B model, so running it means prompting
ai9stars/AutoTriton itself.

Maps the matrix runner's contract onto autotriton_gen.py:
    --task <task.py> --cand_dir <dir> [--rounds N] [--seed S] [--temperature T]

The model is served separately from the shared Qwen endpoint; point
AUTOTRITON_BASE_URL / AUTOTRITON_MODEL at it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "drivers" / "autotriton_gen.py"

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "autotriton8b"


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # AutoTriton is a reasoning model; 0.6 is its documented sampling temperature.
    ap.add_argument("--temperature", type=float, default=0.6)
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.cand_dir or args.out or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not task.exists():
        print(f"[autotriton] ERROR: task not found: {task}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    base_url = env.setdefault("AUTOTRITON_BASE_URL", DEFAULT_BASE_URL)
    model = env.setdefault("AUTOTRITON_MODEL", DEFAULT_MODEL)

    cmd = [
        sys.executable, str(GEN),
        "--task", str(task),
        "--out-cand", str(out_dir),
        "--rounds", str(max(1, args.rounds)),
        "--temperature", str(args.temperature),
    ]
    print(f"[autotriton] LIVE model {model} @ {base_url}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)

    produced = sorted(out_dir.glob("round*_kernel.py"))
    if not produced:
        print("[autotriton] ERROR: model produced no kernel", file=sys.stderr)
        return 1

    io_dir = out_dir / "autotriton_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    (io_dir / "provenance.json").write_text(json.dumps({
        "agent": "autotriton",
        "mode": "model_only",
        "model_repo": "ai9stars/AutoTriton",
        "served_as": model,
        "base_url": base_url,
        "generator": str(GEN),
        "rounds": max(1, args.rounds),
        "temperature": args.temperature,
        "gen_rc": proc.returncode,
        "kernels": [str(p) for p in produced],
    }, indent=2, ensure_ascii=False))

    print(f"[autotriton] wrote {len(produced)} kernel(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
