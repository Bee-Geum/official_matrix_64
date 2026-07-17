#!/usr/bin/env python3
"""Run the real AutoKernel agent loop (drivers/autokernel_qwen_loop.py).

Replaces the shim that only re-prompted a generic LLM. AutoKernel ships no LLM
client of its own -- its README directs you to "spin up Claude, Codex, or any
coding agent in this directory" to read program.md and iteratively edit
kernel.py -- so autokernel_qwen_loop.py IS that coding-agent loop, driving
AutoKernel's own playbook, bridge, and bench_kb evaluator.

This maps the matrix runner's contract onto that loop:
    --task <task.py> --out <dir> [--rounds N] [--seed S] [--temperature T]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "drivers" / "autokernel_qwen_loop.py"
AK = ROOT / "third_party" / "autokernel"


def problem_id_of(task: Path) -> int:
    """KernelBench task files are named '<id>_<Name>.py'."""
    m = re.match(r"(\d+)_", task.name)
    return int(m.group(1)) if m else 0


PATH_SHIM = f'''# [official_matrix_64] make AutoKernel's own helpers importable outside its tree.
# AutoKernel's playbook (kernelbench/program_kb.md) tells the model to
# `from kernels.cuda._compile import compile_cuda`, which only resolves when the
# repo root is on sys.path -- true for its own bench_kb.py, false for the
# benchmark's official oracle, which runs the kernel from another directory and
# would fail at import with ModuleNotFoundError: No module named 'kernels'.
# This prepends the path only; compile_cuda itself is AutoKernel's, unmodified.
import sys as _sys
_AK_ROOT = {str(AK)!r}
if _AK_ROOT not in _sys.path:
    _sys.path.insert(0, _AK_ROOT)
# ---- AutoKernel output (verbatim) ----
'''


def maybe_shim(code: str) -> str:
    """Only prepend when the kernel actually imports AutoKernel's helpers."""
    if re.search(r"^\s*(from|import)\s+kernels[.\s]", code, flags=re.M):
        return PATH_SHIM + code
    return code


def backend_for(task: Path) -> str:
    return os.environ.get("AUTOKERNEL_BACKEND", "cuda")


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.out or args.cand_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not AK.exists():
        print(f"[autokernel] ERROR: upstream repo missing at {AK}", file=sys.stderr)
        return 2
    if not task.exists():
        print(f"[autokernel] ERROR: task not found: {task}", file=sys.stderr)
        return 2

    best = out_dir / "candidate_0000.py"
    # Outside out_dir: the runner's candidate_files() rglobs the candidate dir for *.py,
    # so a build tree inside it risks handing the oracle something that is not the
    # agent's chosen kernel.
    ext_base = out_dir.parent / "_extdir_ak"

    env = dict(os.environ)
    # The loop defaults this to 12.0 (Blackwell); this box is H100 = 9.0.
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    env.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    env.setdefault("EVAL_MODEL", "qwen14b")

    cmd = [
        sys.executable, str(LOOP),
        "--task", str(task),
        "--problem-id", str(problem_id_of(task)),
        "--iters", str(max(1, args.rounds)),
        "--backend", backend_for(task),
        "--out-best", str(best),
        "--ext-base", str(ext_base),
    ]
    print("[autokernel] LIVE upstream loop:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)

    if not best.exists():
        print("[autokernel] ERROR: loop produced no kernel", file=sys.stderr)
        return 1

    shimmed = maybe_shim(best.read_text(errors="ignore"))
    best.write_text(shimmed, errors="ignore")
    print(f"[autokernel] path shim applied: {PATH_SHIM.splitlines()[0] in shimmed}", flush=True)

    io_dir = out_dir / "autokernel_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "autokernel_loop_history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(errors="ignore"))
        except Exception:
            pass
    (io_dir / "provenance.json").write_text(json.dumps({
        "agent": "autokernel",
        "mode": "live_upstream_loop",
        "upstream_repo": str(AK),
        "playbook": str(AK / "kernelbench" / "program_kb.md"),
        "evaluator": str(AK / "kernelbench" / "bench_kb.py"),
        "loop_driver": str(LOOP),
        "backend": backend_for(task),
        "iters": max(1, args.rounds),
        "loop_rc": proc.returncode,
        "history": history,
    }, indent=2, ensure_ascii=False))

    print(f"[autokernel] wrote {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
