#!/usr/bin/env python3
"""Run the real KernelMem agent (third_party/KernelMem/main_memory_latest.py).

Replaces the shim that only re-prompted a generic LLM. KernelMem is a CudaForge
fork plus a memory bank: two static committed files
(memorybank/bottleneck_headroom_kernelstructure.yaml and
memorybank/gate_value_from_kernel_struct) that are read by path and injected as
prompt text. There is no vector store and no bootstrap, so it runs cold.

Contract expected by official_all_matrix_v1.py:
    --task <task.py> --out <dir> [--rounds N] [--seed S] [--temperature T]

Two upstream defaults are wrong for this box and are overridden here rather than
patched: --server_type openai (sends reasoning_effort, which vLLM rejects) and
--max_tokens 16384. The author-machine NCU paths in run_ncu_memory.py *were*
patched -- upstream swallows an NCU failure and silently turns the whole
optimization phase into a no-op that still reports success.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "third_party" / "KernelMem"
CUDA_BIN = "/usr/local/cuda/bin"


def gpu_name() -> str:
    """Map onto a key in KernelMem's GPU_SPEC_INFO (same table as CudaForge)."""
    override = os.environ.get("KERNELMEM_GPU")
    if override:
        return override
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=30,
        )
        raw = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        raw = ""
    for key in ("A100-80GB", "H100", "A100", "L40S", "L4", "T4", "A10G", "Quadro RTX 6000"):
        if key.replace("-", " ").lower() in raw.replace("-", " ").lower():
            return key
    return "H100"


def newest_run_dir(work_dir: Path, before: set[Path]) -> Path | None:
    made = [p for p in work_dir.iterdir() if p.is_dir() and p not in before]
    return max(made, key=lambda p: p.stat().st_mtime) if made else None


def upstream_ran(run_dir: Path) -> tuple[bool, dict]:
    """Did KernelMem actually produce a runnable kernel, by its own account?

    KernelMem takes a KernelBench model .py; the runner hands non-KernelBench
    benchmarks an official_prompt.txt, which it cannot import, so it crashes at
    seed and still leaves kernel_*.py debris on disk. Scraping that debris and
    feeding it to the benchmark's oracle turns "the agent never ran" into a
    scored cell -- and it has already produced a false PASS this way.
    """
    info: dict = {}
    summaries = sorted(run_dir.rglob("summary.json"))
    for path in summaries:
        try:
            summary = json.loads(path.read_text(errors="ignore"))
        except Exception:
            continue
        item = summary[0] if isinstance(summary, list) and summary else summary
        if not isinstance(item, dict):
            continue
        info["summary"] = str(path)
        info["accuracy"] = item.get("accuracy")
        tasks = item.get("tasks") or []
        task = tasks[0] if tasks else item
        info["best_runnable"] = task.get("best_runnable")
        info["best_score"] = task.get("best_score")
        return bool(task.get("best_runnable")), info
    info["summary"] = "none found"
    return False, info


def select_kernel(run_dir: Path) -> tuple[Path | None, dict]:
    """KernelMem records its choice in optimization_tree.json, not summary.json."""
    info: dict = {}
    trees = list(run_dir.rglob("optimization_tree.json"))
    if trees:
        try:
            tree = json.loads(trees[0].read_text(errors="ignore"))
            info["optimization_tree"] = str(trees[0])
            best = tree.get("best_kernel") or tree.get("best") or ""
            if isinstance(best, dict):
                best = best.get("code_path") or best.get("path") or ""
            if best and Path(best).exists():
                return Path(best), info
        except Exception as exc:
            info["tree_error"] = repr(exc)

    kernels = sorted(run_dir.rglob("kernel_*.py"), key=lambda p: p.stat().st_mtime)
    return (kernels[-1] if kernels else None), info


def ncu_evidence(run_dir: Path) -> dict:
    """Upstream swallows NCU failures, so verify the phase actually produced output."""
    profiles = [p for p in run_dir.rglob("*") if p.is_dir() and p.name == "profile"]
    profile_files = [str(p) for d in profiles for p in d.iterdir()] if profiles else []
    opt_rounds = [str(p) for p in run_dir.rglob("opt_round_*.json")]
    errors = [str(p) for p in run_dir.rglob("*_ncu_error.csv")]
    return {
        "profile_dirs": [str(p) for p in profiles],
        "profile_files": profile_files[:20],
        "opt_round_files": opt_rounds,
        "ncu_error_csvs": errors,
        "ncu_phase_ran": bool(profile_files or opt_rounds),
    }


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=int(os.environ.get("KERNELMEM_MAX_TOKENS", "8192")))
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.out or args.cand_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not REPO.exists():
        print(f"[kernelskill] ERROR: upstream repo missing at {REPO}", file=sys.stderr)
        return 2
    if not task.exists():
        print(f"[kernelskill] ERROR: task not found: {task}", file=sys.stderr)
        return 2

    # NOT under out_dir: the runner's candidate_files() rglobs the candidate dir for
    # *.py, so KernelMem's scratch tree living inside it would let a crash artifact be
    # picked up as a candidate even when this driver refuses to emit one.
    work_dir = out_dir.parent / "kernelmem_run"
    work_dir.mkdir(parents=True, exist_ok=True)
    before = {p for p in work_dir.iterdir() if p.is_dir()}

    base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    host_port = base.split("//", 1)[-1].split("/v1")[0]
    address, _, port = host_port.partition(":")

    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:{env.get('PATH', '')}"
    env.setdefault("OPENAI_API_KEY", "EMPTY")

    cmd = [
        sys.executable, str(REPO / "main_memory_latest.py"), str(task),
        "--server_type", "vllm",
        "--server_address", address or "127.0.0.1",
        "--server_port", port or "8000",
        "--model_name", os.environ.get("EVAL_MODEL", "qwen14b"),
        "--gpu", gpu_name(),
        "--round", str(max(1, args.rounds)),
        "--work_dir", str(work_dir),
        "--device", "0",
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--subproc_id", "0",
    ]
    print("[kernelskill] LIVE upstream agent:", " ".join(cmd), flush=True)

    start = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    wall = time.time() - start
    print(f"[kernelskill] upstream rc={proc.returncode} wall={wall:.1f}s", flush=True)

    run_dir = newest_run_dir(work_dir, before)
    if run_dir is None:
        print("[kernelskill] ERROR: KernelMem produced no run directory", file=sys.stderr)
        return 1

    ran, run_info = upstream_ran(run_dir)
    if not ran:
        print(f"[kernelskill] ERROR: KernelMem produced no runnable kernel "
              f"({run_info}); refusing to harvest a crash artifact", file=sys.stderr)
        return 1

    src, tree_info = select_kernel(run_dir)
    if src is None:
        print(f"[kernelskill] ERROR: no kernel found under {run_dir}", file=sys.stderr)
        return 1
    tree_info.update(run_info)

    dst = out_dir / "candidate_0000.py"
    dst.write_text(src.read_text(errors="ignore"), errors="ignore")

    evidence = ncu_evidence(run_dir)
    if not evidence["ncu_phase_ran"]:
        # Not fatal: seed-only runs (--rounds 1) never reach the optimization phase.
        print("[kernelskill] WARNING: no NCU profile output found -- the optimization "
              "phase did not run (upstream swallows NCU errors silently)",
              file=sys.stderr)

    io_dir = out_dir / "kernelskill_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    (io_dir / "provenance.json").write_text(json.dumps({
        "agent": "kernelskill",
        "mode": "live_upstream_agent",
        "upstream_entry": str(REPO / "main_memory_latest.py"),
        "memorybank": [
            str(REPO / "memorybank" / "bottleneck_headroom_kernelstructure.yaml"),
            str(REPO / "memorybank" / "gate_value_from_kernel_struct"),
        ],
        "run_dir": str(run_dir),
        "selected_kernel": str(src),
        "ncu_on_path": shutil.which("ncu"),
        "ncu_evidence": evidence,
        "upstream_rc": proc.returncode,
        **tree_info,
    }, indent=2, ensure_ascii=False))

    print(f"[kernelskill] wrote {dst} (ncu_phase_ran={evidence['ncu_phase_ran']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
