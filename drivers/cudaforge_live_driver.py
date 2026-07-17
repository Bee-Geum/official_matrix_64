#!/usr/bin/env python3
"""Run the real CudaForge agent (third_party/CudaForge/main.py).

Replaces the previous shim that only re-prompted a generic LLM. This invokes
CudaForge's own multi-round agent loop, including its NCU-guided optimization
pass, and hands back the kernel it selected.

Contract expected by official_all_matrix_v1.py:
    --task <task.py> --cand_dir <dir> [--rounds N] [--seed S] [--temperature T]

CudaForge's own contract differs, so this maps between them:
  - task .py            -> positional arch_py
  - --cand_dir          -> --work_dir, then its timestamped child is harvested
  - kernel selection    -> summary.json's best_code_path

NCU is mandatory on CudaForge's optimization path (run_ncu.py resolves it via
shutil.which and hard-exits on failure), so cuda/bin is put on PATH here rather
than relying on the caller's environment.
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
REPO = ROOT / "third_party" / "CudaForge"
CUDA_BIN = "/usr/local/cuda/bin"


def gpu_name() -> str:
    """Map the driver's GPU name onto a key CudaForge's GPU_SPEC_INFO knows.

    build_seed_prompt looks the name up in prompts/hardware/gpu_specs.py and
    raises KeyError on a miss, so the raw nvidia-smi string ("NVIDIA H100 80GB
    HBM3") cannot be passed through. Keys available upstream: L40S, H100, A100,
    A100-80GB, L4, Quadro RTX 6000, T4, A10G.
    """
    override = os.environ.get("CUDAFORGE_GPU")
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
    print(f"[cudaforge] WARNING: GPU {raw!r} has no GPU_SPEC_INFO entry; "
          f"using H100 spec in the prompt", file=sys.stderr)
    return "H100"


def newest_run_dir(work_dir: Path, before: set[Path]) -> Path | None:
    """CudaForge appends a {stamp}_{stem}_{tag} child under --work_dir."""
    made = [p for p in work_dir.iterdir() if p.is_dir() and p not in before]
    if not made:
        return None
    return max(made, key=lambda p: p.stat().st_mtime)


def harvest(run_dir: Path, out_dir: Path) -> tuple[bool, str]:
    """Copy the kernel CudaForge picked into the candidate dir."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False, "CudaForge wrote no summary.json"
    try:
        summary = json.loads(summary_path.read_text(errors="ignore"))
    except Exception as exc:
        return False, f"unreadable summary.json: {exc!r}"

    tasks = summary.get("tasks") or []
    if not tasks:
        return False, "summary.json has no tasks"
    task = tasks[0]

    # CudaForge reports its own verdict. If it never produced a runnable kernel,
    # scraping the newest *kernel*.py off disk hands the benchmark's oracle a
    # crash artifact -- which then scores as a real (failing) candidate and makes
    # the cell look like "the agent ran and was wrong" instead of "the agent
    # never ran". Refuse instead.
    if not task.get("best_runnable", False):
        return False, (f"CudaForge produced no runnable kernel "
                       f"(accuracy={summary.get('accuracy')}, "
                       f"best_score={task.get('best_score')}); refusing to harvest "
                       f"a crash artifact")

    best = task.get("best_code_path") or task.get("best_kernel_path") or ""
    src = Path(best) if best else None
    if src is None or not src.exists():
        cands = sorted(run_dir.rglob("*kernel*.py"), key=lambda p: p.stat().st_mtime)
        src = cands[-1] if cands else None
    if src is None or not src.exists():
        return False, f"no kernel file found under {run_dir}"

    dst = out_dir / "round000_kernel.py"
    dst.write_text(src.read_text(errors="ignore"), errors="ignore")

    (out_dir / "cudaforge_io").mkdir(parents=True, exist_ok=True)
    (out_dir / "cudaforge_io" / "cudaforge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    (out_dir / "cudaforge_io" / "provenance.json").write_text(json.dumps({
        "agent": "cudaforge",
        "mode": "live_upstream_agent",
        "upstream_entry": str(REPO / "main.py"),
        "run_dir": str(run_dir),
        "selected_kernel": str(src),
        "best_score": task.get("best_score"),
        "accuracy": summary.get("accuracy"),
        "ncu_on_path": shutil.which("ncu"),
    }, indent=2, ensure_ascii=False))
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=int(os.environ.get("CUDAFORGE_MAX_TOKENS", "8192")))
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.cand_dir or args.out or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not REPO.exists():
        print(f"[cudaforge] ERROR: upstream repo missing at {REPO}", file=sys.stderr)
        return 2
    if not task.exists():
        print(f"[cudaforge] ERROR: task not found: {task}", file=sys.stderr)
        return 2

    # NOT under out_dir: the runner's candidate_files() rglobs the candidate dir for
    # *.py, so CudaForge's scratch tree living inside it would let a crash artifact be
    # picked up as a candidate even when harvest() refuses to emit one -- which is
    # exactly how this driver produced a false PASS on multikernelbench.
    work_dir = out_dir.parent / "cudaforge_run"
    work_dir.mkdir(parents=True, exist_ok=True)
    before = {p for p in work_dir.iterdir() if p.is_dir()}

    base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    host_port = base.split("//", 1)[-1].split("/v1")[0]
    address, _, port = host_port.partition(":")

    env = dict(os.environ)
    # run_ncu.py: ncu_bin = shutil.which("ncu") or "/usr/bin/ncu", then hard-exits.
    env["PATH"] = f"{CUDA_BIN}:{env.get('PATH', '')}"
    env.setdefault("OPENAI_API_KEY", "EMPTY")

    cmd = [
        sys.executable, str(REPO / "main.py"), str(task),
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
    print("[cudaforge] LIVE upstream agent:", " ".join(cmd), flush=True)

    start = time.time()
    # cwd must be the repo: NCU's subprocess resolves its inputs against Path.cwd(),
    # and the scratch ref_0.py / test_kernel_0.py live at the repo root.
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    wall = time.time() - start
    print(f"[cudaforge] upstream rc={proc.returncode} wall={wall:.1f}s", flush=True)

    run_dir = newest_run_dir(work_dir, before)
    if run_dir is None:
        print("[cudaforge] ERROR: CudaForge produced no run directory", file=sys.stderr)
        return 1

    ok, err = harvest(run_dir, out_dir)
    if not ok:
        print(f"[cudaforge] ERROR: {err}", file=sys.stderr)
        return 1

    print(f"[cudaforge] wrote {out_dir / 'round000_kernel.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
