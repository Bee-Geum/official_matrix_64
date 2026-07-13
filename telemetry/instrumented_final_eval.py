#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def env(key, default):
    return os.environ.get(key, default)


def input_seed(root_seed: int, task_path: str) -> int:
    digest = hashlib.sha256(f"{root_seed}:{task_path}".encode()).hexdigest()
    return int(digest[:8], 16)


def pass_at_k(n, c, k):
    if n == 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--cand_dir", required=True)
    parser.add_argument("--glob", default="round*_kernel.py")
    parser.add_argument("--task_work_dir", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.cand_dir)
    work_dir = Path(args.task_work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(candidate_dir.glob(args.glob))
    if not candidates and args.glob != "*.py":
        candidates = sorted(candidate_dir.glob("*.py"))

    seed_base = input_seed(int(env("ROOT_SEED", "20260611")), args.task)
    timeout = int(env("EVAL_TIMEOUT_SEC", "600"))

    verdicts = []
    evaluation_start = time.perf_counter()

    for index, candidate_path in enumerate(candidates):
        output_path = work_dir / f"verdict_{index:03d}.json"
        job = {
            "ref_path": args.task,
            "cand_path": str(candidate_path),
            "trials": int(env("NUM_CORRECTNESS_TRIALS", "5")),
            "atol": float(env("ATOL", "1e-2")),
            "rtol": float(env("RTOL", "1e-2")),
            "warmup": int(env("EVAL_WARMUP", "10")),
            "timing_iters": int(env("EVAL_TIMING_ITERS", "100")),
            "input_seed_base": seed_base,
            "out_path": str(output_path),
        }
        job_path = work_dir / f"job_{index:03d}.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")

        start = time.perf_counter()
        stdout = b""
        stderr = b""
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "instrumented_eval_worker.py"),
                    str(job_path),
                ],
                timeout=timeout,
                capture_output=True,
            )
            stdout = process.stdout
            stderr = process.stderr
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""

        if stdout:
            (work_dir / f"worker_{index:03d}.stdout.txt").write_bytes(stdout)
        if stderr:
            (work_dir / f"worker_{index:03d}.stderr.txt").write_bytes(stderr)

        if output_path.exists():
            verdict = json.loads(output_path.read_text())
        else:
            verdict = {
                "compiled": False,
                "correct": False,
                "latency_ms": None,
                "ref_latency_ms": None,
                "speedup": None,
                "error": "timeout/crash",
                "stage_s": {"total": time.perf_counter() - start},
                "compile_attempts": 1,
                "profile_runs": 0,
                "correctness_trials_executed": 0,
            }

        verdict.update({
            "candidate": str(candidate_path),
            "candidate_index": index,
            "eval_wall_s": time.perf_counter() - start,
        })
        verdicts.append(verdict)
        print(
            f"[{index + 1}/{len(candidates)}] "
            f"compiled={verdict.get('compiled')} "
            f"correct={verdict.get('correct')} "
            f"speedup={verdict.get('speedup')}"
        )

    compiled = [v for v in verdicts if v.get("compiled")]
    correct = [v for v in verdicts if v.get("correct")]
    speeds = [
        float(v["speedup"])
        for v in correct
        if v.get("speedup") and float(v["speedup"]) > 0
    ]
    n = len(verdicts)
    first_compiled = next((v["candidate_index"] for v in verdicts if v.get("compiled")), None)
    first_correct = next((v["candidate_index"] for v in verdicts if v.get("correct")), None)
    first_faster = next(
        (
            v["candidate_index"]
            for v in verdicts
            if v.get("correct") and (v.get("speedup") or 0) > 1
        ),
        None,
    )

    stage_totals = {
        "compile_s": sum((v.get("stage_s") or {}).get("candidate_compile_import", 0.0) for v in verdicts),
        "correctness_s": sum((v.get("stage_s") or {}).get("correctness", 0.0) for v in verdicts),
        "benchmark_s": sum(
            (v.get("stage_s") or {}).get("reference_benchmark", 0.0)
            + (v.get("stage_s") or {}).get("candidate_benchmark", 0.0)
            for v in verdicts
        ),
        "model_init_s": sum((v.get("stage_s") or {}).get("model_init", 0.0) for v in verdicts),
    }

    best_speedup = max(speeds) if speeds else 0.0
    summary = [{
        "task": args.task,
        "task_dir": str(work_dir),
        "n_candidates": n,
        "n_compiled": len(compiled),
        "n_correct": len(correct),
        "runnable_rate": len(compiled) / n if n else 0.0,
        "correct_rate": len(correct) / n if n else 0.0,
        "pass@1": 1.0 if n and verdicts[0].get("correct") else 0.0,
        "pass@k_analytic": pass_at_k(n, len(correct), 1),
        "fast_1": (
            1.0
            if n
            and verdicts[0].get("correct")
            and (verdicts[0].get("speedup") or 0) > 1
            else 0.0
        ),
        "best_score": best_speedup,
        "best_runnable": bool(compiled),
        "geomean_speedup": statistics.geometric_mean(speeds) if speeds else 0.0,
        "median_speedup": statistics.median(speeds) if speeds else 0.0,
        "compile_attempts": sum(int(v.get("compile_attempts", 1)) for v in verdicts),
        "correctness_trials_executed": sum(int(v.get("correctness_trials_executed", 0)) for v in verdicts),
        "profile_runs": sum(int(v.get("profile_runs", 0)) for v in verdicts),
        "first_compiled_candidate_index": first_compiled,
        "first_correct_candidate_index": first_correct,
        "first_faster_candidate_index": first_faster,
        "unified_eval_wall_s": time.perf_counter() - evaluation_start,
        **stage_totals,
    }]

    (work_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary -> {work_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
