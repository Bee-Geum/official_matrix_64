#!/usr/bin/env python3
"""
canonical_summary.py -- the one summary.json schema every benchmark must emit.

The existing collector (a100_campaign_tool.py / collect_results.py) reads
runs/<run>/<task>__rep<k>/final_eval/summary.json as a LIST of per-task items.
For the matrix to aggregate KernelBench, BackendBench, TritonBench, ParEval, ...
in a single sheet, every benchmark's eval adapter must write that same shape.

Item schema (one dict per evaluated task):
    task            str   benchmark-relative task id (path or op name)
    n_candidates    int   how many candidate files were scored
    n_compiled      int   candidates that built / imported
    n_correct       int   candidates that matched the reference within tol
    runnable_rate   float n_compiled / n_candidates
    correct_rate    float n_correct  / n_candidates
    pass@1          float P(a single sampled candidate is correct)
    fast_1          float P(correct AND speedup > 1.0)  (KernelBench fast_p, p=1)
    best_score      float best speedup among correct candidates (0 if none)
    geomean_speedup float geomean speedup of correct candidates (0 if none)
    best_runnable   bool  did at least one candidate run
    error           str   short signature for the dominant failure, or ""

Use build_item() to construct items from a list of per-candidate verdicts so the
metric math is identical across benchmarks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def build_item(task: str, verdicts: list[dict], error: str = "") -> dict:
    """
    verdicts: list of per-candidate dicts, each with keys:
        compiled : bool
        correct  : bool
        speedup  : float | None   (reference_time / candidate_time)
    """
    n = len(verdicts)
    n_compiled = sum(1 for v in verdicts if v.get("compiled"))
    n_correct = sum(1 for v in verdicts if v.get("correct"))
    correct_speeds = [
        float(v["speedup"])
        for v in verdicts
        if v.get("correct") and v.get("speedup")
    ]
    fast1 = sum(
        1 for v in verdicts
        if v.get("correct") and v.get("speedup") and float(v["speedup"]) > 1.0
    )
    return {
        "task": task,
        "n_candidates": n,
        "n_compiled": n_compiled,
        "n_correct": n_correct,
        "runnable_rate": (n_compiled / n) if n else 0.0,
        "correct_rate": (n_correct / n) if n else 0.0,
        "pass@1": (n_correct / n) if n else 0.0,
        "fast_1": (fast1 / n) if n else 0.0,
        "best_score": max(correct_speeds) if correct_speeds else 0.0,
        "geomean_speedup": _geomean(correct_speeds),
        "best_runnable": n_compiled > 0,
        "error": "" if n_correct else error,
    }


def write_summary(path: Path, items: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def empty_item(task: str, error: str) -> dict:
    """Task that produced no usable candidate (driver crash / timeout / OOM)."""
    return build_item(task, [], error=error)
