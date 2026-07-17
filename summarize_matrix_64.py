#!/usr/bin/env python3
"""Aggregate the 11-agent x 8-benchmark official matrix into clean tables.

Reads per-cell results written by official_all_matrix_v1.py
(<run-root>/cells/<benchmark>/<agent>/cell_result.json) and emits, into --out:
  - matrix_88_official_eval.csv   official-oracle task count per (benchmark, agent)
  - matrix_88_correct.csv         correct task count per (benchmark, agent)
  - matrix_88_summary.csv         per-benchmark rollup
  - SUMMARY.md                    human-readable report
Robust to the fact that `verify` may overwrite official_matrix_per_task.csv;
the cell_result.json files are the source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHMARKS = [
    "kernelbench", "robust_kbench", "tritonbench_t", "tritonbench_g",
    "multikernelbench", "backendbench", "pareval", "sol_execbench",
]
AGENTS = [
    "cudaforge", "autokernel", "autotriton", "drkernel",
    "kernelllm", "incoder32b", "kernelskill",
]
OFFICIAL_ORACLE = {
    "kernelbench": "kb_instrumented", "robust_kbench": "kb_instrumented",
    "tritonbench_t": "TritonBench EVAL", "tritonbench_g": "TritonBench EVAL",
    "multikernelbench": "eval_single_runner", "backendbench": "BackendBench CLI",
    "pareval": "ParEval run-all", "sol_execbench": "sol_execbench.cli",
}


def load_cell(run_root: Path, bench: str, agent: str) -> dict | None:
    f = run_root / "cells" / bench / agent / "cell_result.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(errors="ignore"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    oe = {}   # (bench, agent) -> official_eval_task_count
    cor = {}  # (bench, agent) -> correct_task_count
    nt = {}   # (bench, agent) -> n_tasks
    present = {}
    for b in BENCHMARKS:
        for a in AGENTS:
            c = load_cell(run_root, b, a)
            if c is None:
                continue
            present[(b, a)] = True
            oe[(b, a)] = int(c.get("official_eval_task_count", 0) or 0)
            cor[(b, a)] = int(c.get("correct_task_count", 0) or 0)
            nt[(b, a)] = int(c.get("n_tasks", 0) or 0)

    def write_matrix(path: Path, table: dict):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["benchmark"] + AGENTS)
            for b in BENCHMARKS:
                w.writerow([b] + [table.get((b, a), "") for a in AGENTS])

    write_matrix(out / "matrix_88_official_eval.csv", oe)
    write_matrix(out / "matrix_88_correct.csv", cor)

    # per-benchmark rollup
    rows = []
    for b in BENCHMARKS:
        ran = sum(1 for a in AGENTS if (b, a) in present)
        oe_cells = sum(1 for a in AGENTS if oe.get((b, a), 0) > 0)
        cor_cells = sum(1 for a in AGENTS if cor.get((b, a), 0) > 0)
        rows.append({
            "benchmark": b, "official_oracle": OFFICIAL_ORACLE[b],
            "agents_ran": f"{ran}/11", "official_eval_cells": f"{oe_cells}/11",
            "correct_cells": f"{cor_cells}/11",
        })
    with (out / "matrix_88_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_cells = sum(1 for b in BENCHMARKS for a in AGENTS if (b, a) in present)
    oe_total = sum(1 for b in BENCHMARKS for a in AGENTS if oe.get((b, a), 0) > 0)

    md = []
    md.append("# Official Matrix 88 (11 agents x 8 official benchmarks)\n")
    md.append(f"- cells executed: **{total_cells}/88**")
    md.append(f"- cells with official-oracle verdict: **{oe_total}/88**\n")
    md.append("## Per-benchmark\n")
    md.append("| benchmark | official oracle | agents ran | official_eval | correct |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['benchmark']} | {r['official_oracle']} | {r['agents_ran']} "
                  f"| {r['official_eval_cells']} | {r['correct_cells']} |")
    md.append("\n## official_eval task-count matrix (rows=benchmark, cols=agent)\n")
    md.append("| benchmark | " + " | ".join(a[:8] for a in AGENTS) + " |")
    md.append("|" + "---|" * (len(AGENTS) + 1))
    for b in BENCHMARKS:
        md.append("| " + b + " | " + " | ".join(str(oe.get((b, a), "-")) for a in AGENTS) + " |")
    md.append("\n_official_eval = the benchmark's official oracle ran and returned a verdict._")
    md.append("_correct = that official verdict was a pass. correct<official is a normal benchmark outcome (the agent's kernel failed the real check), not a pipeline error._\n")
    (out / "SUMMARY.md").write_text("\n".join(md))

    print("\n".join(md))
    print(f"\nWrote: {out}/matrix_88_official_eval.csv, matrix_88_correct.csv, "
          f"matrix_88_summary.csv, SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
