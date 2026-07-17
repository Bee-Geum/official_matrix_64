#!/usr/bin/env python3
"""Classify each (benchmark, agent) cell by WHY it landed where it did.

summarize_matrix_64.py reports official_eval / correct, which cannot distinguish
"the agent's kernel was wrong" from "the agent never ran". Both show up as
correct=0, and the second one is not a benchmark result at all.

Categories:
  PASS            the official oracle ran and passed
  AGENT_FAIL      the agent produced a real candidate; the oracle rejected it
  FORMAT_MISMATCH the benchmark's task format is not the agent's input contract
                  (the runner hands non-KernelBench benchmarks a prompt .txt;
                  CudaForge/KernelMem want a KernelBench model .py). Not a score.
  INFRA           harness/dependency failure -- nothing to do with the agent
  NO_VERDICT      the oracle returned nothing usable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARKS = ["kernelbench", "robust_kbench", "tritonbench_t", "tritonbench_g",
              "multikernelbench", "backendbench", "pareval", "sol_execbench"]
AGENTS = ["cudaforge", "autokernel", "kernelskill", "drkernel",
          "autotriton", "kernelllm", "incoder32b"]

# Only these hand the driver an actual task .py (official_all_matrix_v1.py:
# KB_BENCHMARKS). Everything else gets official_prompt.txt.
KB_BENCHMARKS = {"kernelbench", "robust_kbench"}

# Agents whose upstream entry point takes a KernelBench model .py positionally
# and cannot consume a prompt .txt at all.
KB_ONLY_AGENTS = {"cudaforge", "kernelskill", "autokernel"}

INFRA_MARKERS = (
    "ModuleNotFoundError", "No module named", "not covered by locally-available",
    "repo missing", "run directory not found", "summary.json not found",
    "refusing to harvest", "produced no runnable kernel",
)


def classify(bench: str, agent: str, cell: dict | None) -> tuple[str, str]:
    if cell is None:
        return "NO_VERDICT", "cell_result.json missing"

    tasks = cell.get("task_results") or []
    task = tasks[0] if tasks else {}
    correct = int(cell.get("correct_task_count", 0) or 0)
    official = int(cell.get("official_eval_task_count", 0) or 0)
    infra = str(task.get("infrastructure_error") or cell.get("top_infrastructure_error") or "")
    cand_err = str(task.get("candidate_error") or "")

    if correct > 0:
        return "PASS", f"best_score={task.get('best_score')}"

    if agent in KB_ONLY_AGENTS and bench not in KB_BENCHMARKS:
        return "FORMAT_MISMATCH", (f"{agent} takes a KernelBench model .py; "
                                   f"{bench} hands it official_prompt.txt")

    blob = infra + " " + cand_err
    for marker in INFRA_MARKERS:
        if marker in blob:
            return "INFRA", marker

    if official > 0:
        return "AGENT_FAIL", (cand_err or infra or "oracle rejected the candidate")[:90]
    return "NO_VERDICT", (infra or cand_err or "no official verdict")[:90]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    root = Path(args.run_root)

    grid: dict[tuple[str, str], tuple[str, str]] = {}
    for bench in BENCHMARKS:
        for agent in AGENTS:
            path = root / "cells" / bench / agent / "cell_result.json"
            cell = None
            if path.exists():
                try:
                    cell = json.loads(path.read_text(errors="ignore"))
                except Exception:
                    cell = None
            grid[(bench, agent)] = classify(bench, agent, cell)

    short = {"PASS": "PASS", "AGENT_FAIL": "fail", "FORMAT_MISMATCH": "n/a",
             "INFRA": "INFRA", "NO_VERDICT": "-"}
    width = max(len(a) for a in AGENTS) + 1

    print("\n= 7 agents x 8 official benchmarks =\n")
    print(f"{'benchmark':17}" + "".join(f"{a[:9]:>11}" for a in AGENTS))
    for bench in BENCHMARKS:
        row = f"{bench:17}"
        for agent in AGENTS:
            row += f"{short[grid[(bench, agent)][0]]:>11}"
        print(row)

    print("\n  PASS  = official oracle ran and passed")
    print("  fail  = agent produced a real candidate, oracle rejected it (a real result)")
    print("  n/a   = benchmark's task format is not this agent's input contract (NOT a score)")
    print("  INFRA = harness/dependency failure, unrelated to the agent")
    print("  -     = no official verdict")

    counts: dict[str, int] = {}
    for verdict, _ in grid.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\n= totals =")
    for key in ("PASS", "AGENT_FAIL", "FORMAT_MISMATCH", "INFRA", "NO_VERDICT"):
        print(f"  {key:16} {counts.get(key, 0):2d}/{len(grid)}")

    scored = counts.get("PASS", 0) + counts.get("AGENT_FAIL", 0)
    print(f"\n  cells that are actually a benchmark result: {scored}/{len(grid)}")

    print("\n= detail =")
    for (bench, agent), (verdict, why) in sorted(grid.items()):
        if verdict != "PASS":
            print(f"  {bench:17} {agent:12} {verdict:16} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
