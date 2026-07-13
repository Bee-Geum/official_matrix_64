#!/usr/bin/env python3
"""
compat.py -- compute the agent x benchmark compatibility matrix.

"Every agent on every benchmark" is a *matrix*, not a full cross-product: a
CUDA-only agent cannot satisfy a Triton-only benchmark, and an A100 cannot run
an AMD / NPU / TPU benchmark. This module reads registry/agents.csv and
registry/benchmarks.csv and decides, for each cell, whether it RUNs (and how)
or is SKIPped (and why).

Usage:
    python3 registry/compat.py                 # print markdown grid + write compat_matrix.csv
    python3 registry/compat.py --runnable       # print only RUN cells, one "agent benchmark" per line
    python3 registry/compat.py --csv out.csv    # choose output path
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT_ROOT = HERE.parent

# Concrete languages that 'any' expands to for intersection purposes.
ALL_LANGS = {"cuda", "triton"}
import os
TARGET_GPU_LABEL = os.environ.get("UB_TARGET_GPU_LABEL", "RTX PRO 6000")


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        # strip comment / blank lines, keep the header (first non-comment line)
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items()})
    return rows


def langs_of(field: str) -> set[str]:
    field = (field or "").strip().lower()
    if field in ("any", "", "*"):
        return set(ALL_LANGS)
    return {x.strip() for x in field.split("|") if x.strip()}


def hardware_of(field: str) -> set[str]:
    return {x.strip().lower() for x in (field or "").split("|") if x.strip()}


# format -> driver-side shim needed so an existing ModelNew driver can feed it.
# kb_model needs none (drivers already take --task KB.py and emit ModelNew).
SHIM_FOR_FORMAT = {
    "kb_model": "none",
    "op_fill": "op",            # wrap one ATen op as a ModelNew reference
    "triton_standalone": "triton",  # wrap a bare-Triton task as a ModelNew reference
    "parallel_prompt": "parallel",  # ParEval prompt -> ModelNew-style harness
    "inference_schema": "native",   # FlashInfer-Bench native schema driver
    "sol_subgraph": "native",       # SOL-ExecBench native subgraph driver
}


def decide(agent: dict, bench: dict) -> dict:
    a_langs = langs_of(agent["langs"])
    b_langs = langs_of(bench["langs"])
    b_hw = hardware_of(bench["hardware"])
    a100 = bench.get("a100", "no").strip().lower()
    fmt = bench.get("task_format", "").strip()
    shim = SHIM_FOR_FORMAT.get(fmt, "native")

    reasons = []
    status = "RUN"

    # 1. hardware gate
    if "nvidia" not in b_hw or a100 == "no":
        status = "SKIP"
        reasons.append(f"hardware: needs {'|'.join(sorted(b_hw))} (not runnable on {TARGET_GPU_LABEL})")

    # 2. language gate
    lang_used = sorted(a_langs & b_langs)
    if not lang_used:
        status = "SKIP"
        reasons.append(
            f"lang mismatch: agent={'|'.join(sorted(a_langs))} bench={'|'.join(sorted(b_langs))}"
        )

    # 3. notes for cells that RUN but with caveats
    if status == "RUN":
        if a100 == "partial":
            reasons.append("partial: RTX6000-runnable subset only (see prepare.sh)")
        if shim == "op":
            reasons.append("needs op-shim driver (operator-level, not ModelNew)")
        elif shim == "triton":
            reasons.append("needs triton format-shim (bare kernel, not ModelNew)")
        elif shim == "parallel":
            reasons.append("needs parallel-prompt shim")
        elif shim == "native":
            reasons.append("uses benchmark-native evaluator")

    return {
        "agent": agent["agent"],
        "benchmark": bench["benchmark"],
        "status": status,
        "lang_used": "|".join(lang_used) if lang_used else "-",
        "eval_mode": bench.get("eval_mode", ""),
        "shim": shim,
        "a100": a100,
        "agent_confidence": agent.get("confidence", ""),
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def compute_matrix(agents=None, benchmarks=None) -> list[dict]:
    agents = agents or load_csv(HERE / "agents.csv")
    benchmarks = benchmarks or load_csv(HERE / "benchmarks.csv")
    cells = []
    for agent in agents:
        for bench in benchmarks:
            cells.append(decide(agent, bench))
    return cells


def cell_symbol(cell: dict) -> str:
    if cell["status"] == "SKIP":
        return "."
    if cell["shim"] == "none" and cell["a100"] == "yes":
        return "Y"   # clean run
    return "y"       # runs but with a shim and/or only a subset (partial)


def print_grid(cells: list[dict], agents: list[dict], benchmarks: list[dict]) -> None:
    a_names = [a["agent"] for a in agents]
    b_names = [b["benchmark"] for b in benchmarks]
    index = {(c["agent"], c["benchmark"]): c for c in cells}

    width = max(len(n) for n in a_names) + 1
    header = " " * width + "  ".join(b[:5].ljust(5) for b in b_names)
    print(header)
    for a in a_names:
        cells_str = "  ".join(cell_symbol(index[(a, b)]).center(5) for b in b_names)
        print(a.ljust(width) + cells_str)

    print()
    print(f"legend:  Y = clean run (ModelNew, {TARGET_GPU_LABEL})   "
          "y = runs via shim and/or partial subset   . = skip")
    print("columns:", ", ".join(f"{b[:5]}={b}" for b in b_names))

    n_run = sum(c["status"] == "RUN" for c in cells)
    n_clean = sum(cell_symbol(c) == "Y" for c in cells)
    print(f"\ncells: {len(cells)} total   RUN: {n_run} "
          f"(clean: {n_clean}, via-shim/partial: {n_run - n_clean})   "
          f"SKIP: {len(cells) - n_run}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runnable", action="store_true",
                        help="print only RUN cells as 'agent benchmark' lines")
    parser.add_argument("--clean-only", action="store_true",
                        help="with --runnable, restrict to clean (no-shim, target GPU) cells")
    parser.add_argument("--csv", default=str(EXT_ROOT / "compat_matrix.csv"))
    args = parser.parse_args()

    agents = load_csv(HERE / "agents.csv")
    benchmarks = load_csv(HERE / "benchmarks.csv")
    cells = compute_matrix(agents, benchmarks)

    if args.runnable:
        for c in cells:
            if c["status"] != "RUN":
                continue
            if args.clean_only and cell_symbol(c) != "Y":
                continue
            print(f"{c['agent']} {c['benchmark']}")
        return 0

    out = Path(args.csv)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)

    print_grid(cells, agents, benchmarks)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
