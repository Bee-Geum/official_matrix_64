#!/usr/bin/env python3
"""
collect_matrix.py -- benchmark-aware results collector for the agent x benchmark matrix.

This is the matrix generalization of the harness's a100_campaign_tool.py. The original
collector aggregated KernelBench runs only; the A100 archive diagnostic flagged that
runs/ now MIXES campaigns from several benchmarks, so a benchmark-blind walk double-counts
tasks across benchmarks. This collector parses the benchmark out of every run-dir name and
keys every aggregate on it, so KernelBench / BackendBench / TritonBench / ParEval results
never bleed into each other.

Run-dir naming (produced by run_cell.sh, same scheme the harness already uses):

    runs/{agent}_{model}_{benchmark}_{subset}_round{R}_repeat{P}_temp{T}/
         {task_stem}__rep{k}/final_eval/summary.json

The tricky part is that agent names (cuda_l1, cuda_agent), model names (autotriton8b) and
benchmark names (robust_kbench, tritonbench_t, tritonbench_g, sol_execbench) all contain
underscores, so the name cannot be split on '_'. We instead:

    1. strip the fixed suffix  _round(\\d+)_repeat(\\d+)_temp([0-9.]+)
    2. longest-match the agent against the known agent set (prefix, '_'-delimited)
    3. longest-match the benchmark against the known benchmark set as a '_'-delimited infix
       (this disambiguates cuda / cuda_l1 / cuda_agent and tritonbench_t / tritonbench_g)
    4. whatever sits between the agent and the benchmark is the model; whatever follows the
       benchmark is the subset.

summary.json is read with the canonical schema from lib/canonical_summary.py (a LIST of
per-task dicts). Every benchmark's eval adapter emits that shape, which is what lets a single
sheet hold them all.

Outputs (into results/matrix/ by default):
    matrix_summary.csv    one row per (benchmark, agent, level)  -- the headline grid
    matrix_per_task.csv   one row per (benchmark, agent, task)   -- drill-down, reps averaged
    matrix_runs.csv       one row per run-dir actually collected -- provenance / sanity

Usage:
    python3 unified_bench_ext/collect_matrix.py
    python3 unified_bench_ext/collect_matrix.py --benchmark kernelbench   # one benchmark only
    python3 unified_bench_ext/collect_matrix.py --runs-dir /path/runs --out-dir /path/out
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "registry"))
sys.path.insert(0, str(HERE / "lib"))
import compat  # noqa: E402  (registry/compat.py -- agent/benchmark tables)

SUFFIX_RE = re.compile(r"_round(\d+)_repeat(\d+)_temp([0-9.]+)$")
REP_RE = re.compile(r"^(?P<stem>.+)__rep(?P<rep>\d+)$")
LEVEL_RE = re.compile(r"level[_-]?(\d+)", re.IGNORECASE)


# ----------------------------------------------------------------------------- parsing
def known_names():
    agents = [a["agent"] for a in compat.load_csv(compat.HERE / "agents.csv")]
    benches = [b["benchmark"] for b in compat.load_csv(compat.HERE / "benchmarks.csv")]
    models = []
    for a in compat.load_csv(compat.HERE / "agents.csv"):
        m = a.get("model", "").strip()
        if m and m not in models:
            models.append(m)
    # longest first so the longest-match logic is just "first that fits"
    return (sorted(agents, key=len, reverse=True),
            sorted(benches, key=len, reverse=True),
            sorted(models, key=len, reverse=True))


def parse_run_name(name: str, agents, benches):
    """run-dir name -> dict(agent, model, benchmark, subset, rounds, repeat, temp) or None."""
    m = SUFFIX_RE.search(name)
    if not m:
        return None
    rounds, repeat, temp = int(m.group(1)), int(m.group(2)), m.group(3)
    core = name[: m.start()]  # {agent}_{model}_{benchmark}_{subset}

    # agent: longest known agent that is a '_'-delimited prefix of core
    agent = next((a for a in agents if core == a or core.startswith(a + "_")), None)
    if agent is None:
        return None
    rest = core[len(agent):].lstrip("_")  # {model}_{benchmark}_{subset}

    # benchmark: longest known benchmark appearing as a '_'-delimited infix of rest.
    # require a non-empty model in front of it (rest must contain '_<bench>' not start with it).
    padded = "_" + rest + "_"
    best = None  # (start_index, bench)
    for b in benches:  # already longest-first
        idx = padded.find("_" + b + "_")
        if idx == -1:
            continue
        # model is everything before the match (minus the leading pad underscore)
        model = padded[1:idx]
        if not model:  # benchmark would sit where the model should be -> not this one
            continue
        best = (idx, b, model)
        break
    if best is None:
        return None
    _, benchmark, model = best
    after = padded[best[0] + len("_" + benchmark + "_"):]
    subset = after.strip("_") or "full"

    return {
        "agent": agent, "model": model, "benchmark": benchmark, "subset": subset,
        "rounds": rounds, "repeat": repeat, "temp": temp,
    }


def level_of(task: str, subset: str) -> str:
    """Group key for a task. KernelBench-form tasks live in levelN/ dirs; others fall back."""
    m = LEVEL_RE.search(task or "")
    if m:
        return f"level{m.group(1)}"
    return subset or "all"


# ----------------------------------------------------------------------------- math
def _geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


# ----------------------------------------------------------------------------- collection
def read_summary(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, dict):  # tolerate a single-item dict
        data = [data]
    return data if isinstance(data, list) else []


def read_cell_meta(rep_dir: Path):
    p = rep_dir / "telemetry" / "cell_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def collect(runs_dir: Path, only_benchmark: str | None):
    agents, benches, _models = known_names()
    per_task = {}   # (benchmark, agent, task) -> list of per-rep item dicts
    runs_rows = []  # provenance
    skipped = []    # run-dir names we could not parse

    if not runs_dir.exists():
        return per_task, runs_rows, skipped

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        info = parse_run_name(run_dir.name, agents, benches)
        if info is None:
            skipped.append(run_dir.name)
            continue
        if only_benchmark and info["benchmark"] != only_benchmark:
            continue

        n_items = 0
        gen_seconds_total = 0
        for rep_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            rm = REP_RE.match(rep_dir.name)
            if not rm:
                continue
            rep = int(rm.group("rep"))
            summ = read_summary(rep_dir / "final_eval" / "summary.json")
            meta = read_cell_meta(rep_dir)
            gen_seconds_total += int(meta.get("generation_seconds", 0) or 0)

            for item in summ:
                task = str(item.get("task", rep_dir.name))
                key = (info["benchmark"], info["agent"], task)
                row = dict(item)
                row.update({
                    "benchmark": info["benchmark"], "agent": info["agent"],
                    "model": info["model"], "subset": info["subset"],
                    "rounds": info["rounds"], "repeat_param": info["repeat"],
                    "temp": info["temp"], "rep": rep,
                    "level": level_of(task, info["subset"]),
                    "generation_seconds": meta.get("generation_seconds"),
                })
                per_task.setdefault(key, []).append(row)
                n_items += 1

        runs_rows.append({
            "run": run_dir.name, "benchmark": info["benchmark"], "agent": info["agent"],
            "model": info["model"], "subset": info["subset"], "rounds": info["rounds"],
            "repeat": info["repeat"], "temp": info["temp"],
            "n_task_items": n_items, "gen_seconds_total": gen_seconds_total,
        })

    return per_task, runs_rows, skipped


# ----------------------------------------------------------------------------- aggregation
METRIC_KEYS = ["runnable_rate", "correct_rate", "pass@1", "fast_1",
               "best_score", "geomean_speedup"]


def reduce_per_task(per_task):
    """Average each task's metrics over its repeats -> one row per (benchmark, agent, task)."""
    rows = []
    for (bench, agent, task), reps in sorted(per_task.items()):
        agg = {k: _mean([r.get(k) for r in reps]) for k in METRIC_KEYS}
        first = reps[0]
        rows.append({
            "benchmark": bench, "agent": agent, "task": task,
            "level": first.get("level", "all"),
            "subset": first.get("subset", ""),
            "model": first.get("model", ""),
            "n_reps": len(reps),
            "n_candidates": _mean([r.get("n_candidates") for r in reps]),
            "best_runnable": any(r.get("best_runnable") for r in reps),
            **agg,
            "error": first.get("error", "") or "",
        })
    return rows


def reduce_summary(task_rows):
    """One row per (benchmark, agent, level): the headline grid."""
    groups = {}
    for r in task_rows:
        groups.setdefault((r["benchmark"], r["agent"], r["level"]), []).append(r)

    out = []
    for (bench, agent, level), rs in sorted(groups.items()):
        out.append({
            "benchmark": bench, "agent": agent, "level": level,
            "n_tasks": len(rs),
            "mean_runnable_rate": round(_mean([r["runnable_rate"] for r in rs]), 4),
            "mean_correct_rate": round(_mean([r["correct_rate"] for r in rs]), 4),
            "mean_pass@1": round(_mean([r["pass@1"] for r in rs]), 4),
            "mean_fast_1": round(_mean([r["fast_1"] for r in rs]), 4),
            "mean_best_score": round(_mean([r["best_score"] for r in rs]), 4),
            "geomean_speedup": round(_geomean([r["geomean_speedup"] for r in rs]), 4),
            "n_tasks_any_correct": sum(1 for r in rs if r["correct_rate"] > 0),
        })
    return out


# ----------------------------------------------------------------------------- output
def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")  # leave an explicit empty marker
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_grid(summary_rows):
    """Compact benchmark x agent fast_1 grid to stdout (headline number per cell)."""
    if not summary_rows:
        print("[collect] no results found under runs/ -- nothing to aggregate")
        return
    benches = sorted({r["benchmark"] for r in summary_rows})
    agents = sorted({r["agent"] for r in summary_rows})
    # collapse levels: take task-weighted mean fast_1 per (benchmark, agent)
    cell = {}
    for r in summary_rows:
        k = (r["benchmark"], r["agent"])
        acc = cell.setdefault(k, [0.0, 0])
        acc[0] += r["mean_fast_1"] * r["n_tasks"]
        acc[1] += r["n_tasks"]

    width = max(len(b) for b in benches) + 1
    print("\nfast_1 grid (correct AND speedup>1; task-weighted mean over levels):\n")
    print(" " * width + "  ".join(a[:8].ljust(8) for a in agents))
    for b in benches:
        cells = []
        for a in agents:
            acc = cell.get((b, a))
            cells.append((f"{acc[0]/acc[1]:.3f}" if acc and acc[1] else "  -  ").ljust(8))
        print(b.ljust(width) + "  ".join(cells))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None,
                    help="runs/ directory (default: $UB_ROOT/runs or ../runs)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: $UB_ROOT/results/matrix)")
    ap.add_argument("--benchmark", default=None,
                    help="collect only this benchmark (avoids campaign mixing)")
    args = ap.parse_args()

    ub_root = Path(os.environ.get("UB_ROOT", HERE.parent))
    runs_dir = Path(args.runs_dir) if args.runs_dir else ub_root / "runs"
    out_dir = Path(args.out_dir) if args.out_dir else ub_root / "results" / "matrix"

    per_task, runs_rows, skipped = collect(runs_dir, args.benchmark)
    task_rows = reduce_per_task(per_task)
    summary_rows = reduce_summary(task_rows)

    write_csv(out_dir / "matrix_summary.csv", summary_rows)
    write_csv(out_dir / "matrix_per_task.csv", task_rows)
    write_csv(out_dir / "matrix_runs.csv", runs_rows)

    print_grid(summary_rows)
    print(f"[collect] runs collected : {len(runs_rows)}")
    print(f"[collect] tasks (b,a,t)  : {len(task_rows)}")
    print(f"[collect] summary rows   : {len(summary_rows)}")
    if skipped:
        print(f"[collect] unparsed run-dirs ({len(skipped)}): "
              + ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    print(f"[collect] wrote -> {out_dir}/"
          + "{matrix_summary.csv, matrix_per_task.csv, matrix_runs.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
