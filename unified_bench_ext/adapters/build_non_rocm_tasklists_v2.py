#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path.cwd()
TASK_DIR = ROOT / "unified_bench_ext" / "task_lists"
TASK_DIR.mkdir(parents=True, exist_ok=True)
GEN = ROOT / "generated_non_rocm_tasks"
GEN.mkdir(exist_ok=True)


def safe_rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def write_list(name: str, paths: list[Path], kind: str, min_synthetic: int = 2):
    seen = []
    used = set()
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        if p in used:
            continue
        if p.stat().st_size > 4_000_000:
            continue
        used.add(p)
        seen.append(p)

    # 무조건 실행을 위해, 파일이 없으면 synthetic prompt를 만든다.
    while len(seen) < min_synthetic:
        idx = len(seen)
        sp = GEN / f"{name}_synthetic_{idx}.txt"
        sp.write_text(
            f"Benchmark: {name}\n"
            f"Kind: {kind}\n"
            "No concrete task file was auto-located. "
            "This synthetic prompt is used so every non-ROCm benchmark cell executes in smoke mode.\n"
        )
        seen.append(sp)

    txt = TASK_DIR / f"{name}.txt"
    csv_path = TASK_DIR / f"{name}.csv"
    txt.write_text("\n".join(safe_rel(p) for p in seen) + "\n")

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "task_id", "task_path", "task_name", "kind"])
        w.writeheader()
        for i, p in enumerate(seen):
            w.writerow({
                "benchmark": name,
                "task_id": i,
                "task_path": safe_rel(p),
                "task_name": p.name,
                "kind": kind,
            })
    return len(seen)


def glob_many(patterns):
    out = []
    for pat in patterns:
        out.extend(ROOT.glob(pat))
    return sorted(out)


counts = {}

# TritonBench paths identified by repo probes.
counts["tritonbench_t"] = write_list(
    "tritonbench_t",
    glob_many([
        "third_party/TritonBench/data/TritonBench_T_v1/*.py",
        "third_party/TritonBench/**/TritonBench_T_v1/*.py",
        "third_party/TritonBench/performance_metrics/perf_T/*.py",
    ]),
    "triton_python",
)

counts["tritonbench_g"] = write_list(
    "tritonbench_g",
    glob_many([
        "third_party/TritonBench/data/TritonBench_G_v1/*.py",
        "third_party/TritonBench/**/TritonBench_G_v1/*.py",
        "third_party/TritonBench/performance_metrics/perf_G/*.py",
        "third_party/TritonBench/data/**/*.py",
    ]),
    "triton_python",
)

counts["multikernelbench"] = write_list(
    "multikernelbench",
    glob_many([
        "third_party/MultiKernelBench/prompts/*.py",
        "third_party/MultiKernelBench/prompts/*.txt",
        "third_party/MultiKernelBench/**/*.py",
        "third_party/MultiKernelBench/**/*.md",
    ]),
    "prompt_or_python",
)

counts["backendbench_smoke"] = write_list(
    "backendbench_smoke",
    glob_many([
        "third_party/BackendBench/test/fixtures/llm_response/*.txt",
        "third_party/BackendBench/test/test_smoke.py",
        "third_party/BackendBench/test/test_eval.py",
        "third_party/BackendBench/BackendBench/suite/smoke.py",
        "third_party/BackendBench/**/*.py",
    ]),
    "backendbench_probe",
)

counts["flashinfer_bench"] = write_list(
    "flashinfer_bench",
    glob_many([
        "third_party/flashinfer-bench/**/*.py",
        "third_party/flashinfer-bench/**/*.json",
        "third_party/flashinfer-bench/**/*.yaml",
        "third_party/flashinfer-bench/**/*.md",
        "third_party/FlashInfer-Bench/**/*.py",
        "third_party/FlashInfer-Bench/**/*.json",
        "third_party/FlashInfer-Bench/**/*.yaml",
        "third_party/FlashInfer-Bench/**/*.md",
    ]),
    "flashinfer_probe",
)

counts["sol_execbench"] = write_list(
    "sol_execbench",
    glob_many([
        "third_party/SOL-ExecBench/**/*.py",
        "third_party/SOL-ExecBench/**/*.json",
        "third_party/SOL-ExecBench/**/*.yaml",
        "third_party/SOL-ExecBench/**/*.md",
        "third_party/sol-execbench/**/*.py",
        "third_party/sol-execbench/**/*.json",
        "third_party/sol-execbench/**/*.yaml",
        "third_party/sol-execbench/**/*.md",
    ]),
    "sol_probe",
)

# ParEval list already exists in your environment. If missing, rebuild from raw prompts.
pareval_txt = TASK_DIR / "pareval_cuda.txt"
if not pareval_txt.exists() or pareval_txt.stat().st_size == 0:
    counts["pareval_cuda"] = write_list(
        "pareval_cuda",
        glob_many([
            "third_party/ParEval/prompts/raw/*/*/cuda",
            "third_party/ParEval/**/*.json",
            "third_party/ParEval/**/*.txt",
        ]),
        "pareval_cuda_prompt",
    )
else:
    counts["pareval_cuda"] = len([x for x in pareval_txt.read_text(errors="ignore").splitlines() if x.strip()])

# Ensure kernelbench_all250 exists if KernelBench is available.
if not (ROOT / "kernelbench_all250.txt").exists():
    kb = None
    for c in [
        ROOT / "third_party" / "KernelBench" / "KernelBench",
        ROOT / "third_party" / "KernelBench",
        ROOT / "KernelBench",
    ]:
        if (c / "level1").exists():
            kb = c
            break
    if kb is not None:
        rows = []
        def tid(p):
            m = re.match(r"(\d+)_", p.name)
            return int(m.group(1)) if m else 10**9
        for level in ["level1", "level2", "level3"]:
            for p in sorted((kb / level).glob("*.py"), key=tid):
                rows.append({"level": level, "task_id": tid(p), "task_path": safe_rel(p), "task_name": p.name})
        (ROOT / "kernelbench_all250.txt").write_text("\n".join(r["task_path"] for r in rows) + "\n")
        with (ROOT / "kernelbench_all250.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["level", "task_id", "task_path", "task_name"])
            w.writeheader()
            w.writerows(rows)
        counts["kernelbench_all250"] = len(rows)

out = ROOT / "results" / "non_rocm_adapters_v2" / "tasklist_counts.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(counts, indent=2, ensure_ascii=False))
print(json.dumps(counts, indent=2, ensure_ascii=False))
