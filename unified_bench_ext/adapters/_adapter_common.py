from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def normalize_call_args(*args, **kwargs):
    task = (
        kwargs.get("task")
        or kwargs.get("task_path")
        or kwargs.get("task_file")
        or kwargs.get("prompt_path")
        or kwargs.get("problem_path")
        or (args[0] if len(args) >= 1 else None)
    )
    cand_dir = (
        kwargs.get("cand_dir")
        or kwargs.get("candidate_dir")
        or kwargs.get("candidates_dir")
        or kwargs.get("solution_dir")
        or kwargs.get("run_dir")
        or (args[1] if len(args) >= 2 else None)
    )
    out_dir = (
        kwargs.get("out_dir")
        or kwargs.get("eval_dir")
        or kwargs.get("task_work_dir")
        or kwargs.get("output_dir")
        or (args[2] if len(args) >= 3 else None)
    )

    task = Path(str(task or "unknown_task"))
    cand_dir = Path(str(cand_dir or "."))
    out_dir = Path(str(out_dir or cand_dir / "native_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return task, cand_dir, out_dir


def find_candidates(cand_dir: Path):
    patterns = [
        "candidate_*.py",
        "round*_kernel.py",
        "*.py",
        "*.cu",
        "*.cpp",
        "*.cc",
        "*.cuh",
        "*.txt",
    ]
    seen = set()
    out = []
    for pat in patterns:
        for p in cand_dir.rglob(pat):
            if p.is_file() and p not in seen and p.stat().st_size < 8_000_000:
                seen.add(p)
                out.append(p)
    return out


def syntax_check(path: Path):
    try:
        ast.parse(path.read_text(errors="ignore"))
        return True, ""
    except Exception as e:
        return False, repr(e)


def import_check(path: Path):
    try:
        spec = importlib.util.spec_from_file_location("candidate_mod", str(path))
        if spec is None or spec.loader is None:
            return False, "cannot create import spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, ""
    except Exception as e:
        return False, repr(e)


def nvcc_compile_check(path: Path, out_dir: Path):
    if shutil.which("nvcc") is None:
        return False, "nvcc not found"
    exe = out_dir / (path.stem + ".out")
    cmd = ["nvcc", "-O2", str(path), "-o", str(exe)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        (out_dir / (path.stem + ".nvcc.stdout.txt")).write_text(proc.stdout)
        (out_dir / (path.stem + ".nvcc.stderr.txt")).write_text(proc.stderr)
        return proc.returncode == 0, proc.stderr[-4000:]
    except Exception as e:
        return False, repr(e)


def text_candidate_check(path: Path):
    try:
        text = path.read_text(errors="ignore")
    except Exception as e:
        return False, repr(e)
    return bool(text.strip()), "" if text.strip() else "empty text candidate"


def smoke_eval(task: Path, cand_dir: Path, out_dir: Path, adapter_name: str, official_eval: int = 0, extra: dict[str, Any] | None = None):
    candidates = find_candidates(cand_dir)
    verdicts = []
    n_compiled = 0

    for i, c in enumerate(candidates):
        verdict = {
            "candidate": str(c),
            "candidate_index": i,
            "compiled": False,
            "correct": False,
            "syntax_ok": False,
            "import_ok": False,
            "compile_ok": False,
            "error": "",
        }

        if c.suffix == ".py":
            ok, err = syntax_check(c)
            verdict["syntax_ok"] = ok
            if ok:
                imp_ok, imp_err = import_check(c)
                verdict["import_ok"] = imp_ok
                verdict["compile_ok"] = imp_ok
                verdict["compiled"] = imp_ok
                verdict["error"] = imp_err
            else:
                verdict["error"] = err
        elif c.suffix in {".cu", ".cpp", ".cc", ".cuh"}:
            ok, err = nvcc_compile_check(c, out_dir)
            verdict["compile_ok"] = ok
            verdict["compiled"] = ok
            verdict["error"] = err
        else:
            ok, err = text_candidate_check(c)
            verdict["syntax_ok"] = ok
            verdict["compiled"] = ok
            verdict["compile_ok"] = ok
            verdict["error"] = err

        if verdict["compiled"]:
            n_compiled += 1
        verdicts.append(verdict)

    report = {
        "adapter": adapter_name,
        "task": str(task),
        "cand_dir": str(cand_dir),
        "out_dir": str(out_dir),
        "n_candidates": len(candidates),
        "n_compiled": n_compiled,
        "n_correct": 0,
        "official_eval": int(official_eval),
        "forced_smoke_eval": int(not official_eval),
        "first_correct": False,
        "best_score": 0.0,
        "error": "" if candidates else "no candidates found",
        "verdicts": verdicts,
    }
    if extra:
        report.update(extra)

    summary = [{
        "task": str(task),
        "task_dir": str(out_dir),
        "n_candidates": report["n_candidates"],
        "n_compiled": report["n_compiled"],
        "n_correct": report["n_correct"],
        "runnable_rate": report["n_compiled"] / report["n_candidates"] if report["n_candidates"] else 0.0,
        "correct_rate": 0.0,
        "pass@1": 0.0,
        "fast_1": 0.0,
        "best_score": 0.0,
        "geomean_speedup": 0.0,
        "median_speedup": 0.0,
        "native_adapter": adapter_name,
        "official_eval": int(official_eval),
        "forced_smoke_eval": int(not official_eval),
        "error": report["error"],
    }]

    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    (out_dir / "native_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary[0]


def cli_main(adapter_name: str, official_eval: int = 0, extra: dict[str, Any] | None = None):
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", default="")
    parser.add_argument("--cand", "--cand_dir", "--candidate_dir", "--candidates_dir", dest="cand_dir", default="")
    parser.add_argument("--out", "--out_dir", "--eval_dir", "--task_work_dir", "--output_dir", dest="out_dir", default="")
    args, _ = parser.parse_known_args()
    task = Path(args.task or "unknown_task")
    cand_dir = Path(args.cand_dir or ".")
    out_dir = Path(args.out_dir or cand_dir / "native_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = smoke_eval(task, cand_dir, out_dir, adapter_name, official_eval, extra)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
