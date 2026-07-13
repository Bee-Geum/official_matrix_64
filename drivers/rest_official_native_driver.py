#!/usr/bin/env python3

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ROOT", "/home/bi_geum/unified_bench")).resolve()
GPU = os.environ.get("GPU", "0")
TIMEOUT = int(os.environ.get("REST_OFFICIAL_TIMEOUT", "900"))


def run_cmd(cmd: list[str], cwd: Path, out_file: Path, timeout: int = TIMEOUT) -> dict[str, Any]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH','')}"},
        )
        txt = "$ " + " ".join(map(str, cmd)) + "\n"
        txt += f"cwd={cwd}\nreturncode={proc.returncode}\n"
        txt += "----- STDOUT -----\n" + proc.stdout + "\n"
        txt += "----- STDERR -----\n" + proc.stderr + "\n"
        out_file.write_text(txt, errors="ignore")
        return {"cmd": [str(x) for x in cmd], "cwd": str(cwd), "rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "wall_s": time.time()-start, "log": str(out_file)}
    except subprocess.TimeoutExpired as e:
        out_file.write_text("$ " + " ".join(map(str, cmd)) + f"\ncwd={cwd}\nreturncode=124\nTIMEOUT\n", errors="ignore")
        return {"cmd": [str(x) for x in cmd], "cwd": str(cwd), "rc": 124, "stdout": "", "stderr": "TIMEOUT", "wall_s": time.time()-start, "log": str(out_file)}
    except Exception as e:
        out_file.write_text("$ " + " ".join(map(str, cmd)) + "\nERROR\n" + repr(e), errors="ignore")
        return {"cmd": [str(x) for x in cmd], "cwd": str(cwd), "rc": 125, "stdout": "", "stderr": repr(e), "wall_s": time.time()-start, "log": str(out_file)}


def syntax_check(path: Path) -> tuple[bool, str]:
    try:
        ast.parse(path.read_text(errors="ignore"))
        return True, ""
    except Exception as e:
        return False, repr(e)


def find_candidates(cand_dir: Path, glob_pat: str | None) -> list[Path]:
    pats = []
    if glob_pat:
        pats.append(glob_pat)
    pats += ["candidate_*.py", "round*_kernel.py", "*.py"]
    out, seen = [], set()
    for pat in pats:
        for p in cand_dir.rglob(pat):
            if not p.is_file() or p in seen:
                continue
            if "raw_reply" in p.name or "reply" in p.name:
                continue
            if p.stat().st_size < 10_000_000:
                seen.add(p)
                out.append(p)
    return out


def detect_bench(task: Path, bench_root: str | None, cand_dir: Path) -> str:
    low = " ".join([str(task), str(bench_root or ""), str(cand_dir)]).lower()
    if "tritonbench_t" in low or "tritonbench_t_v1" in low or "perf_t" in low:
        return "tritonbench_t"
    if "tritonbench_g" in low or "tritonbench_g_v1" in low or "perf_g" in low:
        return "tritonbench_g"
    return "unknown"


def repo() -> Path | None:
    p = ROOT / "third_party" / "TritonBench"
    return p if p.exists() else None


def infer_task_stem(task: Path, cand_dir: Path) -> str:
    s = str(task)
    if s and s != "unknown_task":
        return Path(s).stem
    for parent in [cand_dir] + list(cand_dir.parents):
        m = re.match(r"(.+)__rep\d+$", parent.name)
        if m:
            return m.group(1)
    return cand_dir.name


def find_official_task_file(tb: Path, bench: str, stem: str) -> Path | None:
    suffix = "T" if bench == "tritonbench_t" else "G"
    for root in [
        tb / "data" / f"TritonBench_{suffix}_v1",
        tb / "performance_metrics" / f"perf_{suffix}",
        tb,
    ]:
        if not root.exists():
            continue
        for name in [f"{stem}.py", f"{stem}_perf.py"]:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    return None


def clear_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def stage_triton(task: Path, cand: Path, out_dir: Path, bench: str) -> tuple[Path, Path, str, str]:
    tb = repo()
    stem = infer_task_stem(task, cand.parent)
    official = find_official_task_file(tb, bench, stem) if tb else None

    stage = out_dir / "tritonbench_official_stage" / stem / cand.stem
    src_dir = stage / "source"
    tgt_dir = stage / "target"

    # Critical v4 fix: remove stale candidate_0000.py / round000_kernel.py.
    clear_dir(src_dir)
    clear_dir(tgt_dir)

    src = src_dir / f"{stem}.py"
    tgt = tgt_dir / f"{stem}.py"

    warning = ""
    if official and official.exists() and official.name.endswith(".py") and not official.name.endswith("_perf.py"):
        shutil.copy2(official, src)
    elif task.exists() and task.is_file():
        shutil.copy2(task, src)
    else:
        src.write_text("")
        warning = f"source file for task {stem} not found; staged empty source"

    shutil.copy2(cand, tgt)

    # Sanity: target dir must contain only original task filename.
    for p in tgt_dir.iterdir():
        if p.name != f"{stem}.py":
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

    return src_dir, tgt_dir, stem, warning


def wipe_perf_tmp(perf_dir: Path, perf_results: Path) -> None:
    # Critical v4 fix: delete all stale tmp files, not just current stem.
    for d in [perf_dir / "tmp", perf_dir / "run_bench" / "tmp"]:
        if d.exists():
            for p in d.glob("*"):
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
        d.mkdir(parents=True, exist_ok=True)
    if perf_results.exists():
        shutil.rmtree(perf_results, ignore_errors=True)
    perf_results.mkdir(parents=True, exist_ok=True)


def write_file_flag(write_file: Path) -> str:
    txt = write_file.read_text(errors="ignore")
    if "--result_folder_path" in txt:
        return "--result_folder_path"
    return "--results_path"


def parse_speed(text: str) -> float | None:
    patterns = [
        r"speed\s*up\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"speedup\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"efficiency\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def eval_tritonbench(bench: str, task: Path, cand: Path, cand_eval_dir: Path) -> dict[str, Any]:
    tb = repo()
    report: dict[str, Any] = {
        "official_attempted": False,
        "official_eval": 0,
        "official_correctness_eval": 0,
        "official_performance_eval": 0,
        "blocked_reason": "",
        "commands": [],
    }
    if tb is None:
        report["blocked_reason"] = "TritonBench repo not found"
        return report

    suffix = "T" if bench == "tritonbench_t" else "G"
    eval_dir = tb / "EVAL" / f"eval_{suffix}"
    if not eval_dir.exists():
        report["blocked_reason"] = f"{eval_dir} not found"
        return report

    call_script = eval_dir / "0_call_acc.py"
    exe_script = eval_dir / "1_exe_acc.py"
    eff_script = eval_dir / "2_efficiency.py"
    if not call_script.exists() or not exe_script.exists():
        report["blocked_reason"] = f"TritonBench official scripts missing in {eval_dir}"
        return report

    src_dir, tgt_dir, stem, stage_warn = stage_triton(task, cand, cand_eval_dir, bench)

    report["official_attempted"] = True
    report["official_eval"] = 1

    c0 = run_cmd([sys.executable, str(call_script), "--source", str(src_dir), "--target", str(tgt_dir), "--GPUs", str(GPU)], eval_dir, cand_eval_dir / f"{cand.stem}.0_call_acc.log.txt")
    c1 = run_cmd([sys.executable, str(exe_script), "--folder", str(tgt_dir), "--GPUs", str(GPU)], eval_dir, cand_eval_dir / f"{cand.stem}.1_exe_acc.log.txt")
    report["commands"] += [c0, c1]

    out1 = c1.get("stdout", "") + c1.get("stderr", "")
    correctness_ok = c0["rc"] == 0 and c1["rc"] == 0 and ("Correct execution rate: 100.00%" in out1 or "Correct execution rate: 100%" in out1)
    report["official_correctness_eval"] = 1
    report["correct"] = int(correctness_ok)

    perf_dir = tb / "performance_metrics" / f"perf_{suffix}"
    speed = None
    performance_ok = False
    if perf_dir.exists() and eff_script.exists():
        write_file = perf_dir / "run_bench" / "write_file.py"
        multi = perf_dir / "run_bench" / "multiprocess_gpu_run.py"
        perf_results = cand_eval_dir / f"{stem}.perf_results"
        wipe_perf_tmp(perf_dir, perf_results)

        if write_file.exists():
            flag = write_file_flag(write_file)
            c2 = run_cmd([sys.executable, str(write_file), "--input_folder_path", str(tgt_dir), flag, str(perf_results)], perf_dir, cand_eval_dir / f"{cand.stem}.perf_write_file.log.txt")
            report["commands"].append(c2)

            if multi.exists():
                c3 = run_cmd([sys.executable, str(multi)], perf_dir, cand_eval_dir / f"{cand.stem}.perf_multiprocess.log.txt")
                report["commands"].append(c3)

            c4 = run_cmd([sys.executable, str(eff_script), "--gen_folder", str(perf_results)], eval_dir, cand_eval_dir / f"{cand.stem}.2_efficiency.log.txt")
            report["commands"].append(c4)

            speed = parse_speed(c4.get("stdout", "") + "\n" + c4.get("stderr", ""))
            performance_ok = c4["rc"] == 0 and speed is not None

    report["official_performance_eval"] = int(performance_ok)
    report["best_score"] = float(speed or 0.0)
    report["blocked_reason"] = stage_warn

    if not correctness_ok:
        report["error"] = "\n".join((c.get("stdout","") + c.get("stderr","")) for c in report["commands"])[-3000:]
    elif not performance_ok:
        report["error"] = "correctness passed, but performance/2_efficiency did not produce a valid speedup"
    else:
        report["error"] = ""
    return report


def eval_one(bench: str, task: Path, cand: Path, out_dir: Path) -> dict[str, Any]:
    ok, syn_err = syntax_check(cand)
    base = {
        "candidate": str(cand),
        "syntax_ok": ok,
        "syntax_error": syn_err,
        "official_attempted": False,
        "official_eval": 0,
        "official_correctness_eval": 0,
        "official_performance_eval": 0,
        "forced_smoke_eval": 0,
        "correct": 0,
        "best_score": 0.0,
        "blocked_reason": "",
        "error": "",
    }
    try:
        if bench in {"tritonbench_t", "tritonbench_g"}:
            base.update(eval_tritonbench(bench, task, cand, out_dir))
        else:
            base["blocked_reason"] = f"{bench} not handled in v4 triton-only driver"
    except Exception as e:
        base["error"] = repr(e)
        base["blocked_reason"] = "triton official adapter exception"
    return base


def write_final_report(bench: str, task: Path, cand_dir: Path, out_dir: Path, candidates: list[Path], verdicts: list[dict[str, Any]]) -> int:
    n_candidates = len(candidates)
    n_official = sum(1 for v in verdicts if int(v.get("official_eval", 0)) == 1)
    n_correct = sum(1 for v in verdicts if int(v.get("correct", 0)) == 1)
    n_compiled = sum(1 for v in verdicts if v.get("syntax_ok") or int(v.get("official_eval", 0)) == 1)
    n_perf = sum(1 for v in verdicts if int(v.get("official_performance_eval", 0)) == 1)
    best_score = max([float(v.get("best_score", 0) or 0) for v in verdicts] + [0.0])
    blocked = [v.get("blocked_reason", "") for v in verdicts if v.get("blocked_reason")]
    report = {
        "adapter": "rest_official_native_driver_v4",
        "benchmark": bench,
        "task": str(task),
        "cand_dir": str(cand_dir),
        "out_dir": str(out_dir),
        "n_candidates": n_candidates,
        "n_compiled": n_compiled,
        "n_correct": n_correct,
        "n_official_attempted": n_official,
        "n_official_performance": n_perf,
        "official_eval": int(n_official > 0),
        "official_performance_eval": int(n_perf > 0),
        "forced_smoke_eval": 0,
        "strict_official": 1,
        "best_score": best_score,
        "blocked_reason": "; ".join(sorted(set(blocked)))[:4000],
        "error": "" if n_perf > 0 else ("official correctness attempted, but official performance score not available" if n_official > 0 else "no official evaluator executed"),
        "verdicts": verdicts,
    }
    summary = [{
        "task": str(task),
        "task_dir": str(out_dir),
        "benchmark": bench,
        "n_candidates": n_candidates,
        "n_compiled": n_compiled,
        "n_correct": n_correct,
        "n_official_attempted": n_official,
        "n_official_performance": n_perf,
        "runnable_rate": n_compiled/n_candidates if n_candidates else 0.0,
        "correct_rate": n_correct/n_candidates if n_candidates else 0.0,
        "pass@1": 1.0 if n_correct > 0 else 0.0,
        "fast_1": 1.0 if best_score > 1.0 else 0.0,
        "best_score": best_score,
        "geomean_speedup": best_score,
        "median_speedup": best_score,
        "official_eval": int(n_official > 0),
        "official_performance_eval": int(n_perf > 0),
        "forced_smoke_eval": 0,
        "native_adapter": "rest_official_native_driver_v4",
        "blocked_reason": report["blocked_reason"],
        "error": report["error"],
    }]
    (out_dir/"native_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir/"verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary[0], indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", default="")
    parser.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default="")
    parser.add_argument("--out", "--out_dir", "--eval_dir", "--task_work_dir", "--output_dir", dest="out_dir", default="")
    parser.add_argument("--bench-root", dest="bench_root", default="")
    parser.add_argument("--glob", default="")
    args, _ = parser.parse_known_args()

    task = Path(args.task) if args.task else Path("unknown_task")
    cand_dir = Path(args.cand_dir or ".")
    out_dir = Path(args.out_dir or cand_dir/"official_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = detect_bench(task, args.bench_root, cand_dir)
    candidates = find_candidates(cand_dir, args.glob)
    verdicts = [eval_one(bench, task, c, out_dir/c.stem) for c in candidates]
    return write_final_report(bench, task, cand_dir, out_dir, candidates, verdicts)

if __name__ == "__main__":
    raise SystemExit(main())
