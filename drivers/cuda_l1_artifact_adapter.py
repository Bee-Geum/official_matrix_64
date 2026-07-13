#!/usr/bin/env python3
import os as _os, sys as _sys
from pathlib import Path as _Path
if _os.environ.get('REST_REMAINING_OFFICIAL') == '1' and any(x in _sys.argv for x in ['--bench-root', '--glob']):
    _script = _Path(__file__).resolve().parent / 'remaining_official_native_driver.py'
    _os.execv(_sys.executable, [_sys.executable, str(_script)] + _sys.argv[1:])

import os as _os, sys as _sys
from pathlib import Path as _Path
if _os.environ.get("REST_OFFICIAL") == "1" and any(x in _sys.argv for x in ["--bench-root", "--glob"]):
    _script = _Path(__file__).resolve().parent / "rest_official_native_driver.py"
    _os.execv(_sys.executable, [_sys.executable, str(_script)] + _sys.argv[1:])


import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def walk(x: Any, path: str = ""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, x


def extract_code(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    text = x.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if m:
        text = m.group(1).strip()
    starts = [text.find(m) for m in ["import torch", "from torch", "class ModelNew"] if text.find(m) != -1]
    if starts:
        text = text[min(starts):].strip()
    if "class ModelNew" not in text:
        return None
    return text + "\n"


def score_key(key: str, task_name: str, task_id: str, level: str) -> int:
    key_l = key.lower()
    score = 0
    if task_id and re.search(rf"(^|\D){re.escape(task_id)}(\D|$)", key_l):
        score += 100
    if level and level in key_l:
        score += 15
    for tok in re.split(r"[_\W]+", task_name.lower()):
        if len(tok) >= 4 and tok in key_l:
            score += 2
    return score


def llm_fallback(root: Path, task: Path, out: Path, rounds: int, temperature: float) -> int:
    print("[cuda_l1] no matching official artifact; using LLM fallback for smoke test")
    print("[cuda_l1] NOTE: fallback result is NOT official CUDA-L1 artifact performance")
    cmd = [
        sys.executable,
        str(root / "drivers" / "generic_llm_kernel_driver.py"),
        "--system", "cuda_l1",
        "--prefix", "candidate",
        "--task", str(task),
        "--out", str(out),
        "--rounds", str(rounds),
        "--seed", "20260611",
        "--temperature", str(temperature),
    ]
    return subprocess.call(cmd, cwd=str(root))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-json", default="h100.json")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    repo = root / "third_party" / "CUDA-L1"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    task = Path(args.task).resolve()
    task_name = task.stem
    m = re.match(r"(\d+)_", task.name)
    task_id = m.group(1) if m else ""
    level = next((p for p in task.parts if re.fullmatch(r"level[123]", p)), "")

    jsons = [
        repo / "optimized_cuda_code" / args.gpu_json,
        repo / "optimized_cuda_code" / "codes" / args.gpu_json,
        repo / "optimized_cuda_code" / "h100.json",
        repo / "optimized_cuda_code" / "codes" / "h100.json",
        repo / "optimized_cuda_code" / "a100.json",
        repo / "optimized_cuda_code" / "codes" / "a100.json",
        repo / "optimized_cuda_code" / "l40.json",
        repo / "optimized_cuda_code" / "codes" / "l40.json",
    ]

    best = None
    for jp in jsons:
        if not jp.exists():
            continue
        try:
            obj = json.loads(jp.read_text(errors="ignore"))
        except Exception:
            continue
        for key, val in walk(obj):
            code = extract_code(val)
            if not code:
                continue
            sc = score_key(key, task_name, task_id, level)
            if best is None or sc > best[0]:
                best = (sc, jp, key, code)

    # Only trust artifact if there is at least some signal of task match.
    if best is not None and best[0] > 0:
        dst = out / "candidate_0000.py"
        dst.write_text(best[3])
        print("[cuda_l1] selected official artifact", best[1], best[2], "score", best[0])
        print("[cuda_l1] wrote", dst)
        return 0

    if args.no_fallback:
        print("[cuda_l1] no matching ModelNew artifact")
        return 0

    return llm_fallback(root, task, out, args.rounds, args.temperature)


if __name__ == "__main__":
    raise SystemExit(main())
