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

import requests


def clean_code(code: str) -> str:
    code = (code or "").strip()
    code = code.replace("```python", "").replace("```py", "").replace("```", "").strip()

    prefixes = [
        "Here's the optimized code:",
        "Here is the optimized code:",
        "Sure, here's the optimized code:",
        "Sure, here is the optimized code:",
        "Below is the optimized code:",
        "The optimized code is:",
    ]
    lines = code.splitlines()
    while lines and lines[0].strip() in prefixes:
        lines = lines[1:]
    code = "\n".join(lines).strip()

    for marker in [
        'if __name__ == "__main__":',
        "if __name__ == '__main__':",
        "# Example usage",
        "### Explanation",
        "## Explanation",
        "Explanation:",
    ]:
        idx = code.find(marker)
        if idx != -1:
            code = code[:idx].strip()

    return code.strip() + "\n"


def extract_code(text: str) -> str:
    text = text or ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    for block in blocks:
        if "class ModelNew" in block or "import torch" in block or "import triton" in block:
            return clean_code(block)
    if blocks:
        return clean_code(blocks[0])

    starts = [text.find(m) for m in ["import torch", "from torch", "import triton", "from triton", "class ModelNew"] if text.find(m) != -1]
    if starts:
        return clean_code(text[min(starts):])
    return clean_code(text)


def read_optional(path: Path, limit: int = 12000) -> str:
    try:
        if path.exists():
            return path.read_text(errors="ignore")[:limit]
    except Exception:
        pass
    return ""


def system_hint(system: str, root: Path) -> str:
    lower = system.lower()
    if "autokernel" in lower:
        for p in [
            root / "third_party" / "autokernel" / "kernelbench" / "program_kb.md",
            root / "third_party" / "AutoKernel" / "kernelbench" / "program_kb.md",
            root / "third_party" / "autokernel" / "README.md",
            root / "third_party" / "AutoKernel" / "README.md",
        ]:
            txt = read_optional(p)
            if txt:
                return "AutoKernel instructions:\n" + txt
    if "triton" in lower:
        return "Triton-oriented generation, but for this smoke run prefer simple Python/torch code unless the task explicitly requires Triton."
    if "cuda_l1" in lower:
        return "CUDA-L1-style candidate generation. For smoke fallback use simple torch code."
    if "ksearch" in lower:
        return "K-Search-style candidate generation. Return self-contained code."
    return "Generic GPU kernel optimization agent."


def build_prompt(system: str, task_src: str, hint: str, feedback: str) -> str:
    return f"""
You are the {system} GPU kernel optimization agent.

{hint}

Task:
Generate an implementation candidate for the provided benchmark task.

Hard requirements:
- Return exactly one valid Python source file.
- Do not include markdown fences or explanations.
- Prefer correctness and syntactic validity.
- For this smoke test, do not use Triton, torch.utils.cpp_extension.load_inline, custom CUDA, C++ extensions, or external files unless the task explicitly requires them.
- If the task is KernelBench-style, define class ModelNew(torch.nn.Module).
- If the task is not KernelBench-style, still produce a self-contained Python module.

Task content:
```python
{task_src}
```

Previous feedback:
{feedback}
"""


def call_server(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    response = requests.post(
        base + "/chat/completions",
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only Python source code. No markdown. No explanation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": float(os.environ.get("TOP_P", "0.95")),
            "max_tokens": max_tokens,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def syntax_compile_check(path: Path) -> tuple[bool, str]:
    try:
        source = path.read_text(errors="ignore")
        ast.parse(source)
        compile(source, str(path), "exec")
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def nvcc_compile_check(path: Path, out_dir: Path) -> tuple[bool, str]:
    if shutil.which("nvcc") is None:
        return False, "nvcc not found"
    exe = out_dir / (path.stem + ".out")
    cmd = ["nvcc", "-O2", str(path), "-o", str(exe)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        (out_dir / (path.stem + ".nvcc.stdout.txt")).write_text(proc.stdout)
        (out_dir / (path.stem + ".nvcc.stderr.txt")).write_text(proc.stderr)
        return proc.returncode == 0, proc.stderr[-4000:]
    except Exception as exc:
        return False, repr(exc)


def text_check(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(errors="ignore")
        return bool(text.strip()), "" if text.strip() else "empty text candidate"
    except Exception as exc:
        return False, repr(exc)


def find_candidates(cand_dir: Path, pattern: str) -> list[Path]:
    candidates = sorted(cand_dir.glob(pattern)) if pattern else []
    if not candidates:
        for pat in ["candidate_*.py", "round*_kernel.py", "*.py", "*.cu", "*.cpp", "*.txt"]:
            candidates.extend(sorted(cand_dir.glob(pat)))
    seen = []
    used = set()
    for p in candidates:
        if p.is_file() and p not in used and p.stat().st_size < 8_000_000:
            used.add(p)
            seen.append(p)
    return seen


def run_native_eval(args: argparse.Namespace) -> int:
    task = Path(args.task) if args.task else Path("unknown_task")
    cand_dir = Path(args.cand_dir or args.out or ".")
    out_dir = Path(args.out or args.out_dir or args.eval_dir or args.task_work_dir or cand_dir / "native_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_candidates(cand_dir, args.glob or "")
    verdicts: list[dict[str, Any]] = []
    n_compiled = 0

    for index, candidate in enumerate(candidates):
        verdict: dict[str, Any] = {
            "candidate": str(candidate),
            "candidate_index": index,
            "compiled": False,
            "correct": False,
            "syntax_ok": False,
            "compile_ok": False,
            "error": "",
        }

        if candidate.suffix == ".py":
            ok, err = syntax_compile_check(candidate)
            verdict["syntax_ok"] = ok
            verdict["compile_ok"] = ok
            verdict["compiled"] = ok
            verdict["error"] = err
        elif candidate.suffix in {".cu", ".cpp", ".cc", ".cuh"}:
            ok, err = nvcc_compile_check(candidate, out_dir)
            verdict["compile_ok"] = ok
            verdict["compiled"] = ok
            verdict["error"] = err
        else:
            ok, err = text_check(candidate)
            verdict["syntax_ok"] = ok
            verdict["compile_ok"] = ok
            verdict["compiled"] = ok
            verdict["error"] = err

        if verdict["compiled"]:
            n_compiled += 1
        verdicts.append(verdict)

    report = {
        "adapter": "generic_llm_kernel_driver_native_smoke",
        "task": str(task),
        "bench_root": str(args.bench_root or ""),
        "cand_dir": str(cand_dir),
        "out_dir": str(out_dir),
        "glob": args.glob or "",
        "n_candidates": len(candidates),
        "n_compiled": n_compiled,
        "n_correct": 0,
        "official_eval": 0,
        "forced_smoke_eval": 1,
        "error": "" if candidates else "no candidates found",
        "verdicts": verdicts,
    }
    summary = [{
        "task": str(task),
        "task_dir": str(out_dir),
        "n_candidates": len(candidates),
        "n_compiled": n_compiled,
        "n_correct": 0,
        "runnable_rate": n_compiled / len(candidates) if candidates else 0.0,
        "correct_rate": 0.0,
        "pass@1": 0.0,
        "fast_1": 0.0,
        "best_score": 0.0,
        "geomean_speedup": 0.0,
        "median_speedup": 0.0,
        "official_eval": 0,
        "forced_smoke_eval": 1,
        "native_adapter": "generic_llm_kernel_driver_native_smoke",
        "error": report["error"],
    }]

    (out_dir / "native_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary[0], indent=2, ensure_ascii=False))
    return 0


def run_generation(args: argparse.Namespace) -> int:
    root = Path.cwd()
    task = Path(args.task).resolve()
    out_dir = Path(args.out or args.cand_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    io_dir = out_dir / f"{args.system}_io"
    io_dir.mkdir(parents=True, exist_ok=True)

    task_src = task.read_text(errors="ignore") if task.exists() else str(task)
    hint = system_hint(args.system, root)
    model = os.environ.get("EVAL_MODEL") or os.environ.get("MODEL_ALIAS") or "qwen14b"
    max_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

    feedback = ""
    for round_id in range(max(1, args.rounds)):
        prompt = build_prompt(args.system, task_src, hint, feedback)
        start = time.time()
        raw = call_server(prompt, model=model, temperature=args.temperature, max_tokens=max_tokens)
        latency = time.time() - start

        (io_dir / f"round{round_id:03d}_raw_reply.txt").write_text(raw)
        code = extract_code(raw)

        if args.prefix == "round":
            path = out_dir / f"round{round_id:03d}_kernel.py"
        else:
            path = out_dir / f"candidate_{round_id:04d}.py"

        path.write_text(code)
        (io_dir / f"round{round_id:03d}_meta.json").write_text(json.dumps({
            "system": args.system,
            "round": round_id,
            "latency_s": latency,
            "candidate": str(path),
            "first_line": code.splitlines()[0] if code.splitlines() else "",
        }, indent=2, ensure_ascii=False))
        print(f"[{args.system}] wrote {path}")
        feedback = "Improve compilation, correctness, and speed."

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Generation args
    parser.add_argument("--task", default="")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--system", default="generic")
    parser.add_argument("--prefix", choices=["candidate", "round"], default="candidate")

    # Native-eval args used by allbench runner
    parser.add_argument("--bench-root", dest="bench_root", default=None)
    parser.add_argument("--glob", default=None)
    parser.add_argument("--out_dir", "--eval_dir", "--task_work_dir", "--output_dir", dest="out_dir", default=None)

    args, unknown = parser.parse_known_args()

    # Native eval mode is selected when the allbench runner passes eval-only args.
    if args.bench_root is not None or args.glob is not None or args.out_dir is not None:
        return run_native_eval(args)

    if not args.task:
        print("ERROR: --task is required in generation mode", file=sys.stderr)
        return 2
    return run_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
