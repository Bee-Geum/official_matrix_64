#!/usr/bin/env python3
"""
autokernel_qwen_loop.py -- FAITHFUL reproduction driver for the AutoKernel agent.

AutoKernel ships no LLM client: its README says "spin up Claude, Codex, or any coding
agent in this directory" to read program.md and iteratively edit kernel.py. This driver
IS that coding-agent loop, backed by the same Qwen2.5-Coder-14B endpoint used for the
other agents (controlled backbone). It uses AutoKernel's OWN artifacts verbatim:
  - kernelbench/program_kb.md   -> system prompt (AutoKernel's optimization playbook)
  - kernelbench/bridge.py setup -> reference.py + starter kernel.py
  - kernelbench/bench_kb.py      -> the evaluator (correctness + speedup, keep/revert signal)

Loop: setup -> {ask Qwen for a new kernel.py -> bench_kb.py -> keep if correct & faster
else revert} x N -> emit the best kernel.py for official-oracle scoring.

Usage:
  python3 autokernel_qwen_loop.py --task <KernelBench_task.py> --problem-id 19 \
      --iters 4 --backend cuda --out-best /path/to/best_kernel.py
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

AK = Path("/home/bi_geum/official_matrix_64/third_party/autokernel").resolve()
KB = AK / "kernelbench"
KB_ACTIVE = AK / "workspace" / "kb_active"
KERNEL_PY = AK / "kernel.py"
PLAYBOOK = KB / "program_kb.md"

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("EVAL_MODEL", "qwen14b")


def call_qwen(system: str, user: str, max_tokens: int = 6144, temperature: float = 0.2) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key="EMPTY")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temperature, top_p=0.95,
    )
    return r.choices[0].message.content or ""


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        # take the block that defines ModelNew, else the longest
        for blk in m:
            if "class ModelNew" in blk:
                return blk.strip()
        return max(m, key=len).strip()
    return text.strip()


def run_setup(task: str, problem_id: int, backend: str) -> None:
    subprocess.run(
        [sys.executable, str(KB / "bridge.py"), "setup", "--level", "1",
         "--problem", str(problem_id), "--backend", backend,
         "--source", "file", "--file-path", str(Path(task).resolve())],
        cwd=str(AK), check=True, capture_output=True, text=True,
    )


def run_bench(ext_dir: Path, timeout: int = 400) -> dict:
    """Run AutoKernel's bench_kb.py on the current kernel.py; return last results entry."""
    env = dict(os.environ)
    env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "12.0")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    # AutoKernel's playbook tells the agent to `from kernels.cuda._compile import compile_cuda`,
    # so the AutoKernel project root must be importable when bench_kb.py loads kernel.py.
    env["PYTHONPATH"] = str(AK) + os.pathsep + env.get("PYTHONPATH", "")
    # compile_cuda cold-compiles inside correctness trial 0 (~33s on Blackwell); the default
    # 30s per-trial timeout is too tight. bench_kb.py reads AK_TRIAL_TIMEOUT (repro patch).
    env["AK_TRIAL_TIMEOUT"] = env.get("AK_TRIAL_TIMEOUT", "200")
    res_path = KB_ACTIVE / "results.json"
    before = len(json.loads(res_path.read_text())) if res_path.exists() else 0
    try:
        p = subprocess.run(
            [sys.executable, str(KB / "bench_kb.py"), "--skip-stability", "--skip-determinism"],
            cwd=str(AK), env=env, timeout=timeout, capture_output=True, text=True,
        )
        tail = (p.stdout or "")[-1500:] + (p.stderr or "")[-800:]
    except subprocess.TimeoutExpired as e:
        tail = "TIMEOUT/crash during bench (" + str(timeout) + "s)"
    entry = {"correctness": "FAIL", "speedup": 0.0, "kernel_time_ms": 0.0, "note": tail[-1200:]}
    if res_path.exists():
        hist = json.loads(res_path.read_text())
        if len(hist) > before:
            entry.update(hist[-1])
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--problem-id", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--backend", choices=["cuda", "triton"], default="cuda")
    ap.add_argument("--out-best", required=True)
    ap.add_argument("--ext-base", default="/home/bi_geum/official_matrix_64/results/repro/_extdir_ak")
    args = ap.parse_args()

    print(f"[setup] problem_id={args.problem_id} backend={args.backend} task={args.task}", flush=True)
    run_setup(args.task, args.problem_id, args.backend)
    playbook = PLAYBOOK.read_text(encoding="utf-8")
    reference = (KB_ACTIVE / "reference.py").read_text(encoding="utf-8")
    starter = KERNEL_PY.read_text(encoding="utf-8")

    best = {"speedup": -1.0, "correct": False, "code": None, "iter": -1}
    last_result_summary = "(no benchmark yet; this is the starter)"
    history = []

    for it in range(args.iters):
        user = f"""You are optimizing a KernelBench problem. Produce a COMPLETE, self-contained
`kernel.py` that defines a `ModelNew(torch.nn.Module)` numerically matching the reference
`Model` (atol=rtol=1e-2) and faster than it. Use {args.backend.upper()} (CUDA C++ via
torch.utils.cpp_extension.load_inline, or Triton @triton.jit). Return ONE python code block only.

# Reference (Model + get_inputs) -- workspace/kb_active/reference.py
```python
{reference}
```

# Current kernel.py
```python
{KERNEL_PY.read_text(encoding='utf-8')}
```

# Last benchmark result
{last_result_summary}

Return the full new kernel.py now."""
        print(f"\n[iter {it}] querying Qwen...", flush=True)
        try:
            raw = call_qwen(playbook, user)
        except Exception as e:
            print(f"[iter {it}] LLM error: {e}", flush=True); continue
        code = extract_code(raw)
        if "class ModelNew" not in code:
            print(f"[iter {it}] no ModelNew in reply; skipping", flush=True); continue

        backup = KERNEL_PY.read_text(encoding="utf-8")
        KERNEL_PY.write_text(code, encoding="utf-8")
        ext_dir = Path(f"{args.ext_base}_{it}_{os.getpid()}")
        shutil.rmtree(ext_dir, ignore_errors=True); ext_dir.mkdir(parents=True, exist_ok=True)
        r = run_bench(ext_dir)
        correct = str(r.get("correctness")) == "PASS"
        speedup = float(r.get("speedup") or 0.0)
        print(f"[iter {it}] correctness={r.get('correctness')} speedup={speedup:.3f} "
              f"kt={r.get('kernel_time_ms')}", flush=True)
        history.append({"iter": it, "correct": correct, "speedup": speedup})

        improved = correct and speedup > best["speedup"]
        if improved:
            best = {"speedup": speedup, "correct": True, "code": code, "iter": it}
            last_result_summary = f"correctness=PASS speedup={speedup:.3f}x (KEPT as best)"
            print(f"[iter {it}] KEEP (new best speedup {speedup:.3f})", flush=True)
        else:
            # revert kernel.py to last good state for the next edit
            KERNEL_PY.write_text(best["code"] or backup, encoding="utf-8")
            reason = "incorrect" if not correct else f"not faster than best {best['speedup']:.3f}"
            last_result_summary = (f"correctness={r.get('correctness')} speedup={speedup:.3f}x "
                                   f"-> REVERTED ({reason}). Error/notes:\n{str(r.get('note',''))[:600]}")
            print(f"[iter {it}] REVERT ({reason})", flush=True)

    out = Path(args.out_best)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best["code"] is not None:
        out.write_text(best["code"], encoding="utf-8")
        print(f"\n[done] best: iter={best['iter']} speedup={best['speedup']:.3f} -> {out}", flush=True)
    else:
        # no correct kernel found; emit the last starter so oracle still has a candidate
        out.write_text(starter, encoding="utf-8")
        print(f"\n[done] NO correct kernel found in {args.iters} iters; wrote starter to {out}", flush=True)
    (out.parent / "autokernel_loop_history.json").write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
