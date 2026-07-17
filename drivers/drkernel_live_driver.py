#!/usr/bin/env python3
"""Run the real Dr.Kernel RL model (hkust-nlp/drkernel-14b) against KernelGYM.

Replaces the shim that routed drkernel into the generic LLM driver against
Qwen -- the registry even claimed model=qwen14b, so "drkernel" was neither the
RL model nor the gym.

Dr.Kernel has no standalone inference entry point: generation exists only as the
rollout half of KernelGYM's verl RL trainer (kernel.main_grading, Hydra + Ray).
Rather than stand up a training cluster to sample one kernel, this drives the
released RL model directly and reproduces the repo's own multi-turn contract:

  - no system prompt                        (prompt_config/multi_turn_kernel.yaml: prompt: null)
  - turn 2+ feeds back the gym's verdict    (its tool_response template, verbatim)
  - ```python``` block extraction           (main_grading.py's _extract_code regex)
  - the gym itself scores each turn         (KernelGYM /evaluate -- the same
                                             environment the model was trained against)

So the RL model and the gym are both real; only verl's trainer harness, which
exists to compute gradients, is not in the loop.

Contract:
    --task <task.py> --cand_dir <dir> [--rounds N] [--seed S] [--temperature T]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
GYM_CONFIG = (ROOT / "third_party" / "KernelGYM" / "drkernel" / "kernel" /
              "config" / "prompt_config" / "multi_turn_kernel.yaml")

DEFAULT_BASE_URL = "http://127.0.0.1:8004/v1"
DEFAULT_MODEL = "drkernel14b"
DEFAULT_GYM_URL = "http://127.0.0.1:10907"

# main_grading.py's tool_response template, verbatim.
FEEDBACK_TEMPLATE = """Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step."""

FIRST_TURN_TEMPLATE = """You are given a PyTorch model. Write an optimized Triton implementation named `ModelNew` whose output numerically matches the reference `Model`.

Reference model:
```python
{ref}
```

Return the implementation as a single ```python``` block. Let's think step by step."""


def extract_code(text: str) -> str:
    """main_grading.py:_extract_code"""
    if not text:
        return ""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def gym_evaluate(gym_url: str, task_id: str, ref: str, kernel: str, timeout: int = 600) -> dict:
    payload = {
        "task_id": task_id,
        "reference_code": ref,
        "kernel_code": kernel,
        "backend": "triton",
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "timeout": 300,
        "force_refresh": True,
    }
    request = urllib.request.Request(
        gym_url.rstrip("/") + "/evaluate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        return {"status": "error", "error": f"HTTP {exc.code}: {exc.read()[:400]!r}"}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)}


def feedback_of(verdict: dict) -> str:
    keep = ("status", "compiled", "correctness", "decoy_kernel", "speedup",
            "reference_runtime", "kernel_runtime", "error", "error_code")
    return json.dumps({k: verdict[k] for k in keep if k in verdict}, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand", "--cand_dir", "--cand-dir", dest="cand_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=8192)
    args, _unknown = ap.parse_known_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.cand_dir or args.out or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not task.exists():
        print(f"[drkernel] ERROR: task not found: {task}", file=sys.stderr)
        return 2

    base_url = os.environ.get("DRKERNEL_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("DRKERNEL_MODEL", DEFAULT_MODEL)
    gym_url = os.environ.get("KERNELGYM_URL", DEFAULT_GYM_URL)

    # The multi-turn loop is only faithful if the gym is actually answering.
    try:
        with urllib.request.urlopen(gym_url.rstrip("/") + "/health", timeout=30) as response:
            healthy = json.load(response).get("status") == "healthy"
    except Exception as exc:
        healthy = False
        print(f"[drkernel] gym health check failed: {exc!r}", file=sys.stderr)
    if not healthy:
        print(f"[drkernel] ERROR: KernelGYM not healthy at {gym_url}; refusing to run "
              f"a 'gym' loop without the gym", file=sys.stderr)
        return 2

    ref = task.read_text(errors="ignore")
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    io_dir = out_dir / "drkernel_io"
    io_dir.mkdir(parents=True, exist_ok=True)

    print(f"[drkernel] LIVE RL model {model} @ {base_url} + KernelGYM @ {gym_url}", flush=True)

    # prompt: null -- upstream sends no system message.
    messages: list[dict] = [{"role": "user", "content": FIRST_TURN_TEMPLATE.format(ref=ref)}]
    turns: list[dict] = []
    gym_usable_turns: list[bool] = []
    best: tuple[float, str] | None = None
    produced: list[Path] = []

    for turn in range(max(1, args.rounds)):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.95,
        )
        raw = resp.choices[0].message.content or ""
        (io_dir / f"turn{turn:03d}_raw_reply.txt").write_text(raw, errors="ignore")

        code = extract_code(raw)
        path = out_dir / f"candidate_{turn:04d}.py"
        path.write_text(code, errors="ignore")
        produced.append(path)

        verdict = gym_evaluate(gym_url, f"drkernel_{task.stem}_t{turn}", ref, code)
        speedup = float(verdict.get("speedup") or 0.0)
        ok = bool(verdict.get("correctness")) and not verdict.get("decoy_kernel", False)

        # KernelGYM only speaks KernelBench (Model/ModelNew). Handed a task in any
        # other format it returns VALIDATION_ERROR without ever running the kernel,
        # so the feedback fed into the next turn is noise, not a verdict -- and this
        # cell is then "the RL model alone", not "RL model + gym". Say so rather than
        # letting the loop report a gym-shaped failure the kernel did not earn.
        if verdict.get("error_code") == "VALIDATION_ERROR" and not verdict.get("compiled"):
            gym_usable = False
            print(f"[drkernel] turn {turn}: gym returned VALIDATION_ERROR -- this task is "
                  f"not KernelBench-format, so gym feedback is unavailable "
                  f"(the benchmark's own oracle still scores the kernel)", flush=True)
        else:
            gym_usable = True
        gym_usable_turns.append(gym_usable)

        print(f"[drkernel] turn {turn}: compiled={verdict.get('compiled')} "
              f"correct={verdict.get('correctness')} decoy={verdict.get('decoy_kernel')} "
              f"speedup={speedup:.4f}", flush=True)

        turns.append({"turn": turn, "candidate": str(path), "verdict": verdict})
        if ok and (best is None or speedup > best[0]):
            best = (speedup, code)

        if turn + 1 >= max(1, args.rounds):
            break
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": FEEDBACK_TEMPLATE.format(feedback=feedback_of(verdict))})

    if best is not None:
        (out_dir / "candidate_0000.py").write_text(best[1], errors="ignore")

    gym_worked = any(gym_usable_turns)
    (io_dir / "provenance.json").write_text(json.dumps({
        "agent": "drkernel",
        "mode": "rl_model_plus_gym" if gym_worked else "rl_model_only_gym_unusable",
        "gym_usable": gym_worked,
        "gym_usable_note": (
            "" if gym_worked else
            "KernelGYM speaks KernelBench (Model/ModelNew) only; this task is in another "
            "format, so it returned VALIDATION_ERROR without running the kernel. The "
            "multi-turn feedback loop is therefore NOT reproduced for this cell -- what "
            "ran is the RL model on its own. The benchmark's official oracle still scored "
            "the kernel, so `correct` remains a real result."
        ),
        "model_repo": "hkust-nlp/drkernel-14b",
        "served_as": model,
        "base_url": base_url,
        "gym_url": gym_url,
        "gym": "KernelGYM /evaluate (the RL environment the model was trained against)",
        "prompt_contract": str(GYM_CONFIG),
        "not_reproduced": "verl trainer harness (gradient computation); inference does not use it",
        "rounds": max(1, args.rounds),
        "best_speedup": best[0] if best else 0.0,
        "turns": turns,
    }, indent=2, ensure_ascii=False))

    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
