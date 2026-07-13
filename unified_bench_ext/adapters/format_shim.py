#!/usr/bin/env python3
"""
format_shim.py -- turn a non-ModelNew benchmark task into a synthetic KernelBench-form
reference (class Model + get_inputs + get_init_inputs) so the EXISTING agent drivers
(which all take --task <KB.py> and emit a ModelNew) can be reused unchanged.

Without this, every agent would need a new per-benchmark driver. With it, "all agents"
plug into op_fill / triton_standalone / parallel_prompt benchmarks for free; the agent
still emits a ModelNew, and you evaluate that ModelNew either with instrumented_final_eval
(if you want a uniform speedup-vs-torch number) or hand the candidate to the benchmark's
native adapter.

Shim kinds:
  op        BackendBench operator -> Model.forward calls torch.<op>(...) on representative
            inputs. CONCRETE below (works for elementwise / binary / common ops).
  triton    TritonBench-G kernel -> wrap its provided PyTorch reference as Model.forward.
            Seam: locate the reference torch fn the benchmark ships for the op.
  parallel  ParEval problem -> wrap the serial reference as Model.forward.
            Seam: pull the reference implementation from the problem spec.

Usage:
  python3 adapters/format_shim.py --kind op --task add \
      --out runs/.../shim_task.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Representative input recipes for common ATen ops (extend as needed).
# Each entry: dict(nargs, init_args, inputs, call) where {core} is the normalized op name.
OP_RECIPES = {
    "default_binary": {
        "nargs": 2,
        "init_args": "[]",
        "inputs": "[torch.randn(4096, 4096, device='cuda'), torch.randn(4096, 4096, device='cuda')]",
        "call": "torch.{core}(x, y)",
    },
    "default_unary": {
        "nargs": 1,
        "init_args": "[]",
        "inputs": "[torch.randn(4096, 4096, device='cuda')]",
        "call": "torch.{core}(x)",
    },
    "matmul": {
        "nargs": 2,
        "init_args": "[]",
        "inputs": "[torch.randn(2048, 2048, device='cuda'), torch.randn(2048, 2048, device='cuda')]",
        "call": "torch.matmul(x, y)",
    },
}

UNARY = {"relu", "gelu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "acos",
         "acosh", "sin", "cos", "neg", "softmax"}
BINARY = {"add", "sub", "mul", "div", "maximum", "minimum", "pow"}


def normalize_op(op: str) -> str:
    """'aten.relu.default' / 'torch.ops.aten.add.Tensor' / 'relu' -> 'relu'.

    Strips the namespace prefix (aten., torch., torch.ops.aten.) and the trailing
    overload selector (.default, .Tensor, .Scalar, .out, ...). torch.<core> is the
    public functional that the wrapped reference calls.
    """
    core = op.strip()
    for pref in ("torch.ops.aten.", "torch.ops.", "torch.", "aten."):
        if core.startswith(pref):
            core = core[len(pref):]
            break
    parts = core.split(".")
    # drop a trailing overload tag (lowercase 'default'/'out' or Capitalized 'Tensor'/'Scalar')
    if len(parts) > 1 and (parts[-1] in ("default", "out") or parts[-1][:1].isupper()):
        parts = parts[:-1]
    return parts[0] if parts else core


def op_recipe(op: str):
    core = normalize_op(op)
    if core in ("matmul", "mm", "bmm", "addmm"):
        return "matmul", OP_RECIPES["matmul"]
    if core in UNARY:
        return "default_unary", OP_RECIPES["default_unary"]
    return "default_binary", OP_RECIPES["default_binary"]  # safe default for elementwise


def render_op_shim(op: str) -> str:
    core = normalize_op(op)
    _kind, rec = op_recipe(op)
    sig = "x, y" if rec["nargs"] == 2 else "x"
    call = rec["call"].format(core=core)
    return f'''import torch
import torch.nn as nn

class Model(nn.Module):
    """Synthetic KernelBench-form reference wrapping BackendBench op '{op}' (torch.{core})."""
    def __init__(self):
        super().__init__()

    def forward(self, {sig}):
        return {call}

def get_inputs():
    return {rec["inputs"]}

def get_init_inputs():
    return {rec["init_args"]}
'''


def render_triton_shim(task: str) -> str:
    # SEAM: TritonBench-G ships a PyTorch reference per op; import/inline it here.
    return f'''import torch
import torch.nn as nn
# TODO(triton shim): replace the body with TritonBench-G's reference torch impl for '{task}'.
# from tritonbench.references import {task} as ref_fn   # adjust import to the repo

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        raise NotImplementedError("inline TritonBench-G reference for {task}")

def get_inputs():
    # TODO: use the benchmark's example_inputs for {task}
    return [torch.randn(4096, device='cuda')]

def get_init_inputs():
    return []
'''


def render_parallel_shim(task: str) -> str:
    # SEAM: ParEval problems include a serial/reference solution; wrap it.
    return f'''import torch
import torch.nn as nn
# TODO(parallel shim): wrap ParEval's reference solution for problem '{task}'.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args):
        raise NotImplementedError("inline ParEval reference for {task}")

def get_inputs():
    return [torch.randn(1 << 20, device='cuda')]

def get_init_inputs():
    return []
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["op", "triton", "parallel"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    render = {"op": render_op_shim, "triton": render_triton_shim, "parallel": render_parallel_shim}[args.kind]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.task))
    print(f"[shim] wrote {args.kind} shim for '{args.task}' -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
