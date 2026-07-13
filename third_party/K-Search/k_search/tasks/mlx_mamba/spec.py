"""Spec and workloads for the MLX Mamba selective scan forward task.

This is an Apple-Silicon / MLX-only task.
The goal is to implement an optimized forward selective scan (SSM recurrence)
with high parallelism across (batch, dim) and within the sequence.

We intentionally keep the spec compact:
- use-case + evaluation contract
- parallelism guidance (no CUDA translation material)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DTypeName = Literal["float16", "float32"]


@dataclass(frozen=True)
class MambaSelectiveScanFwdWorkload:
    batch: int
    dim: int
    dstate: int
    seqlen: int
    ngroups: int = 1
    has_z: bool = True
    delta_softplus: bool = True
    dtype: DTypeName = "float32"

    @property
    def label(self) -> str:
        return (
            f"B{self.batch}_D{self.dim}_N{self.dstate}_L{self.seqlen}_G{self.ngroups}_"
            f"{'z' if self.has_z else 'noz'}_{self.dtype}"
        )


# Defaults chosen to be representative but still safe on Apple Silicon.
# These shapes match common Mamba-ish settings (dstate=16) and include longer sequences
# to reward parallelism over the sequence dimension.
MAMBA_SELECTIVE_SCAN_FWD_WORKLOADS: list[MambaSelectiveScanFwdWorkload] = [
    MambaSelectiveScanFwdWorkload(batch=1, dim=768, dstate=16, seqlen=512, ngroups=1, has_z=True, dtype="float32"),
    MambaSelectiveScanFwdWorkload(batch=1, dim=768, dstate=16, seqlen=2048, ngroups=1, has_z=True, dtype="float32"),
    MambaSelectiveScanFwdWorkload(batch=1, dim=768, dstate=16, seqlen=4096, ngroups=1, has_z=True, dtype="float32"),
    MambaSelectiveScanFwdWorkload(batch=1, dim=2048, dstate=16, seqlen=1024, ngroups=1, has_z=True, dtype="float32"),
    MambaSelectiveScanFwdWorkload(batch=4, dim=768, dstate=16, seqlen=4096, ngroups=1, has_z=True, dtype="float32"),
    MambaSelectiveScanFwdWorkload(batch=4, dim=768, dstate=16, seqlen=1024, ngroups=1, has_z=True, dtype="float32"),
]


def get_definition_text_mlx() -> str:
    return r"""# MLX Task: Mamba Selective Scan Forward (Apple Silicon)

## Objective
Implement the Mamba selective scan forward pass (SSM recurrence) efficiently.

Given tensors:
- u: (B, D, L)
- delta: (B, D, L)
- A: (D, N)
- B: either (B, N, L) or (B, G, N, L)
- C: either (B, N, L) or (B, G, N, L)
- optional D: (D,)
- optional z: (B, D, L)
- optional delta_bias: (D,)

Compute (float32 math is recommended for stability):

1) $\delta' = \text{softplus}(\delta + \text{delta\_bias})$  (or identity if `delta_softplus=False`)
2) For each timestep $t$ and state index $n$:
   - $a_{t,n} = \exp(\delta'_t \cdot A_n)$
   - $b_{t,n} = \delta'_t \cdot u_t \cdot B_{t,n}$
   - $h_{t,n} = a_{t,n} \cdot h_{t-1,n} + b_{t,n}$, with $h_{-1}=0$
3) $y_t = \sum_n h_{t,n} \cdot C_{t,n}$
4) If `D` is provided: $y_t \leftarrow y_t + D \cdot u_t$
5) If `z` is provided: $y_t \leftarrow y_t \cdot \text{silu}(z_t)$ where $\text{silu}(x)=x\cdot\sigma(x)$

Return `out = y` with shape (B, D, L).

## Entrypoint (required)
Return Python code defining exactly this function:

    def run(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
        ...
        return out

## Performance goal
Beat the provided naive baseline by reducing kernel launches and exploiting parallelism.

### Parallelism guidance (important)
- Parallelize over `(batch, dim)` (each (b, d) slice is independent).
- The sequence dimension `L` is a scan/reduction-style dependency, but you can still
  **increase parallelism** by using a *parallel prefix scan* formulation.
- The recurrence can be expressed with an associative operator on pairs $(a, b)$:

  $$(a_1,b_1) \circ (a_0,b_0) = (a_1 a_0,\; a_1 b_0 + b_1)$$

  so a block-wide parallel scan over timesteps can compute the same result as a sequential loop.

Practical implementation options (choose what works):
- Use `mx.fast.metal_kernel` and implement a chunked parallel scan inside one threadgroup.
  A common mapping is: one threadgroup per `(batch, dim)`, with threads cooperatively processing
  the sequence in chunks (e.g., `CHUNK_SIZE = NUM_THREADS * ITEMS_PER_THREAD`).
- If you stay in pure MLX ops, avoid Python loops over `L` whenever possible.

## Notes
- Correctness is checked against a naive reference within tolerances.
- Evaluation reports submission latency, baseline latency, and speedup.
- Keep the API and output shape exactly as specified.
- Operation like softplus are under mlx.nn.softplus, but you can also implement them with core ops for more control.

## MLX Python API reference (quick, for this task)
This section is meant to be self-contained because the coding agent may not have web access.
Upstream docs (Python API): https://ml-explore.github.io/mlx/build/html/python/

### Imports / conventions
- Prefer:
  - `import mlx.core as mx`
  - `import mlx.nn as nn` (optional, for `nn.softplus`, `nn.silu`, etc.)
- MLX execution is lazy; materialize results when benchmarking or when you need actual values:
  - `mx.eval(out)` (or `mx.eval(tree_of_arrays)`)
  - `mx.eval(a, b, c)` is allowed too.

### Core types / dtypes
- Arrays are `mx.array` objects.
- Common dtypes you may need:
  - floats: `mx.float16`, `mx.float32`
  - ints: `mx.int32`, `mx.int64`
  - bool: `mx.bool_`
- Casting: `x.astype(mx.float32)` (there is no `mx.astype(...)` helper in `mlx.core`)

### Array creation (MLX vs NumPy/JAX conventions)
MLX is **not** a drop-in NumPy/JAX clone; some familiar creators do not exist.

Common creators that DO exist in `mlx.core`:
- `mx.zeros(shape, dtype=...)`, `mx.ones(shape, dtype=...)`, `mx.full(shape, fill_value, dtype=...)`
- `mx.zeros_like(x)`, `mx.ones_like(x)`
- `mx.arange(start, stop=None, step=1, dtype=...)`, `mx.linspace(start, stop, num, dtype=...)`
- `mx.eye(n, m=None, k=0, dtype=...)`, `mx.identity(n, dtype=...)`

Common “NumPy-isms” that DO NOT exist (and what to use instead):
- `mx.empty(shape, dtype=...)` does not exist
  - Use `mx.zeros(shape, dtype=...)` or `mx.full(shape, fill_value, dtype=...)`, then overwrite.
- `mx.empty_like(x)` does not exist
  - Use `mx.zeros_like(x)` (or `mx.zeros(x.shape, dtype=x.dtype)`), then overwrite.
- `mx.full_like(x, fill_value)` does not exist
  - Use `mx.full(x.shape, fill_value, dtype=x.dtype)`.

Random sampling (also a common source of hallucinated APIs):
- Use `mx.random.normal(shape, ...)`, `mx.random.uniform(shape, ...)`, `mx.random.randint(low, high, shape, ...)`
- There is no `mx.randn(...)` / `mx.rand(...)` convenience in `mlx.core`.

### Common ops you are expected to use
- Elementwise: `mx.exp`, `mx.log`, `mx.maximum`, `mx.minimum`, `mx.where`, `mx.sigmoid`, `mx.tanh`, `mx.sqrt`, `mx.rsqrt`
- Reductions: `mx.sum(x, axis=..., keepdims=...)`, `mx.mean`, `mx.max`, `mx.min`
- Shape/axis: `mx.reshape(x, shape)`, `mx.transpose(x, axes)`, `mx.moveaxis(x, source, destination)`, `mx.expand_dims`, `mx.squeeze`, `mx.broadcast_to`
- Matmul / contractions: `mx.matmul(a, b)`, `mx.einsum(subscripts, *operands)`

### Verified MLX API names (subset; prefer these exact spellings)
DO NOT invent APIs. If something isn’t listed here, prefer building it from these primitives or use a custom Metal kernel.

Transforms / execution:
- `mx.eval`, `mx.async_eval`, `mx.compile`, `mx.disable_compile`, `mx.enable_compile`

Array creation:
- `mx.zeros`, `mx.ones`, `mx.full`
- `mx.zeros_like`, `mx.ones_like`
- `mx.arange`, `mx.linspace`
- `mx.eye`, `mx.identity`
- `mx.random.normal`, `mx.random.uniform`, `mx.random.randint`, `mx.random.bernoulli`, `mx.random.categorical`

Scan-like cumulative ops:
- `mx.cumsum(x, axis=..., reverse=False, inclusive=True)`
- `mx.cumprod(x, axis=..., reverse=False, inclusive=True)`
- `mx.cummax(x, axis=..., reverse=False, inclusive=True)`
- `mx.cummin(x, axis=..., reverse=False, inclusive=True)`

Indexing / gather helpers (often useful to avoid Python loops):
- `mx.take(x, indices, axis=...)`, `mx.take_along_axis(x, indices, axis=...)`
- `mx.slice(x, start_indices, axes, slice_size, ...)`, `mx.slice_update(x, update, start_indices, axes, ...)`

Reshape / layout helpers:
- `mx.flatten(x, start_axis=0, end_axis=-1)`, `mx.unflatten(x, axis, shape)`
- `mx.stack(arrays, axis=...)`, `mx.concatenate(arrays, axis=...)`, `mx.split(x, indices_or_sections, axis=...)`
- `mx.contiguous(x, allow_col_major=False)`

Math / reductions (selected):
- `mx.exp`, `mx.expm1`, `mx.log`, `mx.log1p`, `mx.logsumexp`, `mx.logcumsumexp`
- `mx.sum`, `mx.mean`, `mx.prod`, `mx.max`, `mx.min`
- `mx.softmax(x, axis=...)`, `mx.sigmoid`, `mx.tanh`
- `mx.clip`, `mx.where`, `mx.abs`, `mx.square`, `mx.sqrt`, `mx.rsqrt`

Fast path / custom kernels:
- `mx.fast.metal_kernel` (and call the returned kernel)
- `mx.fast.scaled_dot_product_attention` (not needed for this task, but exists)

### Activations used here
- Softplus: `nn.softplus(x)` (or implement manually with core ops if needed)
- SiLU: `nn.silu(x)` (also called swish)

### Compilation (important for performance + correctness)
- Signature (most relevant args):
  - `mx.compile(fun, inputs=None, outputs=None, shapeless=False) -> callable`
  - `@mx.compile` decorator form is supported.
- Recompilation happens when you change number of inputs, dtypes, ndim; shape changes can also trigger recompilation unless `shapeless=True`.
- Compiled functions should be pure (no printing / side effects). First call traces with placeholder arrays.
- If you must debug a compiled function, you can disable compilation globally (outside this submission code path) via `mx.disable_compile()`.

### Custom Metal kernels (the key MLX-specific API)
You can JIT a custom Metal kernel from a source-string *function body*.

Factory signature (key params):
- `mx.fast.metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=True, atomic_outputs=False) -> callable`

Important: `source` is ONLY the kernel body; MLX generates the full `[[kernel]]` signature.

    kernel = mx.fast.metal_kernel(
        name="my_kernel",
        input_names=[...],
        output_names=[...],
      source='''...Metal kernel BODY only (no signature)...''',
        header="",  # optional helper code / includes
        ensure_row_contiguous=True,  # default; set False to use *_shape/_strides/_ndim
        atomic_outputs=False,
    )

Then call it like:

Call signature (most commonly used args):
- `kernel(inputs=[...], template=[...], grid=(gx,gy,gz), threadgroup=(tx,ty,tz), output_shapes=[...], output_dtypes=[...], init_value=..., verbose=...) -> list[mx.array]`

    outs = kernel(
        inputs=[inp0, inp1, ...],
        template=[("T", mx.float32), ("FLAG", True), ("K", 128)],
        grid=(grid_x, grid_y, grid_z),
        threadgroup=(tg_x, tg_y, tg_z),
        output_shapes=[shape0, shape1, ...],
        output_dtypes=[dtype0, dtype1, ...],
        init_value=0,   # optional; useful when accumulating into outputs
        verbose=False,  # set True to print generated Metal for debugging
    )
    out0 = outs[0]

Notes:
- Build the kernel once (module-level cache) and reuse it; creating kernels repeatedly adds big overhead.
- `ensure_row_contiguous=True` may copy non-contiguous inputs (simplifies indexing).
- If `ensure_row_contiguous=False` and your `source` references `inp_shape` / `inp_strides` / `inp_ndim`, MLX provides them so you can index correctly.
- Output arrays are row-contiguous.

Minimal (invented) pattern for caching a kernel:

  _KERNEL = None
  def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
      _KERNEL = mx.fast.metal_kernel(
        name="...",
        input_names=[...],
        output_names=[...],
        source='''...''',
      )
    return _KERNEL

### MLX gotchas seen in this task (read carefully)
These are common failure modes from the evaluation logs.

1) **`x.shape` is a Python tuple**
- `x.shape` returns a tuple like `(B, D, L)`.
- Do NOT do `x.shape.ndim` (that fails: tuple has no `.ndim`).
- Correct ways:
  - `x.ndim` (preferred)
  - `len(x.shape)`

2) **Broadcasting a `(D,)` vector into `(B, D, L)`**
MLX does not automatically guess the middle axis for 1D vectors.
If `delta` is `(B, D, L)` and `delta_bias` is `(D,)`, do:

    delta = delta + delta_bias[None, :, None]

Same for `D` (skip connection vector):

    out = out + D[None, :, None] * u

3) **Broadcasting `A: (D, N)` against `delta: (B, D, L)`**
To form `a = exp(delta * A)` with result shape `(B, D, L, N)`, you must expand BOTH sides:

    a = mx.exp(delta[:, :, :, None] * A[None, :, None, :])

If you write `delta[..., None] * A` it will usually FAIL to broadcast.

4) **`mlx.nn` is a separate module (no `mx.nn.*`)**
- `mx` is `mlx.core`.
- There is no `mx.nn`.
- If you need softplus/silu: `import mlx.nn as nn` then use:
  - `nn.softplus(x)`
  - `nn.silu(x)`

5) **Avoid guessing op names**
These names are NOT in `mlx.core` (and have caused failures):
- `mx.associative_scan`  (don’t use)
- `mx.cumulative_product` (don’t use)

Use the actual MLX scan-like ops:
- cumulative product: `mx.cumprod(x, axis=...)`
- cumulative sum: `mx.cumsum(x, axis=...)`

If you need an associative/prefix scan for the recurrence, implement it explicitly (e.g., chunked scan in a `mx.fast.metal_kernel`) rather than relying on a non-existent high-level op.

6) **If you see `AttributeError: module 'mlx.core' has no attribute ...`**
This usually means the code used a NumPy/JAX/Torch name that MLX does not provide.
Common ones seen in this task:
- `mx.empty` / `mx.empty_like` (use `mx.zeros` / `mx.zeros_like` instead)
- `mx.full_like` (use `mx.full(x.shape, fill_value, dtype=x.dtype)`)
- `mx.randn` / `mx.rand` (use `mx.random.normal` / `mx.random.uniform`)
"""
