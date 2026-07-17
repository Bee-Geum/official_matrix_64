import math
import torch

import triton
import triton.language as tl


# Fast, numerically accurate approximation of erf(x) (Abramowitz and Stegun 7.1.26).
# Single-precision accurate; max error ~1.5e-7 in float32.
@triton.jit
def _erf_approx(x):
    # constants in float32
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = tl.where(x >= 0, 1.0, -1.0)
    ax = tl.abs(x)
    t = 1.0 / (1.0 + p * ax)
    # polynomial evaluation
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    # TODO: exp might not be in tl; if not, replace with tl.math.exp
    e = tl.exp(-(ax * ax))
    r = 1.0 - y * e
    return sign * r


@triton.jit
def _gelu_exact_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # upcast to fp32 for math
    x32 = x.to(tl.float32)

    z = x32 * 0.7071067811865476  # 1/sqrt(2)
    erf_z = _erf_approx(z)

    y32 = 0.5 * x32 * (1.0 + erf_z)
    y = y32.to(x.dtype)

    tl.store(y_ptr + offs, y, mask=mask)


def gelu_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of GELU (exact, via high-accuracy erf approximation).
    Matches torch.nn.GELU default behavior closely for float32.
    Supports float16/bfloat16 by upcasting to float32 for math.
    """
    assert x.is_cuda, "gelu_kernel_impl requires a CUDA tensor"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"Unsupported dtype {x.dtype}; use float16/bfloat16/float32"

    y = torch.empty_like(x)
    N = x.numel()

    # Choose a block size; 1024 is a good default. You can tune to 2048/4096.
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    _gelu_exact_kernel[grid](
        x, y,
        N,
        BLOCK=BLOCK,
        num_warps=4,   # good default for 1024
        num_stages=2,
    )
    return y
