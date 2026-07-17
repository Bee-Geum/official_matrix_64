import types
_CANDIDATE_CODE = 'import math\nimport torch\n\nimport triton\nimport triton.language as tl\n\n\n# Fast, numerically accurate approximation of erf(x) (Abramowitz and Stegun 7.1.26).\n# Single-precision accurate; max error ~1.5e-7 in float32.\n@triton.jit\ndef _erf_approx(x):\n    # constants in float32\n    a1 = 0.254829592\n    a2 = -0.284496736\n    a3 = 1.421413741\n    a4 = -1.453152027\n    a5 = 1.061405429\n    p = 0.3275911\n\n    sign = tl.where(x >= 0, 1.0, -1.0)\n    ax = tl.abs(x)\n    t = 1.0 / (1.0 + p * ax)\n    # polynomial evaluation\n    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t\n    # TODO: exp might not be in tl; if not, replace with tl.math.exp\n    e = tl.exp(-(ax * ax))\n    r = 1.0 - y * e\n    return sign * r\n\n\n@triton.jit\ndef _gelu_exact_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n\n    x = tl.load(x_ptr + offs, mask=mask, other=0.0)\n    # upcast to fp32 for math\n    x32 = x.to(tl.float32)\n\n    z = x32 * 0.7071067811865476  # 1/sqrt(2)\n    erf_z = _erf_approx(z)\n\n    y32 = 0.5 * x32 * (1.0 + erf_z)\n    y = y32.to(x.dtype)\n\n    tl.store(y_ptr + offs, y, mask=mask)\n\n\ndef gelu_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n    """\n    Triton implementation of GELU (exact, via high-accuracy erf approximation).\n    Matches torch.nn.GELU default behavior closely for float32.\n    Supports float16/bfloat16 by upcasting to float32 for math.\n    """\n    assert x.is_cuda, "gelu_kernel_impl requires a CUDA tensor"\n    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \\\n        f"Unsupported dtype {x.dtype}; use float16/bfloat16/float32"\n\n    y = torch.empty_like(x)\n    N = x.numel()\n\n    # Choose a block size; 1024 is a good default. You can tune to 2048/4096.\n    BLOCK = 1024\n    grid = (triton.cdiv(N, BLOCK),)\n\n    _gelu_exact_kernel[grid](\n        x, y,\n        N,\n        BLOCK=BLOCK,\n        num_warps=4,   # good default for 1024\n        num_stages=2,\n    )\n    return y\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['gelu_kernel_impl', 'gelu', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def gelu_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
