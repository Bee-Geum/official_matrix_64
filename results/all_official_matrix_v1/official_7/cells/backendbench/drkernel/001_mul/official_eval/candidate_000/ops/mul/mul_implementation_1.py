import types
_CANDIDATE_CODE = 'import torch\n\ntry:\n    import triton\n    import triton.language as tl\n    TRITON_AVAILABLE = True\nexcept Exception:\n    TRITON_AVAILABLE = False\n\n\n# Simple, fast elementwise square kernel for float32\n@triton.jit\ndef square_kernel(X, Y, N, BLOCK: tl.constexpr):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n    x = tl.load(X + offs, mask=mask, eviction_policy="evict_last")\n    y = x * x\n    tl.store(Y + offs, y, mask=mask)\n\n\nclass MulTriton:\n    @staticmethod\n    def mul_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n        # Assumes: x is 1D, float32\n        if not TRITON_AVAILABLE:\n            raise RuntimeError("Triton is not available")\n\n        if x.dtype != torch.float32:\n            raise TypeError(f"Expected float32, got {x.dtype}")\n        if x.dim() != 1:\n            raise ValueError(f"Expected 1D tensor, got shape {tuple(x.shape)}")\n\n        N = x.shape[0]\n        y = torch.empty_like(x)\n\n        # Choose a block size; 1024 is a good default for pointwise fp32\n        BLOCK = 1024\n        grid = (triton.cdiv(N, BLOCK),)\n\n        square_kernel[grid](\n            x, y, N,\n            BLOCK=BLOCK,\n            num_warps=4,   # reasonable default for 1024 block\n            num_stages=2,  # small pipeline depth is fine here\n        )\n        return y\n\n\n# The benchmark harness will look for \'mul_kernel_impl\'\nmul_kernel_impl = MulTriton.mul_kernel_impl\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['mul_kernel_impl', 'mul', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def mul_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
