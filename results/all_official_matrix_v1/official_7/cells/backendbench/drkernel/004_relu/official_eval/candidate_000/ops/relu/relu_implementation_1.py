import types
_CANDIDATE_CODE = 'import torch\n\ntry:\n    import triton\n    import triton.language as tl\n    _TRITON_AVAILABLE = True\nexcept Exception:\n    _TRITON_AVAILABLE = False\n\n\n# Simple, memory-bound elementwise ReLU kernel.\n# Autotune over a few reasonable block sizes / num_warps.\n@triton.autotune(\n    configs=[\n        triton.Config({\'BLOCK\': 1024}, num_warps=4, num_stages=2),\n        triton.Config({\'BLOCK\': 2048}, num_warps=4, num_stages=2),\n        triton.Config({\'BLOCK\': 4096}, num_warps=8, num_stages=2),\n    ],\n    key=[\'N\'],\n)\n@triton.jit\ndef _relu_kernel(X, Y, N: tl.int32, BLOCK: tl.int32):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n    x = tl.load(X + offs, mask=mask, other=0)\n    zero = tl.zeros([BLOCK], dtype=x.dtype)\n    y = tl.maximum(x, zero)\n    tl.store(Y + offs, y, mask=mask)\n\n\ndef relu_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n    # Preconditions\n    if not _TRITON_AVAILABLE:\n        raise RuntimeError("Triton is not available")\n    if not x.is_cuda:\n        raise RuntimeError(f"Expected CUDA tensor, got device={x.device}")\n    if not x.is_floating_point():\n        raise RuntimeError(f"Expected floating point tensor, got dtype={x.dtype}")\n\n    # Ensure contiguous for coalesced access\n    x_contig = x.contiguous()\n    y = torch.empty_like(x_contig)\n\n    # Flatten to 1D\n    x_flat = x_contig.view(-1)\n    y_flat = y.view(-1)\n    N = x_flat.numel()\n\n    # Launch kernel\n    grid = (triton.cdiv(N, 1024),)  # grid is recomputed internally by autotune; this is a hint\n    _relu_kernel(x_flat, y_flat, N, grid=grid)\n\n    # Reshape back to original\n    return y.view_as(x)\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['relu_kernel_impl', 'relu', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def relu_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
