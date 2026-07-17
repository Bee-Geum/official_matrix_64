import types
_CANDIDATE_CODE = 'import torch\n\ntry:\n    import triton\n    import triton.language as tl\n    TRITON_AVAILABLE = True\nexcept Exception:\n    TRITON_AVAILABLE = False\n\n\n# Simple, fast pointwise kernel: y = x + 1\n@triton.jit\ndef add_one_kernel(x_ptr, y_ptr, N: tl.int32, BLOCK: tl.int32):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n    x = tl.load(x_ptr + offs, mask=mask)\n    y = x + 1.0\n    tl.store(y_ptr + offs, y, mask=mask)\n\n\ndef add_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n    # Validate environment\n    if not TRITON_AVAILABLE:\n        raise RuntimeError("Triton is not available")\n\n    if not x.is_cuda:\n        raise ValueError(f"Expected CUDA tensor, got device={x.device}")\n\n    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):\n        raise TypeError(f"Unsupported dtype {x.dtype}; expected float32/float16/bfloat16")\n\n    # Ensure contiguous for simple 1D indexing\n    if not x.is_contiguous():\n        x = x.contiguous()\n\n    # Flatten to 1D\n    orig_shape = x.shape\n    N = x.numel()\n    x_flat = x.view(-1)\n\n    # Allocate output\n    y_flat = torch.empty_like(x_flat)\n\n    # Launch configuration\n    BLOCK = 1024\n    grid = (triton.cdiv(N, BLOCK),)\n\n    # Launch kernel\n    add_one_kernel[grid](\n        x_flat, y_flat,\n        N,\n        BLOCK=BLOCK,\n        num_warps=4,   # good default for memory-bound pointwise\n        num_stages=2,  # small pipeline depth is fine here\n    )\n\n    # Reshape back\n    return y_flat.view(orig_shape)\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['add_kernel_impl', 'add', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def add_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
