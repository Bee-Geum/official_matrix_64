import types
_CANDIDATE_CODE = 'import torch\n\ntry:\n    import triton\n    import triton.language as tl\n    _TRITON_AVAILABLE = True\nexcept Exception:\n    _TRITON_AVAILABLE = False\n\n\n# Simple 1D elementwise kernel: y = x * 0.2  (equivalent to x / 5)\n@triton.jit\ndef _div_by_five_kernel(X, Y, N, BLOCK: tl.constexpr):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n    x = tl.load(X + offs, mask=mask)\n    # multiply by reciprocal to avoid division\n    y = x * 0.2\n    tl.store(Y + offs, y, mask=mask)\n\n\ndef div_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n    """\n    Triton implementation of y = x / 5 for a 1D contiguous tensor x.\n    Always uses GPU if available; falls back to PyTorch only if Triton/CUDA is unavailable.\n    """\n    if not _TRITON_AVAILABLE:\n        raise RuntimeError("Triton is not available")\n\n    if not torch.cuda.is_available():\n        raise RuntimeError("CUDA is not available")\n\n    # Ensure floating dtype\n    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):\n        x = x.float()\n\n    # Contiguous is required for this simple kernel\n    if not x.is_contiguous():\n        x = x.contiguous()\n\n    # Move to GPU if needed\n    x_gpu = x\n    moved_to_gpu = False\n    if not x_gpu.is_cuda:\n        x_gpu = x_gpu.to(\'cuda\')\n        moved_to_gpu = True\n\n    N = x_gpu.numel()\n    y_gpu = torch.empty_like(x_gpu)\n\n    # Choose launch params\n    BLOCK = 1024\n    grid = (triton.cdiv(N, BLOCK),)\n\n    # Launch kernel\n    _div_by_five_kernel[grid](\n        x_gpu, y_gpu, N,\n        BLOCK=BLOCK,\n        num_warps=4,  # good default for memory-bound\n        num_stages=2,\n    )\n\n    # Move back to original device if we moved\n    if moved_to_gpu and x.device != torch.device(\'cuda\'):\n        return y_gpu.to(x.device)\n    return y_gpu\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['div_kernel_impl', 'div', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def div_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
