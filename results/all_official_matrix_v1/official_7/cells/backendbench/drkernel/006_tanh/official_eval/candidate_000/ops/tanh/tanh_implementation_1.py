import types
_CANDIDATE_CODE = 'import triton\nimport triton.language as tl\n\n# Elementwise tanh kernel.\n# Computes y = tanh(x) using a numerically stable formulation:\n# tanh(x) = 1 - 2 / (exp(2x) + 1)\n@triton.jit\ndef tanh_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n\n    # Load as input dtype, then upcast to float32 for math\n    x = tl.load(x_ptr + offs, mask=mask, other=0)\n    x32 = x.to(tl.float32)\n\n    # Compute tanh via stable formula\n    two_x = 2.0 * x32\n    e = tl.exp(two_x)              # exp(2x)\n    y32 = 1.0 - 2.0 / (e + 1.0)    # 1 - 2 / (exp + 1)\n\n    # Cast back to input dtype and store\n    y = y32.to(x.dtype)\n    tl.store(y_ptr + offs, y, mask=mask)\n\n\ndef tanh_kernel_impl(x):\n    """\n    Triton implementation of tanh.\n    Entry point expected by the benchmark.\n    """\n    if not x.is_cuda:\n        raise RuntimeError("tanh_kernel_impl expects a CUDA tensor")\n\n    # Ensure contiguous for coalesced access\n    was_contig = x.is_contiguous()\n    x_c = x.contiguous()\n\n    N = x_c.numel()\n    y = torch.empty_like(x_c)\n\n    # Choose block and grid\n    BLOCK = 1024\n    grid = (triton.cdiv(N, BLOCK),)\n\n    # Launch kernel\n    tanh_kernel[grid](\n        x_c, y, N,\n        BLOCK=BLOCK,\n        num_warps=4,   # reasonable default for pointwise\n        num_stages=2,\n    )\n\n    # Restore original shape (y is already same shape as x_c)\n    if not was_contig:\n        # If the original was non-contiguous, return with same shape but possibly different strides.\n        # torch.empty_like on non-contig returns contig; we can view as original shape.\n        return y.view(x.shape)\n    return y\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['tanh_kernel_impl', 'tanh', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def tanh_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
