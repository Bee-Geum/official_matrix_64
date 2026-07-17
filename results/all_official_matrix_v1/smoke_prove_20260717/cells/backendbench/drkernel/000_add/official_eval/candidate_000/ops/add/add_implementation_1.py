import types
_CANDIDATE_CODE = 'import torch\n\ntry:\n    import triton\n    import triton.language as tl\n    _TRITON_AVAILABLE = True\nexcept Exception:\n    _TRITON_AVAILABLE = False\n\n\n# Simple, bandwidth-optimal elementwise: y = x + x  =>  y = 2 * x\n@triton.jit\ndef _double_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n    x = tl.load(x_ptr + offs, mask=mask)\n    y = x + x\n    tl.store(y_ptr + offs, y, mask=mask)\n\n\nclass ModelNew(torch.nn.Module):\n    def __init__(self, block_size: int = 1024, num_warps: int = 4):\n        super().__init__()\n        self.block_size = block_size\n        self.num_warps = num_warps\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        # Validate device\n        if not _TRITON_AVAILABLE:\n            raise RuntimeError("Triton is not available. Install triton to use this backend.")\n        if not x.is_cuda:\n            raise RuntimeError(f"Expected CUDA tensor, got device={x.device}.")\n        if not x.is_contiguous():\n            # Could make contiguous; for now enforce for simplicity\n            raise RuntimeError("Input must be contiguous for this Triton kernel.")\n\n        # Allocate output\n        y = torch.empty_like(x)\n\n        N = x.numel()\n        if N == 0:\n            return y\n\n        grid = (triton.cdiv(N, self.block_size),)\n\n        # Launch\n        _double_kernel[grid](\n            x, y,\n            N,\n            BLOCK=self.block_size,\n            num_warps=self.num_warps,\n        )\n        return y\n'
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
