import types
_CANDIDATE_CODE = 'import math\nimport torch\n\ntry:\n    import triton\n    import triton.language as tl\n    TRITON_AVAILABLE = True\nexcept Exception:\n    TRITON_AVAILABLE = False\n\n\n# Simple 1D subtraction kernel: out = x - y\n# Assumes float32, 1D contiguous tensors.\n@triton.jit\ndef sub_kernel(x_ptr, y_ptr, out_ptr, N: tl.int32, BLOCK: tl.int32):\n    pid = tl.program_id(axis=0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < N\n    x = tl.load(x_ptr + offs, mask=mask, eviction_policy=\'evict_last\')\n    y = tl.load(y_ptr + offs, mask=mask, eviction_policy=\'evict_last\')\n    tl.store(out_ptr + offs, x - y, mask=mask)\n\n\nclass ModelNew(torch.nn.Module):\n    def __init__(self, block_size: int = 1024):\n        super().__init__()\n        self.block_size = block_size\n\n    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:\n        # Fallback if Triton/CUDA not available\n        if (not TRITON_AVAILABLE) or (not torch.cuda.is_available()):\n            # Match reference behavior: out = x - y\n            return x - y\n\n        # Validate inputs\n        if x.device.type != \'cuda\' or y.device.type != \'cuda\':\n            raise RuntimeError("ModelNew requires CUDA tensors.")\n        if x.dtype != torch.float32 or y.dtype != torch.float32:\n            raise RuntimeError(f"ModelNew expects float32 tensors, got {x.dtype} and {y.dtype}.")\n        if x.shape != y.shape:\n            raise RuntimeError(f"Shape mismatch: {x.shape} vs {y.shape}.")\n        if not x.is_contiguous() or not y.is_contiguous():\n            # For simplicity, enforce contiguous; could also use .contiguous()\n            raise RuntimeError("ModelNew expects contiguous tensors.")\n\n        N = x.numel()\n        out = torch.empty_like(x)\n\n        # Launch configuration\n        BLOCK = self.block_size\n        grid = (triton.cdiv(N, BLOCK),)\n\n        sub_kernel[grid](\n            x, y, out,\n            N,\n            BLOCK=BLOCK,\n            num_warps=4,   # good default for 1024 block\n            num_stages=2,  # simple pipeline\n        )\n\n        return out\n'
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['sub_kernel_impl', 'sub', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def sub_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
