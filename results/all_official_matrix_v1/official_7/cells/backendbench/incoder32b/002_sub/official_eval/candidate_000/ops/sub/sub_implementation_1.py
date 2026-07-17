import types
_CANDIDATE_CODE = "import torch\nimport triton\nimport triton.language as tl\n\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return x - x.mean()\n\nclass ModelNew(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return sub_kernel_impl(x)\n\n@triton.jit\ndef sub_kernel(\n    x_ptr,\n    out_ptr,\n    n_elements,\n    BLOCK_SIZE: tl.constexpr,\n):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    mean = tl.sum(x, axis=0) / n_elements\n    out = x - mean\n    tl.store(out_ptr + offsets, out, mask=mask)\n\ndef sub_kernel_impl(x):\n    out = torch.empty_like(x)\n    n_elements = x.numel()\n    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)\n    sub_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)\n    return out\n"
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
