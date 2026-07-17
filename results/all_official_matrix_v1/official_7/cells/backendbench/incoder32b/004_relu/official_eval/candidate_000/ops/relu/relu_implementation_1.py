import types
_CANDIDATE_CODE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\ndef relu_kernel(\n    x_ptr,\n    out_ptr,\n    n_elements,\n    BLOCK_SIZE: tl.constexpr,\n):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    out = tl.where(x > 0, x, 0.0)\n    tl.store(out_ptr + offsets, out, mask=mask)\n\n\ndef relu_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n    out = torch.empty_like(x)\n    n_elements = x.numel()\n    if n_elements == 0:\n        return out\n    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)\n    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)\n    return out\n\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n\n\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return relu_kernel_impl(x)\n\n\ndef relu_kernel_impl(x: torch.Tensor) -> torch.Tensor:\n    out = torch.empty_like(x)\n    n_elements = x.numel()\n    if n_elements == 0:\n        return out\n    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)\n    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)\n    return out\n"
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
