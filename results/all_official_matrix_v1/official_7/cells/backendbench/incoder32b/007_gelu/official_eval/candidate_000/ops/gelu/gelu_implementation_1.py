import types
_CANDIDATE_CODE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return torch.nn.functional.gelu(x)\n\n@triton.jit\ndef gelu_kernel(\n    x_ptr,\n    output_ptr,\n    n_elements,\n    BLOCK_SIZE: tl.constexpr,\n):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n    sqrt_2_over_pi = 0.7978845608028654\n    coeff = 0.044715\n    x_cubed = x * x * x\n    inner = sqrt_2_over_pi * (x + coeff * x_cubed)\n    tanh_inner = tl.tanh(inner)\n    output = 0.5 * x * (1 + tanh_inner)\n    tl.store(output_ptr + offsets, output, mask=mask)\n\ndef gelu_kernel_impl(x):\n    output = torch.empty_like(x)\n    n_elements = x.numel()\n    if n_elements == 0:\n        return output\n    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)\n    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)\n    return output\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return gelu_kernel_impl(x)\n"
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
