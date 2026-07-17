import types
_CANDIDATE_CODE = 'import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x, y):\n        return x * y\n\n@triton.jit\ndef mul_kernel(\n    x_ptr,\n    y_ptr,\n    out_ptr,\n    n_elements,\n    BLOCK_SIZE: tl.constexpr,\n):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    y = tl.load(y_ptr + offsets, mask=mask)\n    out = x * y\n    tl.store(out_ptr + offsets, out, mask=mask)\n\ndef mul_kernel_impl(x, y):\n    assert x.is_cuda and y.is_cuda, "Inputs must be on CUDA"\n    out = torch.empty_like(x)\n    n_elements = x.numel()\n    grid = lambda meta: (triton.cdiv(n_elements, meta[\'BLOCK_SIZE\']),)\n    mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)\n    return out\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x, y):\n        return mul_kernel_impl(x, y)\n\nif __name__ == "__main__":\n    model_old = Model()\n    model_new = ModelNew()\n    x = torch.randn(1024, 1024, device=\'cuda\')\n    y = torch.randn(1024, 1024, device=\'cuda\')\n    out_old = model_old(x, y)\n    out_new = model_new(x, y)\n    print(torch.allclose(out_old, out_new, atol=1e-2, rtol=1e-2))\n    import time\n    start = time.time()\n    for _ in range(100):\n        model_old(x, y)\n    torch.cuda.synchronize()\n    print("Old:", time.time() - start)\n    start = time.time()\n    for _ in range(100):\n        model_new(x, y)\n    torch.cuda.synchronize()\n    print("New:", time.time() - start)\n\n__all__ = ["ModelNew"]\n'
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
