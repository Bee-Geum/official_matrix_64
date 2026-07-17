import types
_CANDIDATE_CODE = 'import torch\nimport triton\nimport triton.language as tl\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return torch.tanh(x)\n\n@triton.jit\ndef tanh_kernel(\n    x_ptr,\n    out_ptr,\n    n_elements,\n    BLOCK_SIZE: tl.constexpr,\n):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    out = tl.tanh(x)\n    tl.store(out_ptr + offsets, out, mask=mask)\n\ndef tanh_kernel_impl(x):\n    out = torch.empty_like(x)\n    n_elements = x.numel()\n    grid = lambda meta: (triton.cdiv(n_elements, meta[\'BLOCK_SIZE\']),)\n    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)\n    return out\n\nclass ModelNew(torch.nn.Module):\n    def forward(self, x):\n        return tanh_kernel_impl(x)\n\nif __name__ == "__main__":\n    model_ref = Model()\n    model_new = ModelNew()\n    x = torch.randn(1024, 1024, device=\'cuda\')\n    with torch.no_grad():\n        y_ref = model_ref(x)\n        y_new = model_new(x)\n    print(torch.allclose(y_ref, y_new, atol=1e-2, rtol=1e-2))\n    import time\n    torch.cuda.synchronize()\n    start = time.time()\n    for _ in range(100):\n        _ = model_ref(x)\n    torch.cuda.synchronize()\n    t_ref = time.time() - start\n    start = time.time()\n    for _ in range(100):\n        _ = model_new(x)\n    torch.cuda.synchronize()\n    t_new = time.time() - start\n    print(f"Reference: {t_ref:.4f}s, New: {t_new:.4f}s, Speedup: {t_ref/t_new:.2f}x")\n'
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
