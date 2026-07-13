import types
_CANDIDATE_CODE = 'import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef sub_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    y = tl.load(y_ptr + offsets, mask=mask)\n    z = x - y\n    tl.store(z_ptr + offsets, z, mask=mask)\n\nclass DirectoryBackend:\n    def __init__(self):\n        pass\n\n    def sub_kernel_impl(self, x, y):\n        assert x.shape == y.shape, "Input tensors must have the same shape"\n        n_elements = x.numel()\n        BLOCK_SIZE = 1024\n        grid = lambda meta: (triton.cdiv(n_elements, meta[\'BLOCK_SIZE\']),)\n        z = torch.empty_like(x)\n        sub_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)\n        return z\n'
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
