import torch
import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    p_ptr, g_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay, step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    g = tl.load(g_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    one = 1.0
    bias_correction1 = one - tl.math.pow(beta1, step)
    bias_correction2 = one - tl.math.pow(beta2, step)

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    denom = tl.math.sqrt(v_hat) + eps

    m_new = beta1 * m + (one - beta1) * g
    v_new = beta2 * v + (one - beta2) * (g * g)

    update = lr * (m_hat / denom) + weight_decay * p
    p_new = p - update

    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)
    tl.store(p_ptr + offsets, p_new, mask=mask)

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step = 0
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        step = self._step

        for group in self.param_groups:
            params = group['params']
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                state = self.state[p]

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                
                m = state['m']
                v = state['v']

                n_elements = p.numel()
                if n_elements == 0:
                    continue

                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                adam_kernel[grid](
                    p, grad, m, v,
                    lr, beta1, beta2, eps, weight_decay, step,
                    n_elements,
                    BLOCK_SIZE=1024,
                )

        return loss

def test_Adam():
    results = {}

    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()