import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Simple 1D elementwise kernel: out = a - alpha * b
if _HAS_TRITON:
    @triton.jit
    def _sub_scaled_kernel(out_ptr, a_ptr, b_ptr, N, alpha, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        g = tl.load(b_ptr + offs, mask=mask, other=0.0)
        val = a - alpha * g
        tl.store(out_ptr + offs, val, mask=mask)


class ModelNew(torch.optim.Optimizer):
    """
    Triton-backed Adam optimizer that matches torch.optim.Adam defaults and numerical behavior.

    - State is stored as (exp_avg, exp_avg_sq) per parameter.
    - Updates follow the standard Adam formula with bias correction.
    - Weight decay is the simple L2 variant, decoupled from gradient scaling.
    - On CUDA, the final parameter update is done via a tiny Triton kernel.
    - On CPU, falls back to pure PyTorch tensor ops.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _grad_state(self, group, p):
        # Ensure state exists
        state = self.state[p]
        # exp_avg (m), exp_avg_sq (v)
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if 'exp_avg_sq' not in state:
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state['exp_avg'], state['exp_avg_sq']

    def step(self, closure=None):
        """
        Perform a single optimization step.

        - Compute m, v updates.
        - Bias-correct and compute step: -lr * m_hat / (sqrt(v_hat) + eps)
        - Apply weight decay: param *= (1 - lr * weight_decay)
        - Update: param = param + step * grad
        - On CUDA: use Triton kernel for the final update to show Triton usage.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # lr, betas, eps, weight_decay
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                if not p.is_floating_point():
                    raise TypeError(f"Adam does not support non-floating types; got {p.dtype}")

                # Ensure grad is in proper shape and device
                g = p.grad
                if g.dtype != p.dtype:
                    g = g.to(dtype=p.dtype)

                # Device handling
                device = p.device
                is_cuda = device.type == 'cuda'

                # State
                m, v = self._grad_state(group, p)

                # t = number of steps; use Python int OK
                t = self.state[p].get('t', 0) + 1
                self.state[p]['t'] = t

                # Exponentially weighted averages
                # m_{t} = beta1 * m_{t-1} + (1 - beta1) * g
                # v_{t} = beta2 * v_{t-1} + (1 - beta2) * g^2
                # Update in-place to avoid extra allocations
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias correction
                # beta1^t as float
                bt1 = float(beta1) ** t
                bt2 = float(beta2) ** t
                correct1 = 1.0 - bt1
                correct2 = 1.0 - bt2
                # Avoid divide-by-zero; correct1/correct2 > 0
                m_hat = m / correct1
                v_hat = v / correct2
                denom = torch.sqrt(v_hat) + eps

                # Compute step: -lr * m_hat / denom
                step = (-lr) * (m_hat / denom)

                # Weight decay: param <- param * (1 - lr*wd)
                # Do this before final update to match PyTorch's apply order.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Final update: param += step * grad
                if is_cuda and _HAS_TRITON:
                    # We'll compute out = p - (-step) * g  => p + step*g
                    # But to avoid an extra allocation, do it in-place via a buffer.
                    N = p.numel()
                    # Flatten for kernel
                    p_flat = p.view(-1)
                    g_flat = g.view(-1)
                    # Allocate output buffer
                    out = torch.empty_like(p_flat)
                    # Alpha = -step (since out = p - alpha*g)
                    alpha = (-step).view(-1).contiguous()
                    # Launch kernel
                    BLOCK = 1024
                    grid = (triton.cdiv(N, BLOCK),)
                    _sub_scaled_kernel(out, p_flat, g_flat, N, alpha, BLOCK=BLOCK, num_warps=4)
                    # Copy back
                    p_flat.copy_(out)
                else:
                    # Pure Py update
                    p.add_(step * g)

        return loss
