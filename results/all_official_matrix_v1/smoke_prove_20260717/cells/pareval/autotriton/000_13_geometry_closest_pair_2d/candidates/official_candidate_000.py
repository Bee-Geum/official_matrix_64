import torch
import triton
import triton.language as tl

@triton.jit
def local_min_kernel(points, numPoints, local_mins, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= numPoints:
        return
    offs_i = pid * 2
    x_i = tl.load(points + offs_i)
    y_i = tl.load(points + offs_i + 1)
    local_min = 1e20
    for j in range(pid + 1, numPoints):
        offs_j = j * 2
        x_j = tl.load(points + offs_j)
        y_j = tl.load(points + offs_j + 1)
        dx = x_j - x_i
        dy = y_j - y_i
        dist = tl.sqrt(dx * dx + dy * dy)
        if dist < local_min:
            local_min = dist
    tl.store(local_mins + pid, local_min)

@triton.jit
def reduce_min_kernel(local_mins, numPoints, distance_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numPoints
    data = tl.load(local_mins + offsets, mask=mask, other=float('inf'))
    block_min = tl.min(data, axis=0)
    current_value = tl.load(distance_ptr)
    while current_value > block_min:
        if tl.atomic_cas(distance_ptr, current_value, block_min):
            break
        current_value = tl.load(distance_ptr)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points):
        n = points.shape[0]
        if n < 2:
            return torch.tensor(float('inf'), device=points.device)
        points_f32 = points.to(torch.float32)
        distance = torch.tensor([float('inf')], device=points.device, dtype=torch.float32)
        local_mins = torch.empty(n, device=points.device, dtype=torch.float32)
        grid = lambda meta: (n,)
        local_min_kernel[grid](points_f32, n, local_mins, BLOCK_SIZE=1024)
        num_blocks = (n + 1023) // 1024
        reduce_min_kernel[(num_blocks,)](local_mins, n, distance, BLOCK_SIZE=1024)
        return distance[0]
