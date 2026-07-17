import math
import triton
import triton.language as tl

# Assuming 'points' is a flat array of doubles: [x0, y0, x1, y1, ...]
@triton.jit
def _closest_pair_block(points_ptr,  # *double
                        numPoints: tl.int32,
                        out_ptr,     # *double
                        BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    # If this block is out of range, early exit (not strictly needed if grid is exact).
    if start >= numPoints:
        return

    # Vector of indices for this block
    idx = start + tl.arange(0, BLOCK)
    valid = idx < numPoints

    # Load x, y; invalid lanes get 0
    x = tl.load(points_ptr + 2 * idx, mask=valid, other=0.0)
    y = tl.load(points_ptr + 2 * idx + 1, mask=valid, other=0.0)

    # Initialize min distance to a large value
    min_dist = tl.full((), 1e20, dtype=tl.float32)

    # Compare all pairs within the block: i in [0, BLOCK), j in [i+1, BLOCK)
    # Note: quadratic work; only viable for small BLOCK / small numPoints
    for i in range(BLOCK):
        xi = x[i]
        yi = y[i]
        # Only consider j > i to avoid duplicate work and self-pair
        for j in range(i + 1, BLOCK):
            xj = x[j]
            yj = y[j]
            # Distance
            dx = xi - xj
            dy = yi - yj
            dist = tl.sqrt(dx * dx + dy * dy)
            # Update min
            min_dist = tl.minimum(min_dist, dist)

    # Store result for this block
    tl.store(out_ptr + pid, min_dist)


class ModelNew:
    def __init__(self, max_block=256):
        self.max_block = max_block

    def __call__(self, points):
        # points: list of Point or numpy array; convert to flat double array
        if isinstance(points, (list, tuple)):
            # Convert to numpy for easy flattening
            import numpy as np
            arr = np.array(points, dtype=np.float64)
        else:
            arr = points

        # Flatten to [x0, y0, x1, y1, ...]
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        flat = arr.reshape(-1)

        n = flat.shape[0] // 2
        if n == 0:
            return float('inf')
        if n == 1:
            return 0.0

        BLOCK = self.max_block
        grid = (triton.cdiv(n, BLOCK),)

        # Output per-block minima
        out = np.empty(grid[0], dtype=np.float32)

        _closest_pair_block[grid](
            flat,  # points_ptr
            n,     # numPoints (in pairs)
            out,   # out_ptr
            BLOCK=BLOCK,
        )

        # Reduce on host to get global minimum
        dist = float(out.min())
        return dist
