import torch
import triton
import triton.language as tl

def sigmoid_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized sigmoid implementation using Triton.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        Tensor with sigmoid applied element-wise
    """
    # Flatten the tensor for 1D processing
    original_shape = x.shape
    x_flat = x.view(-1)
    n_elements = x_flat.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x_flat)
    
    # Configure kernel launch parameters
    # Using 1024 threads per block for good occupancy on H100
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _sigmoid_kernel[grid](
        x_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Restore original shape
    return output.view(original_shape)

@triton.jit
def _sigmoid_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for sigmoid computation.
    
    Uses the optimized formula: 1 / (1 + exp(-x))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to avoid out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sigmoid using optimized numerical stability
    # For negative values: exp(x) / (1 + exp(x))
    # For positive values: 1 / (1 + exp(-x))
    # This avoids overflow for large negative values
    
    # Compute exp of negative absolute value for stability
    neg_abs_x = -tl.abs(x)
    exp_neg_abs_x = tl.exp(neg_abs_x)
    
    # Compute denominator: 1 + exp(-|x|)
    denominator = 1.0 + exp_neg_abs_x
    
    # For x >= 0: 1 / (1 + exp(-x)) = 1 / denominator
    # For x < 0: exp(x) / (1 + exp(x)) = exp(-|x|) / denominator
    result = tl.where(x >= 0, 1.0 / denominator, exp_neg_abs_x / denominator)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


class ModelNew(torch.nn.Module):
    """
    Optimized model using custom Triton sigmoid kernel.
    """
    def __init__(self):
        super().__init__()
        # No learnable parameters needed for sigmoid
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using optimized sigmoid kernel.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after applying sigmoid
        """
        return sigmoid_kernel_impl(x)


# Reference model for comparison
class Model(torch.nn.Module):
    """
    Reference model using PyTorch's built-in sigmoid.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


if __name__ == "__main__":
    # Test the implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    model_ref = Model().to(device)
    model_new = ModelNew().to(device)
    
    # Test with random input
    test_input = torch.randn(10000, device=device)
    
    # Get outputs
    with torch.no_grad():
        output_ref = model_ref(test_input)
        output_new = model_new(test_input)
    
    # Check numerical accuracy
    max_diff = torch.max(torch.abs(output_ref - output_new)).item()
    print(f"Maximum difference: {max_diff:.6f}")
    
    # Check if within tolerance
    tolerance = 1e-2
    if max_diff < tolerance:
        print("✓ Outputs match within tolerance")
    else:
        print("✗ Outputs differ beyond tolerance")
    
    # Benchmark performance
    import time
    
    # Warm up
    for _ in range(10):
        _ = model_new(test_input)
    
    # Benchmark new model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = model_new(test_input)
    torch.cuda.synchronize()
    new_time = time.time() - start_time
    
    # Benchmark reference model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = model_ref(test_input)
    torch.cuda.synchronize()
    ref_time = time.time() - start_time
    
    print(f"\nPerformance comparison:")
    print(f"Reference model: {ref_time:.4f} seconds")
    print(f"Optimized model: {new_time:.4f} seconds")
    print(f"Speedup: {ref_time/new_time:.2f}x")
