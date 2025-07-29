"""
Benchmark utilities for Omni-Perf-Bench
"""

import torch
import triton
from typing import Dict, Any, Callable

def benchmark_cuda_function(func: Callable, inputs: Dict[str, Any], 
                           num_warmup: int = 50, num_runs: int = 100) -> float:
    """Benchmark a CUDA function using triton's do_bench"""
    
    # Prepare function call
    def kernel_fn():
        # Filter inputs to only include tensors and valid parameters
        filtered_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                filtered_inputs[k] = v
            elif k in ['normalized_shape', 'eps', 'dim']:
                filtered_inputs[k] = v
        
        return func(**filtered_inputs)
    
    # Use triton's benchmarking utility
    ms = triton.testing.do_bench(kernel_fn, warmup=num_warmup, rep=num_runs)
    
    return ms / 1000.0  # Convert to seconds

def benchmark_torch_function(func: Callable, inputs: Dict[str, Any],
                           num_warmup: int = 50, num_runs: int = 100) -> float:
    """Benchmark a PyTorch function"""
    
    def kernel_fn():
        # Handle different function signatures
        if 'normalized_shape' in inputs:  # layer_norm
            return func(inputs['input'], inputs['normalized_shape'], 
                       inputs.get('weight'), inputs.get('bias'), inputs.get('eps', 1e-5))
        elif 'diagonal' in inputs:  # diagonal_matmul
            return func(inputs['input'], inputs['diagonal'])
        else:  # softmax and others
            return func(inputs['input'])
    
    ms = triton.testing.do_bench(kernel_fn, warmup=num_warmup, rep=num_runs)
    
    return ms / 1000.0  # Convert to seconds