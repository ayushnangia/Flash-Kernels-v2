#!/usr/bin/env python3
"""
Enhanced benchmark script that collects both performance and microarchitectural metrics
Part of Omni-Perf-Bench Phase 1 implementation
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from benchmarks.metrics_collector import MetricsCollector, collect_kernel_metrics
from utils.benchmark_utils import benchmark_torch_function, benchmark_cuda_function

# Import kernel implementations
from new_kernels.layer_norm.layer_norm import LayerNormFunction
from new_kernels.softmax.softmax import SoftmaxFunction  
from new_kernels.diagonal_matmul.diagonal_matmul import DiagonalMatmul
from new_kernels.fused_linear_rowsum.fused_linear_rowsum import fused_linear_residual_rowsum_forward

# Create wrapper functions for consistent interface
def layer_norm_cuda(input, normalized_shape, weight, bias, eps):
    return LayerNormFunction.apply(input, weight, bias, eps)

def softmax_cuda(input, dim=-1):
    return SoftmaxFunction.apply(input)

def diagonal_matmul_cuda(input, diagonal):
    return DiagonalMatmul.apply(input, diagonal)

def fused_linear_residual_rowsum_cuda(x, W, residual=None):
    return fused_linear_residual_rowsum_forward(x, W, residual)

# Kernel configurations
KERNEL_CONFIGS = {
    'layer_norm': {
        'cuda_fn': layer_norm_cuda,
        'torch_fn': torch.nn.functional.layer_norm,
        'prepare_inputs': lambda bs, seq_len, hidden: {
            'input': torch.randn(bs, seq_len, hidden, device='cuda'),
            'normalized_shape': (hidden,),
            'weight': torch.randn(hidden, device='cuda'),
            'bias': torch.randn(hidden, device='cuda'),
            'eps': 1e-5
        },
        'test_sizes': [
            (32, 512, 768),    # Small
            (64, 1024, 1024),  # Medium
            (128, 2048, 4096), # Large
        ]
    },
    'softmax': {
        'cuda_fn': softmax_cuda,
        'torch_fn': lambda x: torch.nn.functional.softmax(x, dim=-1),
        'prepare_inputs': lambda bs, seq_len, hidden: {
            'input': torch.randn(bs, seq_len, hidden, device='cuda')
        },
        'test_sizes': [
            (32, 512, 512),
            (64, 1024, 1024),
            (128, 2048, 2048),
        ]
    },
    'diagonal_matmul': {
        'cuda_fn': diagonal_matmul_cuda,
        'torch_fn': lambda x, d: x @ torch.diag(d),
        'prepare_inputs': lambda bs, seq_len, hidden: {
            'input': torch.randn(bs, seq_len, hidden, device='cuda'),
            'diagonal': torch.randn(hidden, device='cuda')
        },
        'test_sizes': [
            (32, 512, 768),
            (64, 1024, 1024),
            (128, 2048, 4096),
        ]
    }
}

def run_enhanced_benchmark(kernel_name: str, collect_metrics: bool = True):
    """Run benchmark with optional metrics collection"""
    
    if kernel_name not in KERNEL_CONFIGS:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    
    config = KERNEL_CONFIGS[kernel_name]
    results = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== Enhanced Benchmark: {kernel_name} ===")
    print(f"Timestamp: {timestamp}")
    print(f"Metrics collection: {'Enabled' if collect_metrics else 'Disabled'}")
    
    for size in config['test_sizes']:
        print(f"\nTesting size: {size}")
        
        # Prepare inputs
        inputs = config['prepare_inputs'](*size)
        
        # Benchmark CUDA implementation
        cuda_time = benchmark_cuda_function(
            config['cuda_fn'],
            inputs,
            num_warmup=50,
            num_runs=100
        )
        
        # Benchmark PyTorch implementation
        if 'torch_fn' in config:
            torch_inputs = {k: v for k, v in inputs.items() 
                          if k in ['input', 'normalized_shape', 'weight', 'bias', 'eps', 'diagonal']}
            torch_time = benchmark_torch_function(
                config['torch_fn'],
                torch_inputs,
                num_warmup=50,
                num_runs=100
            )
            speedup = torch_time / cuda_time
        else:
            torch_time = None
            speedup = None
        
        result = {
            'kernel': kernel_name,
            'batch_size': size[0],
            'seq_len': size[1],
            'hidden_dim': size[2] if len(size) > 2 else None,
            'cuda_time_ms': cuda_time * 1000,
            'torch_time_ms': torch_time * 1000 if torch_time else None,
            'speedup': speedup,
            'timestamp': timestamp
        }
        
        # Collect microarchitectural metrics if enabled
        if collect_metrics:
            print("Collecting microarchitectural metrics...")
            
            # Create simplified kernel wrapper for NCU profiling
            kernel_module = f"new_kernels.{kernel_name}.Functional.{kernel_name}_f"
            
            try:
                metrics = collect_kernel_metrics(
                    kernel_name=f"{kernel_name}_cuda",
                    kernel_path=kernel_module,
                    test_args={
                        k: {'shape': v.shape, 'dtype': str(v.dtype).replace('torch.', ''), 'device': 'cuda'}
                        for k, v in inputs.items() if isinstance(v, torch.Tensor)
                    }
                )
                
                # Add key metrics to results
                result.update({
                    'occupancy_pct': metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', None),
                    'sm_efficiency_pct': metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', None),
                    'memory_throughput_pct': metrics.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', None),
                    'achieved_tflops': metrics.get('achieved_tflops', None),
                    'achieved_bandwidth_gbps': metrics.get('achieved_bandwidth_gbps', None),
                    'arithmetic_intensity': metrics.get('arithmetic_intensity', None),
                    'compute_efficiency_pct': metrics.get('compute_efficiency_pct', None),
                    'memory_efficiency_pct': metrics.get('memory_efficiency_pct', None),
                })
                
            except Exception as e:
                print(f"Warning: Metrics collection failed: {e}")
                print("Continuing with performance metrics only...")
        
        results.append(result)
        
        # Print summary
        print(f"CUDA Time: {result['cuda_time_ms']:.3f} ms")
        if result['torch_time_ms']:
            print(f"PyTorch Time: {result['torch_time_ms']:.3f} ms")
            print(f"Speedup: {result['speedup']:.2f}x")
        
        if collect_metrics and 'occupancy_pct' in result:
            print(f"Occupancy: {result['occupancy_pct']:.1f}%")
            print(f"SM Efficiency: {result['sm_efficiency_pct']:.1f}%")
            print(f"Memory Throughput: {result['memory_throughput_pct']:.1f}%")
            if result.get('achieved_tflops'):
                print(f"Achieved TFLOP/s: {result['achieved_tflops']:.2f}")
            if result.get('arithmetic_intensity'):
                print(f"Arithmetic Intensity: {result['arithmetic_intensity']:.2f} FLOP/byte")
    
    # Save results
    output_dir = Path("benchmarks/data/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / f"{kernel_name}_enhanced_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save as JSON with metadata
    metadata = {
        'kernel': kernel_name,
        'timestamp': timestamp,
        'device': torch.cuda.get_device_name(),
        'cuda_version': torch.version.cuda,
        'metrics_enabled': collect_metrics,
        'results': results
    }
    
    json_path = output_dir / f"{kernel_name}_enhanced_benchmark_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced benchmark with microarchitectural metrics')
    parser.add_argument('--kernel', type=str, choices=list(KERNEL_CONFIGS.keys()),
                       help='Kernel to benchmark')
    parser.add_argument('--all', action='store_true', 
                       help='Benchmark all kernels')
    parser.add_argument('--no-metrics', action='store_true',
                       help='Disable metrics collection (performance only)')
    
    args = parser.parse_args()
    
    if args.all:
        kernels = list(KERNEL_CONFIGS.keys())
    elif args.kernel:
        kernels = [args.kernel]
    else:
        print("Error: Specify --kernel or --all")
        parser.print_help()
        sys.exit(1)
    
    # Check if NCU is available
    if not args.no_metrics:
        try:
            import subprocess
            result = subprocess.run(['which', 'ncu'], capture_output=True)
            if result.returncode != 0:
                print("Warning: NVIDIA Nsight Compute (ncu) not found in PATH")
                print("Metrics collection will be disabled")
                print("To enable metrics, install NVIDIA Nsight Compute")
                args.no_metrics = True
        except Exception:
            args.no_metrics = True
    
    # Run benchmarks
    all_results = {}
    for kernel in kernels:
        results = run_enhanced_benchmark(kernel, collect_metrics=not args.no_metrics)
        all_results[kernel] = results
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()