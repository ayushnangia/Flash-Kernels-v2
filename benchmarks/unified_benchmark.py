#!/usr/bin/env python3
"""
Unified benchmarking script that integrates performance and metrics collection
Provides a single entry point for all Omni-Perf-Bench benchmarking needs
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import torch
import triton

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.enhanced_metrics_collector import EnhancedMetricsCollector, MetricsResult
from utils.metrics_utils import detect_gpu_architecture, format_metrics_report

# Import kernel implementations with proper error handling
try:
    from new_kernels.layer_norm.layer_norm import LayerNormFunction
    from new_kernels.softmax.softmax import SoftmaxFunction  
    from new_kernels.diagonal_matmul.diagonal_matmul import DiagonalMatMulFunction
    from new_kernels.fused_linear_rowsum.fused_linear_rowsum import fused_linear_residual_rowsum_forward
    KERNELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some kernels not available: {e}")
    KERNELS_AVAILABLE = False

class UnifiedBenchmark:
    """Unified benchmark runner with integrated metrics collection"""
    
    def __init__(self, output_dir: str = "benchmarks/data/unified",
                 enable_metrics: bool = True,
                 enable_validation: bool = True):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_metrics = enable_metrics
        self.enable_validation = enable_validation
        
        # Initialize metrics collector if enabled
        if self.enable_metrics:
            self.metrics_collector = EnhancedMetricsCollector(
                output_dir=str(self.output_dir / "metrics")
            )
        else:
            self.metrics_collector = None
        
        # Detect GPU
        self.device_caps = detect_gpu_architecture()
        self.device_name = torch.cuda.get_device_name()
        
        # Configure kernels
        self.kernel_configs = self._setup_kernel_configs()
        
        # Results storage
        self.results = []
        
    def _setup_kernel_configs(self) -> Dict[str, Dict[str, Any]]:
        """Setup kernel configurations"""
        
        configs = {}
        
        if KERNELS_AVAILABLE:
            # Layer norm configuration
            configs['layer_norm'] = {
                'cuda_fn': lambda inp, w, b, eps: LayerNormFunction.apply(inp, w, b, eps),
                'torch_fn': torch.nn.functional.layer_norm,
                'prepare_inputs': self._prepare_layer_norm_inputs,
                'validate_fn': self._validate_layer_norm,
                'test_configs': [
                    {'batch_size': 32, 'seq_len': 512, 'hidden_dim': 768, 'dtype': torch.float32},
                    {'batch_size': 64, 'seq_len': 1024, 'hidden_dim': 1024, 'dtype': torch.float32},
                    {'batch_size': 128, 'seq_len': 2048, 'hidden_dim': 4096, 'dtype': torch.float32},
                ]
            }
            
            # Softmax configuration
            configs['softmax'] = {
                'cuda_fn': lambda inp: SoftmaxFunction.apply(inp),
                'torch_fn': lambda inp: torch.nn.functional.softmax(inp, dim=-1),
                'prepare_inputs': self._prepare_softmax_inputs,
                'validate_fn': self._validate_softmax,
                'test_configs': [
                    {'batch_size': 32, 'seq_len': 512, 'hidden_dim': 512, 'dtype': torch.float32},
                    {'batch_size': 64, 'seq_len': 1024, 'hidden_dim': 1024, 'dtype': torch.float32},
                    {'batch_size': 128, 'seq_len': 2048, 'hidden_dim': 2048, 'dtype': torch.float32},
                ]
            }
            
            # Diagonal matmul configuration
            configs['diagonal_matmul'] = {
                'cuda_fn': lambda inp, diag: DiagonalMatMulFunction.apply(inp, diag),
                'torch_fn': lambda inp, diag: inp @ torch.diag(diag),
                'prepare_inputs': self._prepare_diagonal_matmul_inputs,
                'validate_fn': self._validate_diagonal_matmul,
                'test_configs': [
                    {'batch_size': 32, 'seq_len': 512, 'hidden_dim': 768, 'dtype': torch.float32},
                    {'batch_size': 64, 'seq_len': 1024, 'hidden_dim': 1024, 'dtype': torch.float32},
                    {'batch_size': 128, 'seq_len': 2048, 'hidden_dim': 4096, 'dtype': torch.float32},
                ]
            }
            
            # Add BF16 configs if supported
            if self.device_caps.has_bf16:
                for kernel_config in configs.values():
                    bf16_configs = []
                    for config in kernel_config['test_configs']:
                        bf16_config = config.copy()
                        bf16_config['dtype'] = torch.bfloat16
                        bf16_configs.append(bf16_config)
                    kernel_config['test_configs'].extend(bf16_configs)
        
        return configs
    
    def _prepare_layer_norm_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for layer norm kernel"""
        bs = config['batch_size']
        seq_len = config['seq_len'] 
        hidden = config['hidden_dim']
        dtype = config['dtype']
        
        return {
            'input': torch.randn(bs, seq_len, hidden, dtype=dtype, device='cuda'),
            'weight': torch.randn(hidden, dtype=dtype, device='cuda'),
            'bias': torch.randn(hidden, dtype=dtype, device='cuda'),
            'eps': 1e-5
        }
    
    def _prepare_softmax_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for softmax kernel"""
        bs = config['batch_size']
        seq_len = config['seq_len']
        hidden = config['hidden_dim']
        dtype = config['dtype']
        
        return {
            'input': torch.randn(bs, seq_len, hidden, dtype=dtype, device='cuda')
        }
    
    def _prepare_diagonal_matmul_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for diagonal matmul kernel"""
        bs = config['batch_size']
        seq_len = config['seq_len']
        hidden = config['hidden_dim']
        dtype = config['dtype']
        
        return {
            'input': torch.randn(bs, seq_len, hidden, dtype=dtype, device='cuda'),
            'diagonal': torch.randn(hidden, dtype=dtype, device='cuda')
        }
    
    def _validate_layer_norm(self, cuda_output: torch.Tensor, 
                           torch_output: torch.Tensor,
                           config: Dict[str, Any]) -> bool:
        """Validate layer norm output"""
        
        if not self.enable_validation:
            return True
            
        # Use appropriate tolerance based on dtype
        if config['dtype'] == torch.bfloat16:
            rtol, atol = 1e-2, 1e-3
        else:
            rtol, atol = 1e-4, 1e-5
        
        return torch.allclose(cuda_output, torch_output, rtol=rtol, atol=atol)
    
    def _validate_softmax(self, cuda_output: torch.Tensor,
                        torch_output: torch.Tensor,
                        config: Dict[str, Any]) -> bool:
        """Validate softmax output"""
        
        if not self.enable_validation:
            return True
            
        if config['dtype'] == torch.bfloat16:
            rtol, atol = 1e-2, 1e-3
        else:
            rtol, atol = 1e-4, 1e-5
        
        return torch.allclose(cuda_output, torch_output, rtol=rtol, atol=atol)
    
    def _validate_diagonal_matmul(self, cuda_output: torch.Tensor,
                                torch_output: torch.Tensor,
                                config: Dict[str, Any]) -> bool:
        """Validate diagonal matmul output"""
        
        if not self.enable_validation:
            return True
            
        if config['dtype'] == torch.bfloat16:
            rtol, atol = 1e-2, 1e-3
        else:
            rtol, atol = 1e-4, 1e-5
        
        return torch.allclose(cuda_output, torch_output, rtol=rtol, atol=atol)
    
    def benchmark_kernel(self, kernel_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single kernel configuration"""
        
        if kernel_name not in self.kernel_configs:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        
        kernel_config = self.kernel_configs[kernel_name]
        
        # Prepare inputs
        inputs = kernel_config['prepare_inputs'](config)
        
        # Prepare kernel functions
        cuda_fn = kernel_config['cuda_fn']
        torch_fn = kernel_config['torch_fn']
        
        # Create partial functions for benchmarking
        def run_cuda():
            # Extract only tensor inputs for CUDA kernel
            cuda_inputs = {k: v for k, v in inputs.items() 
                          if k in ['input', 'weight', 'bias', 'eps', 'diagonal']}
            if kernel_name == 'layer_norm':
                return cuda_fn(inputs['input'], inputs['weight'], 
                             inputs['bias'], inputs['eps'])
            elif kernel_name == 'softmax':
                return cuda_fn(inputs['input'])
            else:
                return cuda_fn(**cuda_inputs)
        
        def run_torch():
            if kernel_name == 'layer_norm':
                return torch_fn(inputs['input'], (config['hidden_dim'],),
                              inputs['weight'], inputs['bias'], inputs['eps'])
            elif kernel_name == 'softmax':
                return torch_fn(inputs['input'])
            else:
                return torch_fn(**inputs)
        
        # Measure performance
        logger.info(f"Benchmarking {kernel_name} with config: {config}")
        
        # Use Triton's benchmarking utility for accurate timing
        cuda_time_ms = triton.testing.do_bench(run_cuda, warmup=50, rep=100)
        torch_time_ms = triton.testing.do_bench(run_torch, warmup=50, rep=100)
        
        speedup = torch_time_ms / cuda_time_ms
        
        # Validate outputs if enabled
        validation_passed = True
        if self.enable_validation:
            with torch.no_grad():
                cuda_output = run_cuda()
                torch_output = run_torch()
                validation_passed = kernel_config['validate_fn'](
                    cuda_output, torch_output, config
                )
                if not validation_passed:
                    logger.warning(f"Validation failed for {kernel_name} with config {config}")
        
        # Collect metrics if enabled
        metrics_result = None
        if self.enable_metrics and self.metrics_collector:
            try:
                metrics_result = self.metrics_collector.collect_metrics(
                    cuda_fn, inputs, kernel_name,
                    num_warmup=10, num_runs=50
                )
                metrics_result.speedup = speedup
            except Exception as e:
                logger.warning(f"Metrics collection failed: {e}")
        
        # Compile results
        result = {
            'kernel': kernel_name,
            'timestamp': datetime.now().isoformat(),
            'device': self.device_name,
            'config': config,
            'cuda_time_ms': cuda_time_ms,
            'torch_time_ms': torch_time_ms,
            'speedup': speedup,
            'validation_passed': validation_passed,
        }
        
        # Add metrics if available
        if metrics_result:
            result['metrics'] = metrics_result.to_dict()
        
        return result
    
    def run_benchmark_suite(self, kernels: Optional[List[str]] = None,
                          configs: Optional[List[Dict[str, Any]]] = None):
        """Run full benchmark suite"""
        
        if kernels is None:
            kernels = list(self.kernel_configs.keys())
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print(f"Unified Benchmark Suite - {timestamp}")
        print(f"Device: {self.device_name}")
        print(f"Architecture: {self.device_caps.architecture.value}")
        print(f"Metrics Collection: {'Enabled' if self.enable_metrics else 'Disabled'}")
        print(f"Validation: {'Enabled' if self.enable_validation else 'Disabled'}")
        print(f"{'='*60}\n")
        
        all_results = []
        metrics_results = []
        
        for kernel_name in kernels:
            if kernel_name not in self.kernel_configs:
                logger.warning(f"Skipping unknown kernel: {kernel_name}")
                continue
            
            kernel_config = self.kernel_configs[kernel_name]
            test_configs = configs if configs else kernel_config['test_configs']
            
            print(f"\nBenchmarking {kernel_name}...")
            print("-" * 40)
            
            for test_config in test_configs:
                try:
                    result = self.benchmark_kernel(kernel_name, test_config)
                    all_results.append(result)
                    
                    if 'metrics' in result:
                        metrics_results.append(MetricsResult(**result['metrics']))
                    
                    # Print summary
                    print(f"Config: {test_config}")
                    print(f"  CUDA: {result['cuda_time_ms']:.3f} ms")
                    print(f"  PyTorch: {result['torch_time_ms']:.3f} ms") 
                    print(f"  Speedup: {result['speedup']:.2f}x")
                    
                    if 'metrics' in result:
                        m = result['metrics']
                        if m.get('occupancy_pct'):
                            print(f"  Occupancy: {m['occupancy_pct']:.1f}%")
                        if m.get('achieved_tflops'):
                            print(f"  TFLOP/s: {m['achieved_tflops']:.2f}")
                    
                    if not result['validation_passed']:
                        print("  ⚠️  Validation FAILED")
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {kernel_name} with {test_config}: {e}")
        
        # Save results
        self._save_results(all_results, metrics_results, timestamp)
        
        print(f"\n{'='*60}")
        print(f"Benchmark Complete - Results saved to {self.output_dir}")
        print(f"{'='*60}\n")
        
        return all_results
    
    def _save_results(self, results: List[Dict[str, Any]], 
                     metrics_results: List[MetricsResult],
                     timestamp: str):
        """Save benchmark results in multiple formats"""
        
        # Save raw results as JSON
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            row = {
                'kernel': result['kernel'],
                'batch_size': result['config']['batch_size'],
                'seq_len': result['config']['seq_len'],
                'hidden_dim': result['config']['hidden_dim'],
                'dtype': str(result['config']['dtype']),
                'cuda_time_ms': result['cuda_time_ms'],
                'torch_time_ms': result['torch_time_ms'],
                'speedup': result['speedup'],
                'validation': result['validation_passed'],
            }
            
            # Add key metrics if available
            if 'metrics' in result:
                m = result['metrics']
                row.update({
                    'occupancy_pct': m.get('occupancy_pct'),
                    'sm_efficiency_pct': m.get('sm_efficiency_pct'),
                    'achieved_tflops': m.get('achieved_tflops'),
                    'arithmetic_intensity': m.get('arithmetic_intensity'),
                })
            
            summary_data.append(row)
        
        # Save summary as CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / f"benchmark_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # Generate summary statistics
            self._generate_summary_report(df, timestamp)
        
        # Save detailed metrics if available
        if metrics_results and self.metrics_collector:
            self.metrics_collector.save_results(
                metrics_results, 
                f"detailed_metrics_{timestamp}"
            )
    
    def _generate_summary_report(self, df: pd.DataFrame, timestamp: str):
        """Generate a summary report of benchmark results"""
        
        report_path = self.output_dir / f"summary_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Unified Benchmark Summary Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Device: {self.device_name}\n")
            f.write(f"\n{'='*60}\n\n")
            
            # Overall statistics
            f.write("Overall Performance:\n")
            f.write(f"  Average Speedup: {df['speedup'].mean():.2f}x\n")
            f.write(f"  Min Speedup: {df['speedup'].min():.2f}x\n")
            f.write(f"  Max Speedup: {df['speedup'].max():.2f}x\n")
            
            # Validation summary
            if 'validation' in df.columns:
                validation_rate = df['validation'].sum() / len(df) * 100
                f.write(f"  Validation Pass Rate: {validation_rate:.1f}%\n")
            
            # Per-kernel summary
            f.write(f"\n{'='*60}\n")
            f.write("Per-Kernel Summary:\n\n")
            
            for kernel in df['kernel'].unique():
                kernel_df = df[df['kernel'] == kernel]
                f.write(f"{kernel}:\n")
                f.write(f"  Configurations tested: {len(kernel_df)}\n")
                f.write(f"  Average speedup: {kernel_df['speedup'].mean():.2f}x\n")
                
                if 'achieved_tflops' in kernel_df.columns:
                    tflops = kernel_df['achieved_tflops'].dropna()
                    if len(tflops) > 0:
                        f.write(f"  Average TFLOP/s: {tflops.mean():.2f}\n")
                
                if 'occupancy_pct' in kernel_df.columns:
                    occ = kernel_df['occupancy_pct'].dropna()
                    if len(occ) > 0:
                        f.write(f"  Average occupancy: {occ.mean():.1f}%\n")
                
                f.write("\n")
            
            # Data type comparison if available
            if df['dtype'].nunique() > 1:
                f.write(f"{'='*60}\n")
                f.write("Data Type Comparison:\n\n")
                
                for dtype in df['dtype'].unique():
                    dtype_df = df[df['dtype'] == dtype]
                    f.write(f"{dtype}:\n")
                    f.write(f"  Average speedup: {dtype_df['speedup'].mean():.2f}x\n")
                    f.write(f"  Configurations: {len(dtype_df)}\n\n")

def main():
    """Main entry point for unified benchmarking"""
    
    parser = argparse.ArgumentParser(
        description='Unified benchmark runner for Omni-Perf-Bench'
    )
    
    parser.add_argument('--kernels', nargs='+', 
                       help='Kernels to benchmark (default: all)')
    parser.add_argument('--no-metrics', action='store_true',
                       help='Disable metrics collection')
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable output validation')
    parser.add_argument('--output-dir', type=str, 
                       default='benchmarks/data/unified',
                       help='Output directory for results')
    parser.add_argument('--config-file', type=str,
                       help='JSON file with custom test configurations')
    
    args = parser.parse_args()
    
    # Load custom configs if provided
    custom_configs = None
    if args.config_file:
        with open(args.config_file, 'r') as f:
            custom_configs = json.load(f)
    
    # Create benchmark runner
    benchmark = UnifiedBenchmark(
        output_dir=args.output_dir,
        enable_metrics=not args.no_metrics,
        enable_validation=not args.no_validation
    )
    
    # Run benchmarks
    try:
        results = benchmark.run_benchmark_suite(
            kernels=args.kernels,
            configs=custom_configs
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()