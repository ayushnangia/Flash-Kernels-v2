"""
Enhanced metrics collection for Omni-Perf-Bench with improved robustness
Includes NCU integration, CUPTI fallback, and architecture-specific optimizations
"""

import subprocess
import json
import csv
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch

from utils.metrics_utils import (
    detect_gpu_architecture, 
    get_architecture_specific_metrics,
    check_ncu_availability,
    validate_metrics,
    calculate_roofline_ceilings,
    format_metrics_report,
    GPUArchitecture
)

logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:
    """Container for metrics collection results"""
    kernel_name: str
    timestamp: str
    device_name: str
    architecture: str
    
    # Performance metrics
    execution_time_ms: float
    speedup: Optional[float] = None
    
    # Microarchitectural metrics
    occupancy_pct: Optional[float] = None
    sm_efficiency_pct: Optional[float] = None
    memory_throughput_pct: Optional[float] = None
    l2_hit_rate_pct: Optional[float] = None
    
    # Computed metrics
    achieved_tflops: Optional[float] = None
    achieved_bandwidth_gbps: Optional[float] = None
    arithmetic_intensity: Optional[float] = None
    
    # Efficiency metrics
    compute_efficiency_pct: Optional[float] = None
    memory_efficiency_pct: Optional[float] = None
    
    # Stall analysis
    stall_reasons: Optional[Dict[str, float]] = None
    
    # Raw metrics
    raw_metrics: Optional[Dict[str, float]] = None
    
    # Validation
    validation_passed: bool = True
    validation_warnings: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EnhancedMetricsCollector:
    """Robust metrics collection with multiple fallback strategies"""
    
    def __init__(self, output_dir: str = "benchmarks/data/metrics", 
                 cache_dir: str = ".metrics_cache"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Detect GPU capabilities
        self.device_caps = detect_gpu_architecture()
        self.device_name = torch.cuda.get_device_name()
        
        # Check NCU availability
        self.ncu_available, self.ncu_path = check_ncu_availability()
        if self.ncu_available:
            logger.info(f"NCU found at: {self.ncu_path}")
        else:
            logger.warning("NCU not available, will use fallback methods")
        
        # Get architecture-specific metrics
        self.ncu_metrics = get_architecture_specific_metrics(self.device_caps.architecture)
        
        # Calculate theoretical peaks
        self.roofline_ceilings = self._calculate_device_ceilings()
        
        # Cache for repeated measurements
        self.metrics_cache = {}
        
    def _calculate_device_ceilings(self) -> Dict[str, float]:
        """Calculate device-specific performance ceilings"""
        props = torch.cuda.get_device_properties(0)
        clock_rate_ghz = props.clock_rate / 1e6
        return calculate_roofline_ceilings(self.device_caps, clock_rate_ghz)
    
    def collect_metrics(self, kernel_fn, kernel_args: Dict[str, Any], 
                       kernel_name: str, num_warmup: int = 10,
                       num_runs: int = 100) -> MetricsResult:
        """Collect comprehensive metrics for a kernel"""
        
        # Check cache first
        cache_key = self._generate_cache_key(kernel_name, kernel_args)
        if cache_key in self.metrics_cache:
            logger.info(f"Using cached metrics for {kernel_name}")
            return self.metrics_cache[cache_key]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # First, measure execution time accurately
        exec_time_ms = self._measure_execution_time(kernel_fn, kernel_args, 
                                                   num_warmup, num_runs)
        
        # Initialize result
        result = MetricsResult(
            kernel_name=kernel_name,
            timestamp=timestamp,
            device_name=self.device_name,
            architecture=self.device_caps.architecture.value,
            execution_time_ms=exec_time_ms
        )
        
        # Try to collect detailed metrics
        if self.ncu_available:
            try:
                raw_metrics = self._collect_ncu_metrics(kernel_fn, kernel_args, kernel_name)
                result.raw_metrics = raw_metrics
                self._populate_result_from_raw_metrics(result, raw_metrics)
            except Exception as e:
                logger.warning(f"NCU collection failed: {e}, using fallback")
                self._collect_fallback_metrics(result, kernel_fn, kernel_args)
        else:
            self._collect_fallback_metrics(result, kernel_fn, kernel_args)
        
        # Validate metrics
        if result.raw_metrics:
            validation = validate_metrics(result.raw_metrics)
            result.validation_passed = validation["valid"]
            result.validation_warnings = validation["warnings"]
        
        # Cache result
        self.metrics_cache[cache_key] = result
        
        return result
    
    def _measure_execution_time(self, kernel_fn, kernel_args: Dict[str, Any],
                               num_warmup: int, num_runs: int) -> float:
        """Accurate kernel execution time measurement"""
        
        # Prepare kernel call
        def run_kernel():
            return kernel_fn(**kernel_args)
        
        # Warmup
        for _ in range(num_warmup):
            run_kernel()
            torch.cuda.synchronize()
        
        # Timed runs
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        
        for i in range(num_runs):
            start_events[i].record()
            run_kernel()
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        # Calculate median time
        times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
        return np.median(times)
    
    def _collect_ncu_metrics(self, kernel_fn, kernel_args: Dict[str, Any], 
                           kernel_name: str) -> Dict[str, float]:
        """Collect metrics using NVIDIA Nsight Compute"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_script:
            # Generate standalone script for NCU profiling
            script_content = self._generate_ncu_script(kernel_fn, kernel_args, kernel_name)
            tmp_script.write(script_content)
            tmp_script.flush()
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_output:
                try:
                    # Build NCU command with retry logic
                    for attempt in range(3):
                        ncu_cmd = [
                            self.ncu_path,
                            "--target-processes", "all",
                            "--kernel-name-base", "function",
                            "--launch-skip", "2",  # Skip warmup runs
                            "--launch-count", "1",
                            "--metrics", ",".join(self.ncu_metrics),
                            "--csv",
                            "--export", tmp_output.name,
                            "python", tmp_script.name
                        ]
                        
                        result = subprocess.run(
                            ncu_cmd, 
                            capture_output=True, 
                            text=True,
                            timeout=60
                        )
                        
                        if result.returncode == 0:
                            break
                        elif attempt < 2:
                            logger.warning(f"NCU attempt {attempt + 1} failed, retrying...")
                            time.sleep(1)
                        else:
                            raise RuntimeError(f"NCU failed after 3 attempts: {result.stderr}")
                    
                    # Parse CSV output
                    return self._parse_ncu_output(tmp_output.name)
                    
                finally:
                    # Cleanup
                    os.unlink(tmp_script.name)
                    if os.path.exists(tmp_output.name):
                        os.unlink(tmp_output.name)
    
    def _generate_ncu_script(self, kernel_fn, kernel_args: Dict[str, Any], 
                           kernel_name: str) -> str:
        """Generate standalone Python script for NCU profiling"""
        
        # This is a simplified version - in practice, we'd need to handle
        # more complex kernel signatures and imports
        return f"""
import torch
import sys
sys.path.append('.')

# Initialize inputs
{self._generate_input_initialization(kernel_args)}

# Import kernel
from {kernel_fn.__module__} import {kernel_fn.__name__}

# Warmup runs
for _ in range(3):
    {kernel_fn.__name__}({', '.join(kernel_args.keys())})
    torch.cuda.synchronize()

# Profile run
torch.cuda.synchronize()
{kernel_fn.__name__}({', '.join(kernel_args.keys())})
torch.cuda.synchronize()
"""
    
    def _generate_input_initialization(self, kernel_args: Dict[str, Any]) -> str:
        """Generate code to initialize kernel inputs"""
        lines = []
        for name, value in kernel_args.items():
            if isinstance(value, torch.Tensor):
                lines.append(f"{name} = torch.randn({list(value.shape)}, "
                           f"dtype=torch.{str(value.dtype).replace('torch.', '')}, "
                           f"device='cuda')")
            else:
                lines.append(f"{name} = {repr(value)}")
        return "\n".join(lines)
    
    def _parse_ncu_output(self, csv_path: str) -> Dict[str, float]:
        """Parse NCU CSV output with better error handling"""
        metrics = {}
        
        try:
            with open(csv_path, 'r') as f:
                # NCU CSV format can vary, try different parsing strategies
                content = f.read()
                
                # Try standard CSV parsing
                f.seek(0)
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Handle different NCU output formats
                    metric_name = row.get('Metric Name', row.get('metric', ''))
                    metric_value = row.get('Metric Value', row.get('value', ''))
                    
                    if metric_name and metric_value:
                        try:
                            # Clean up value
                            value_str = metric_value.strip()
                            if value_str.endswith('%'):
                                value = float(value_str[:-1])
                            else:
                                value = float(value_str.replace(',', ''))
                            
                            metrics[metric_name] = value
                        except ValueError:
                            logger.debug(f"Could not parse metric {metric_name}: {metric_value}")
                
        except Exception as e:
            logger.error(f"Error parsing NCU output: {e}")
        
        return metrics
    
    def _collect_fallback_metrics(self, result: MetricsResult, 
                                kernel_fn, kernel_args: Dict[str, Any]):
        """Collect basic metrics without NCU"""
        
        # Estimate FLOP count (very rough approximation)
        total_elements = 1
        for arg in kernel_args.values():
            if isinstance(arg, torch.Tensor):
                total_elements *= arg.numel()
        
        # Assume 2 FLOPs per element (very rough)
        estimated_flops = total_elements * 2
        
        # Calculate achieved performance
        result.achieved_tflops = (estimated_flops / result.execution_time_ms) / 1e9
        
        # Estimate memory traffic
        total_bytes = 0
        for arg in kernel_args.values():
            if isinstance(arg, torch.Tensor):
                total_bytes += arg.numel() * arg.element_size()
        
        # Assume read + write
        total_bytes *= 2
        result.achieved_bandwidth_gbps = (total_bytes / result.execution_time_ms) / 1e6
        
        # Calculate arithmetic intensity
        if total_bytes > 0:
            result.arithmetic_intensity = estimated_flops / total_bytes
        
        # Calculate efficiency (using theoretical peaks)
        if self.roofline_ceilings:
            result.compute_efficiency_pct = (
                result.achieved_tflops / self.roofline_ceilings['fp32_peak_tflops'] * 100
            )
            result.memory_efficiency_pct = (
                result.achieved_bandwidth_gbps / self.roofline_ceilings['dram_bandwidth_gbps'] * 100
            )
    
    def _populate_result_from_raw_metrics(self, result: MetricsResult, 
                                        raw_metrics: Dict[str, float]):
        """Extract key metrics from raw NCU data"""
        
        # Direct mappings
        metric_mappings = {
            "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_efficiency_pct",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
            "l2_cache_hit_rate": "l2_hit_rate_pct"
        }
        
        for ncu_name, result_attr in metric_mappings.items():
            if ncu_name in raw_metrics:
                setattr(result, result_attr, raw_metrics[ncu_name])
        
        # Calculate derived metrics
        self._calculate_derived_metrics(result, raw_metrics)
        
        # Extract stall reasons
        stall_metrics = {k: v for k, v in raw_metrics.items() 
                        if "stall" in k and v > 0}
        if stall_metrics:
            result.stall_reasons = stall_metrics
    
    def _calculate_derived_metrics(self, result: MetricsResult, 
                                 raw_metrics: Dict[str, float]):
        """Calculate performance metrics from raw data"""
        
        # FLOP calculation
        flop_metrics = [
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
        ]
        
        if all(m in raw_metrics for m in flop_metrics):
            total_flops = (
                raw_metrics[flop_metrics[0]] +
                raw_metrics[flop_metrics[1]] +
                2 * raw_metrics[flop_metrics[2]]  # FMA = 2 ops
            )
            
            if "gpu__time_duration.sum" in raw_metrics:
                time_seconds = raw_metrics["gpu__time_duration.sum"] * 1e-9
                if time_seconds > 0:
                    result.achieved_tflops = total_flops / time_seconds / 1e12
        
        # Memory bandwidth
        if all(k in raw_metrics for k in ["dram__bytes_read.sum", 
                                          "dram__bytes_write.sum",
                                          "gpu__time_duration.sum"]):
            total_bytes = (raw_metrics["dram__bytes_read.sum"] + 
                          raw_metrics["dram__bytes_write.sum"])
            time_seconds = raw_metrics["gpu__time_duration.sum"] * 1e-9
            
            if time_seconds > 0:
                result.achieved_bandwidth_gbps = total_bytes / time_seconds / 1e9
            
            # Arithmetic intensity
            if total_bytes > 0 and total_flops > 0:
                result.arithmetic_intensity = total_flops / total_bytes
        
        # Efficiency calculations
        if result.achieved_tflops and self.roofline_ceilings:
            result.compute_efficiency_pct = (
                result.achieved_tflops / self.roofline_ceilings['fp32_peak_tflops'] * 100
            )
        
        if result.achieved_bandwidth_gbps and self.roofline_ceilings:
            result.memory_efficiency_pct = (
                result.achieved_bandwidth_gbps / self.roofline_ceilings['dram_bandwidth_gbps'] * 100
            )
    
    def _generate_cache_key(self, kernel_name: str, 
                          kernel_args: Dict[str, Any]) -> str:
        """Generate cache key for metrics results"""
        # Create a simple hash based on kernel name and tensor shapes
        key_parts = [kernel_name]
        for name, value in sorted(kernel_args.items()):
            if isinstance(value, torch.Tensor):
                key_parts.append(f"{name}:{value.shape}:{value.dtype}")
            else:
                key_parts.append(f"{name}:{value}")
        return "|".join(key_parts)
    
    def save_results(self, results: List[MetricsResult], 
                    output_name: Optional[str] = None):
        """Save metrics results to multiple formats"""
        
        if not output_name:
            output_name = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save as JSON
        json_path = self.output_dir / f"{output_name}.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / f"{output_name}.csv"
        if results:
            import pandas as pd
            df = pd.DataFrame([r.to_dict() for r in results])
            df.to_csv(csv_path, index=False)
        
        # Generate report
        report_path = self.output_dir / f"{output_name}_report.txt"
        with open(report_path, 'w') as f:
            for result in results:
                f.write(f"\n{'='*60}\n")
                f.write(f"Kernel: {result.kernel_name}\n")
                f.write(f"Timestamp: {result.timestamp}\n")
                f.write(f"Execution Time: {result.execution_time_ms:.3f} ms\n")
                
                if result.raw_metrics:
                    validation = validate_metrics(result.raw_metrics)
                    report = format_metrics_report(
                        result.raw_metrics, 
                        validation,
                        self.device_caps
                    )
                    f.write(report)
                f.write(f"\n{'='*60}\n")
        
        logger.info(f"Results saved to {self.output_dir / output_name}.*")
        
        return json_path, csv_path, report_path