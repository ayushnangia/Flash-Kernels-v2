"""
Utilities for GPU metrics collection and analysis
Part of Omni-Perf-Bench enhanced metrics infrastructure
"""

import torch
import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GPUArchitecture(Enum):
    """GPU Architecture families"""
    VOLTA = "volta"          # SM 7.0
    TURING = "turing"        # SM 7.5
    AMPERE = "ampere"        # SM 8.0, 8.6
    ADA = "ada"              # SM 8.9
    HOPPER = "hopper"        # SM 9.0
    UNKNOWN = "unknown"

@dataclass
class GPUCapabilities:
    """GPU capabilities and limits"""
    architecture: GPUArchitecture
    compute_capability: Tuple[int, int]
    has_tensor_cores: bool
    has_bf16: bool
    has_tf32: bool
    fp32_cores_per_sm: int
    tensor_cores_per_sm: int
    max_threads_per_sm: int
    max_warps_per_sm: int
    shared_memory_per_sm_kb: int
    l2_cache_size_mb: float
    memory_bus_width_bits: int

def detect_gpu_architecture(device: Optional[torch.device] = None) -> GPUCapabilities:
    """Detect GPU architecture and capabilities"""
    
    if device is None:
        device = torch.cuda.current_device()
    
    props = torch.cuda.get_device_properties(device)
    major, minor = props.major, props.minor
    
    # Determine architecture
    if (major, minor) == (7, 0):
        arch = GPUArchitecture.VOLTA
        fp32_cores = 64
        tensor_cores = 8
    elif (major, minor) == (7, 5):
        arch = GPUArchitecture.TURING
        fp32_cores = 64
        tensor_cores = 8
    elif major == 8 and minor in [0, 6]:
        arch = GPUArchitecture.AMPERE
        fp32_cores = 128  # 2x FP32 units in Ampere
        tensor_cores = 4
    elif (major, minor) == (8, 9):
        arch = GPUArchitecture.ADA
        fp32_cores = 128
        tensor_cores = 4
    elif major == 9:
        arch = GPUArchitecture.HOPPER
        fp32_cores = 128
        tensor_cores = 4
    else:
        arch = GPUArchitecture.UNKNOWN
        fp32_cores = 32  # Conservative estimate
        tensor_cores = 0
    
    return GPUCapabilities(
        architecture=arch,
        compute_capability=(major, minor),
        has_tensor_cores=major >= 7,
        has_bf16=major >= 8,
        has_tf32=major >= 8,
        fp32_cores_per_sm=fp32_cores,
        tensor_cores_per_sm=tensor_cores,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        max_warps_per_sm=props.max_threads_per_multi_processor // 32,
        shared_memory_per_sm_kb=props.shared_memory_per_multi_processor // 1024,
        l2_cache_size_mb=props.l2_cache_size / (1024 * 1024) if hasattr(props, 'l2_cache_size') else 0,
        memory_bus_width_bits=props.memory_bus_width
    )

def get_architecture_specific_metrics(arch: GPUArchitecture) -> List[str]:
    """Get architecture-specific NCU metrics"""
    
    base_metrics = [
        # Occupancy and efficiency
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        
        # Memory metrics
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2_cache__throughput.avg.pct_of_peak_sustained_elapsed",
        "shared__throughput.avg.pct_of_peak_sustained_elapsed",
        
        # Instruction metrics
        "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_elapsed",
        
        # Stall analysis
        "smsp__warp_stall_long_sb_pct",
        "smsp__warp_stall_wait_pct",
        "smsp__warp_stall_sync_pct",
        "smsp__warp_stall_mem_throttle_pct",
        
        # Memory access patterns
        "l1_cache_global_hit_rate",
        "l2_cache_hit_rate",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        
        # Execution metrics
        "sm__cycles_elapsed.avg",
        "gpu__time_duration.sum",
    ]
    
    # Add architecture-specific metrics
    if arch in [GPUArchitecture.AMPERE, GPUArchitecture.ADA, GPUArchitecture.HOPPER]:
        base_metrics.extend([
            # Tensor core metrics
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed",
            
            # BF16/TF32 metrics
            "smsp__sass_thread_inst_executed_op_bf16_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_tf32_pred_on.sum",
        ])
    
    if arch == GPUArchitecture.HOPPER:
        base_metrics.extend([
            # H100 specific metrics
            "sm__mma_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "smsp__thread_inst_executed_per_inst_executed.ratio",
        ])
    
    return base_metrics

def check_ncu_availability() -> Tuple[bool, Optional[str]]:
    """Check if NCU is available and return its path"""
    
    try:
        # Try common NCU locations
        ncu_paths = [
            "ncu",  # In PATH
            "/usr/local/cuda/bin/ncu",
            "/opt/nvidia/nsight-compute/ncu",
            os.path.expanduser("~/nsight-compute/ncu"),
        ]
        
        for ncu_path in ncu_paths:
            try:
                result = subprocess.run(
                    [ncu_path, "--version"], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return True, ncu_path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        return False, None
        
    except Exception as e:
        logger.warning(f"Error checking NCU availability: {e}")
        return False, None

def validate_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Validate collected metrics and provide diagnostics"""
    
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "suggestions": []
    }
    
    # Check for missing critical metrics
    critical_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "gpu__time_duration.sum"
    ]
    
    for metric in critical_metrics:
        if metric not in metrics or metrics[metric] is None:
            validation["errors"].append(f"Missing critical metric: {metric}")
            validation["valid"] = False
    
    # Check for suspicious values
    if "sm__warps_active.avg.pct_of_peak_sustained_active" in metrics:
        occupancy = metrics["sm__warps_active.avg.pct_of_peak_sustained_active"]
        if occupancy < 10:
            validation["warnings"].append(f"Very low occupancy: {occupancy:.1f}%")
            validation["suggestions"].append("Consider increasing block size or reducing register usage")
        elif occupancy > 95:
            validation["warnings"].append(f"Suspiciously high occupancy: {occupancy:.1f}%")
    
    # Check memory efficiency
    if all(k in metrics for k in ["dram__throughput.avg.pct_of_peak_sustained_elapsed", 
                                   "l2_cache_hit_rate"]):
        mem_throughput = metrics["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
        l2_hit_rate = metrics.get("l2_cache_hit_rate", 0)
        
        if mem_throughput > 80 and l2_hit_rate < 50:
            validation["warnings"].append("High memory throughput with low L2 hit rate")
            validation["suggestions"].append("Consider data layout optimization or tiling")
    
    # Check for stalls
    stall_metrics = [k for k in metrics.keys() if "stall" in k]
    if stall_metrics:
        total_stall = sum(metrics.get(k, 0) for k in stall_metrics)
        if total_stall > 50:
            validation["warnings"].append(f"High warp stalls: {total_stall:.1f}%")
            
            # Identify dominant stall reason
            max_stall = max(stall_metrics, key=lambda k: metrics.get(k, 0))
            if "long_sb" in max_stall:
                validation["suggestions"].append("Reduce shared memory bank conflicts")
            elif "sync" in max_stall:
                validation["suggestions"].append("Minimize synchronization points")
            elif "mem_throttle" in max_stall:
                validation["suggestions"].append("Reduce memory pressure or use better access patterns")
    
    return validation

def calculate_roofline_ceilings(device_caps: GPUCapabilities, 
                               clock_rate_ghz: float) -> Dict[str, float]:
    """Calculate various roofline ceilings for the device"""
    
    ceilings = {}
    
    # FP32 compute ceilings
    fp32_peak_flops = (device_caps.fp32_cores_per_sm * 
                       torch.cuda.get_device_properties(0).multi_processor_count * 
                       clock_rate_ghz * 1e9 * 2)  # 2 ops per FMA
    
    ceilings['fp32_peak_tflops'] = fp32_peak_flops / 1e12
    
    # Tensor core ceilings (if available)
    if device_caps.has_tensor_cores:
        # Approximate tensor core throughput
        if device_caps.architecture == GPUArchitecture.AMPERE:
            tensor_ops_per_cycle = 256  # 16x16x16 matrix per tensor core
        elif device_caps.architecture == GPUArchitecture.HOPPER:
            tensor_ops_per_cycle = 512  # Larger matrix operations
        else:
            tensor_ops_per_cycle = 128  # Volta/Turing
        
        tensor_peak_flops = (device_caps.tensor_cores_per_sm * 
                            torch.cuda.get_device_properties(0).multi_processor_count *
                            clock_rate_ghz * 1e9 * tensor_ops_per_cycle)
        
        ceilings['tensor_peak_tflops'] = tensor_peak_flops / 1e12
        
        if device_caps.has_bf16:
            ceilings['bf16_peak_tflops'] = ceilings['tensor_peak_tflops'] * 2
        
        if device_caps.has_tf32:
            ceilings['tf32_peak_tflops'] = ceilings['fp32_peak_tflops'] * 2
    
    # Memory bandwidth ceilings
    props = torch.cuda.get_device_properties(0)
    memory_clock_ghz = props.memory_clock_rate / 1e6
    
    # DRAM bandwidth
    dram_bandwidth_gbps = (memory_clock_ghz * device_caps.memory_bus_width_bits * 2) / 8
    ceilings['dram_bandwidth_gbps'] = dram_bandwidth_gbps
    
    # L2 cache bandwidth (approximate - typically 2-4x DRAM)
    ceilings['l2_bandwidth_gbps'] = dram_bandwidth_gbps * 3
    
    # Shared memory bandwidth (per SM)
    ceilings['smem_bandwidth_gbps_per_sm'] = 128  # Approximate for modern GPUs
    
    # Calculate ridge points
    ceilings['fp32_ridge_point'] = (ceilings['fp32_peak_tflops'] * 1e12) / (dram_bandwidth_gbps * 1e9)
    
    if 'tensor_peak_tflops' in ceilings:
        ceilings['tensor_ridge_point'] = (ceilings['tensor_peak_tflops'] * 1e12) / (dram_bandwidth_gbps * 1e9)
    
    return ceilings

def get_cupti_metrics() -> Dict[str, float]:
    """Fallback to CUPTI-based metrics collection (simplified)"""
    
    # This is a placeholder for CUPTI integration
    # In practice, this would use pycuda or cupy's profiling APIs
    logger.warning("CUPTI metrics collection not yet implemented")
    return {}

def format_metrics_report(metrics: Dict[str, float], 
                         validation: Dict[str, Any],
                         device_caps: GPUCapabilities) -> str:
    """Format metrics into a human-readable report"""
    
    report = []
    report.append(f"=== GPU Metrics Report ===")
    report.append(f"Architecture: {device_caps.architecture.value}")
    report.append(f"Compute Capability: {device_caps.compute_capability}")
    report.append("")
    
    # Occupancy and efficiency
    if "sm__warps_active.avg.pct_of_peak_sustained_active" in metrics:
        report.append(f"Occupancy: {metrics['sm__warps_active.avg.pct_of_peak_sustained_active']:.1f}%")
    
    if "sm__throughput.avg.pct_of_peak_sustained_elapsed" in metrics:
        report.append(f"SM Efficiency: {metrics['sm__throughput.avg.pct_of_peak_sustained_elapsed']:.1f}%")
    
    # Memory metrics
    report.append("\nMemory Performance:")
    if "dram__throughput.avg.pct_of_peak_sustained_elapsed" in metrics:
        report.append(f"  DRAM Throughput: {metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed']:.1f}%")
    
    if "l2_cache_hit_rate" in metrics:
        report.append(f"  L2 Hit Rate: {metrics['l2_cache_hit_rate']:.1f}%")
    
    # Stall analysis
    stall_metrics = [(k, v) for k, v in metrics.items() if "stall" in k and v > 5]
    if stall_metrics:
        report.append("\nWarp Stalls:")
        for metric, value in sorted(stall_metrics, key=lambda x: x[1], reverse=True):
            stall_type = metric.split("__")[-1].replace("_pct", "").replace("_", " ").title()
            report.append(f"  {stall_type}: {value:.1f}%")
    
    # Validation results
    if validation["warnings"]:
        report.append("\nWarnings:")
        for warning in validation["warnings"]:
            report.append(f"  ⚠ {warning}")
    
    if validation["suggestions"]:
        report.append("\nOptimization Suggestions:")
        for suggestion in validation["suggestions"]:
            report.append(f"  → {suggestion}")
    
    return "\n".join(report)