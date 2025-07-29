#!/bin/bash
# Run Phase 1 benchmarks with enhanced metrics collection

echo "=== Omni-Perf-Bench Phase 1: Enhanced Metrics Collection ==="
echo "This script runs benchmarks with microarchitectural metrics"
echo ""

# Check if running in Docker/container with GPU access
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected. Please ensure:"
    echo "  - You have an NVIDIA GPU installed"
    echo "  - NVIDIA drivers are properly installed"
    echo "  - If using Docker, use --gpus all flag"
    exit 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check for Nsight Compute
if ! command -v ncu &> /dev/null; then
    echo "Warning: NVIDIA Nsight Compute (ncu) not found"
    echo "Microarchitectural metrics collection will be disabled"
    echo ""
    echo "To install Nsight Compute:"
    echo "  1. Download from: https://developer.nvidia.com/nsight-compute"
    echo "  2. Add to PATH: export PATH=/path/to/nsight-compute/bin:\$PATH"
    echo ""
    echo "Continuing with performance metrics only..."
    METRICS_FLAG="--no-metrics"
else
    echo "Found Nsight Compute at: $(which ncu)"
    METRICS_FLAG=""
fi

# Create output directories
mkdir -p benchmarks/data/enhanced
mkdir -p benchmarks/visualizations/roofline

# Run enhanced benchmarks for all kernels
echo ""
echo "Running enhanced benchmarks..."
python benchmarks/scripts/benchmark_with_metrics.py --all $METRICS_FLAG

# Generate roofline visualizations if we have metrics
if [ -z "$METRICS_FLAG" ]; then
    echo ""
    echo "Generating roofline visualizations..."
    python benchmarks/roofline_visualizer.py
fi

# Generate comparison report
echo ""
echo "Generating Phase 1 summary report..."
python -c "
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Load latest results
data_dir = Path('benchmarks/data/enhanced')
results = []

for json_file in sorted(data_dir.glob('*_enhanced_benchmark_*.json'), reverse=True)[:4]:
    with open(json_file, 'r') as f:
        data = json.load(f)
        for r in data['results']:
            r['device'] = data['device']
        results.extend(data['results'])

if results:
    df = pd.DataFrame(results)
    
    # Summary statistics
    print('\\n=== Phase 1 Implementation Summary ===')
    print(f'Total benchmarks run: {len(df)}')
    print(f'Kernels tested: {df[\"kernel\"].nunique()}')
    print(f'Device: {df[\"device\"].iloc[0] if \"device\" in df else \"Unknown\"}')
    
    if 'occupancy_pct' in df.columns:
        print('\\n--- Microarchitectural Metrics ---')
        metrics = ['occupancy_pct', 'sm_efficiency_pct', 'memory_throughput_pct', 
                  'compute_efficiency_pct', 'memory_efficiency_pct']
        
        for metric in metrics:
            if metric in df.columns and df[metric].notna().any():
                print(f'{metric}: {df[metric].mean():.1f}% (avg)')
    
    print('\\n--- Performance Summary ---')
    for kernel in df['kernel'].unique():
        kernel_df = df[df['kernel'] == kernel]
        avg_speedup = kernel_df['speedup'].mean() if 'speedup' in kernel_df else None
        if avg_speedup:
            print(f'{kernel}: {avg_speedup:.2f}x average speedup vs PyTorch')
    
    # Save summary
    summary_path = Path('benchmarks/data/enhanced/phase1_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_benchmarks': len(df),
        'kernels_tested': list(df['kernel'].unique()),
        'metrics_collected': 'occupancy_pct' in df.columns,
        'average_metrics': {
            metric: float(df[metric].mean()) 
            for metric in ['occupancy_pct', 'sm_efficiency_pct', 'memory_throughput_pct']
            if metric in df.columns and df[metric].notna().any()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'\\nSummary saved to: {summary_path}')
else:
    print('No results found')
"

echo ""
echo "=== Phase 1 Benchmark Complete ==="
echo ""
echo "Next steps:"
echo "  1. Review roofline plots in benchmarks/visualizations/roofline/"
echo "  2. Check detailed metrics in benchmarks/data/enhanced/"
echo "  3. Run validation: bash benchmarks/scripts/post_feature_validate.sh phase1_metrics 20250729_121844"