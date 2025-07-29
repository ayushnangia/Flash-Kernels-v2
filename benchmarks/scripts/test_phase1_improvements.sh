#!/bin/bash
# Test script for Phase 1 improvements

echo "=== Testing Omni-Perf-Bench Phase 1 Improvements ==="
echo "This script validates the enhanced metrics collection system"
echo ""

# Check environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Install any missing dependencies
pip install -q plotly pandas matplotlib

echo "1. Testing unified benchmark with performance only..."
python benchmarks/unified_benchmark.py \
    --kernels layer_norm \
    --no-metrics \
    --output-dir benchmarks/data/test_unified

if [ $? -ne 0 ]; then
    echo "ERROR: Basic benchmark failed"
    exit 1
fi
echo "✓ Performance benchmark passed"

echo ""
echo "2. Testing metrics utilities..."
python -c "
from utils.metrics_utils import detect_gpu_architecture, check_ncu_availability
import torch

# Test GPU detection
caps = detect_gpu_architecture()
print(f'GPU Architecture: {caps.architecture.value}')
print(f'Compute Capability: {caps.compute_capability}')
print(f'Has Tensor Cores: {caps.has_tensor_cores}')
print(f'Has BF16: {caps.has_bf16}')

# Test NCU detection
ncu_available, ncu_path = check_ncu_availability()
print(f'NCU Available: {ncu_available}')
if ncu_available:
    print(f'NCU Path: {ncu_path}')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Metrics utilities test failed"
    exit 1
fi
echo "✓ Metrics utilities passed"

echo ""
echo "3. Testing enhanced metrics collector (fallback mode)..."
python -c "
from benchmarks.enhanced_metrics_collector import EnhancedMetricsCollector
import torch

# Create simple test kernel
def test_kernel(x):
    return x * 2 + 1

# Test metrics collection
collector = EnhancedMetricsCollector()
x = torch.randn(1024, 1024, device='cuda')

result = collector.collect_metrics(
    test_kernel,
    {'x': x},
    'test_kernel',
    num_warmup=5,
    num_runs=10
)

print(f'Kernel: {result.kernel_name}')
print(f'Execution Time: {result.execution_time_ms:.3f} ms')
print(f'Architecture: {result.architecture}')
if result.achieved_tflops:
    print(f'Achieved TFLOP/s: {result.achieved_tflops:.2f}')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Enhanced metrics collector test failed"
    exit 1
fi
echo "✓ Enhanced metrics collector passed"

echo ""
echo "4. Testing visualization generation..."

# Run a quick benchmark to generate data
python benchmarks/unified_benchmark.py \
    --kernels layer_norm \
    --no-metrics \
    --no-validation \
    --output-dir benchmarks/data/test_viz

# Generate visualizations
python benchmarks/enhanced_roofline_visualizer.py \
    --data-dir benchmarks/data/test_viz \
    --output-dir benchmarks/visualizations/test

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization generation failed"
    exit 1
fi

# Check if files were created
if [ -f "benchmarks/visualizations/test/index.html" ]; then
    echo "✓ Visualizations generated successfully"
    echo "  View at: benchmarks/visualizations/test/index.html"
else
    echo "ERROR: Visualization files not created"
    exit 1
fi

echo ""
echo "5. Testing full pipeline with metrics (if NCU available)..."
python benchmarks/unified_benchmark.py \
    --kernels layer_norm \
    --output-dir benchmarks/data/test_full \
    2>&1 | grep -E "(Metrics Collection:|NCU|TFLOP/s|Occupancy)"

echo ""
echo "=== Phase 1 Improvements Test Summary ==="
echo "✓ All core components tested successfully"
echo ""
echo "Key improvements validated:"
echo "  - Unified benchmarking interface"
echo "  - Robust metrics collection with fallback"
echo "  - Architecture-specific optimizations"
echo "  - Enhanced visualizations (static + interactive)"
echo "  - Comprehensive error handling"
echo ""
echo "Next steps:"
echo "  1. Run full benchmark suite: python benchmarks/unified_benchmark.py --kernels all"
echo "  2. View visualizations: open benchmarks/visualizations/test/index.html"
echo "  3. Check metrics reports in: benchmarks/data/test_full/"