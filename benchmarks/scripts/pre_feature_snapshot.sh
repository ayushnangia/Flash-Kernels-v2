#!/bin/bash

# Pre-Feature Testing Snapshot Script
# Run this before implementing each feature to establish a baseline

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

FEATURE_NAME=${1:-"unknown_feature"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_DIR="benchmarks/baselines/${FEATURE_NAME}_${TIMESTAMP}"

echo "=== Pre-Feature Snapshot for: ${FEATURE_NAME} ==="
echo "Timestamp: ${TIMESTAMP}"

# Create baseline directory
mkdir -p "${BASELINE_DIR}"

# 1. Run all correctness tests and save output
echo "Running correctness tests one by one..."

# Create a combined output file
COMBINED_OUTPUT="${BASELINE_DIR}/baseline_tests.txt"
echo "=== PyTest Results ===" > "${COMBINED_OUTPUT}"
echo "Timestamp: ${TIMESTAMP}" >> "${COMBINED_OUTPUT}"
echo "Feature: ${FEATURE_NAME}" >> "${COMBINED_OUTPUT}"
echo "" >> "${COMBINED_OUTPUT}"

# Test files to run
TEST_FILES=("test_diagonal_matmul.py" "test_fused_linear_rowsum.py" "test_layer_norm.py" "test_softmax.py")

for test_file in "${TEST_FILES[@]}"; do
    echo "Running ${test_file}..."
    echo "=== ${test_file} ===" >> "${COMBINED_OUTPUT}"
    echo "Start time: $(date)" >> "${COMBINED_OUTPUT}"
    
    # Run individual test without timeout and save to separate file
    # Temporarily disable "exit on error" so that failures don't abort the whole script
    set +e
    pytest -q "evals/${test_file}" > "${BASELINE_DIR}/${test_file}.txt" 2>&1
    EXIT_CODE=$?
    set -e
    
    echo "End time: $(date)" >> "${COMBINED_OUTPUT}"
    echo "Exit code: ${EXIT_CODE}" >> "${COMBINED_OUTPUT}"
    
    if [ ${EXIT_CODE} -eq 124 ]; then
        echo "TIMEOUT: ${test_file} took longer than 5 minutes" >> "${COMBINED_OUTPUT}"
        echo "TIMEOUT: ${test_file} took longer than 5 minutes"
    elif [ ${EXIT_CODE} -eq 0 ]; then
        echo "PASS: ${test_file} completed successfully" >> "${COMBINED_OUTPUT}"
        echo "PASS: ${test_file} completed successfully"
    else
        echo "FAIL: ${test_file} failed with exit code ${EXIT_CODE}" >> "${COMBINED_OUTPUT}"
        echo "FAIL: ${test_file} failed with exit code ${EXIT_CODE}"
    fi
    
    echo "" >> "${COMBINED_OUTPUT}"
    echo "--- ${test_file} output ---" >> "${COMBINED_OUTPUT}"
    cat "${BASELINE_DIR}/${test_file}.txt" >> "${COMBINED_OUTPUT}"
    echo "" >> "${COMBINED_OUTPUT}"
    echo "========================================" >> "${COMBINED_OUTPUT}"
    echo "" >> "${COMBINED_OUTPUT}"
done

echo "Test results saved to ${BASELINE_DIR}/baseline_tests.txt"
echo "Individual test outputs saved to ${BASELINE_DIR}/"

# 2. Run benchmarks and save current data
echo "Running benchmarks..."
cp benchmarks/data/all_benchmark_data.csv "${BASELINE_DIR}/baseline_benchmark_data.csv" 2>/dev/null || echo "No existing benchmark data found"

# Run a quick benchmark for layer_norm as reference
echo "Running layer_norm benchmark..."
python3 benchmarks/scripts/benchmark_layer_norm.py > "${BASELINE_DIR}/benchmark_output.txt" 2>&1 || true

# 3. Capture system information
echo "Capturing system information..."
python3 -c "
import json
import torch
import platform
import os

system_info = {
    'python_version': platform.python_version(),
    'pytorch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'platform': platform.platform(),
    'timestamp': '${TIMESTAMP}',
    'feature_name': '${FEATURE_NAME}'
}

with open('${BASELINE_DIR}/system_info.json', 'w') as f:
    json.dump(system_info, f, indent=2)
"
echo "System info saved to ${BASELINE_DIR}/system_info.json"

# 4. Generate baseline visualizations if they exist
if [ -f "benchmarks/data/all_benchmark_data.csv" ]; then
    echo "Generating baseline visualizations..."
    python3 benchmarks/benchmark_visualizer.py --kernel-name layer_norm --metric-name speed 2>/dev/null || true
    cp benchmarks/visualizations/*.png "${BASELINE_DIR}/" 2>/dev/null || true
fi

# 5. Create summary file
echo "Creating summary..."
cat > "${BASELINE_DIR}/summary.txt" << EOF
Pre-Feature Snapshot Summary
============================
Feature: ${FEATURE_NAME}
Timestamp: ${TIMESTAMP}
Directory: ${BASELINE_DIR}

Files created:
- baseline_tests.txt: PyTest output
- baseline_benchmark_data.csv: Current benchmark data
- system_info.json: System configuration
- benchmark_output.txt: Benchmark run output
- *.png: Baseline visualizations (if any)

Next steps:
1. Implement the feature
2. Run post_feature_validate.sh ${FEATURE_NAME} ${TIMESTAMP}
EOF

echo "=== Snapshot Complete ==="
echo "Baseline saved to: ${BASELINE_DIR}"
echo "Use this timestamp for validation: ${TIMESTAMP}"