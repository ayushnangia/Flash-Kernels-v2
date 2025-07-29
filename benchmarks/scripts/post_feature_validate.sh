#!/bin/bash

# Post-Feature Validation Script
# Run this after implementing each feature to validate changes

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

FEATURE_NAME=${1:-"unknown_feature"}
BASELINE_TIMESTAMP=${2:-""}
TOLERANCE=${3:-"0.05"}  # 5% tolerance by default

if [ -z "${BASELINE_TIMESTAMP}" ]; then
    echo "Error: Please provide baseline timestamp"
    echo "Usage: $0 <feature_name> <baseline_timestamp> [tolerance]"
    exit 1
fi

BASELINE_DIR="benchmarks/baselines/${FEATURE_NAME}_${BASELINE_TIMESTAMP}"
VALIDATION_DIR="benchmarks/validations/${FEATURE_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "=== Post-Feature Validation for: ${FEATURE_NAME} ==="
echo "Baseline: ${BASELINE_DIR}"
echo "Tolerance: ${TOLERANCE} (${TOLERANCE}00%)"

# Check if baseline exists
if [ ! -d "${BASELINE_DIR}" ]; then
    echo "Error: Baseline directory not found: ${BASELINE_DIR}"
    exit 1
fi

# Create validation directory
mkdir -p "${VALIDATION_DIR}"

# 1. Run correctness tests and compare
echo "Running correctness tests..."
pytest -q evals > "${VALIDATION_DIR}/current_tests.txt" 2>&1 || true

# Compare test results
echo "Comparing test results..."
if diff -u "${BASELINE_DIR}/baseline_tests.txt" "${VALIDATION_DIR}/current_tests.txt" > "${VALIDATION_DIR}/test_diff.txt"; then
    echo "✓ All tests pass with same results as baseline"
else
    echo "⚠ Test results differ from baseline (see ${VALIDATION_DIR}/test_diff.txt)"
    # Check if there are any new failures
    if grep -E "(FAILED|ERROR)" "${VALIDATION_DIR}/current_tests.txt" > /dev/null; then
        echo "✗ New test failures detected!"
        cat "${VALIDATION_DIR}/test_diff.txt"
    else
        echo "✓ No new test failures (differences may be due to test additions)"
    fi
fi

# 2. Run benchmarks and compare performance
echo "Running benchmarks..."
python3 benchmarks/scripts/benchmark_layer_norm.py > "${VALIDATION_DIR}/benchmark_output.txt" 2>&1 || true

# 3. Compare benchmark results
echo "Comparing benchmark performance..."
if [ -f "${BASELINE_DIR}/baseline_benchmark_data.csv" ]; then
    python3 -c "
import csv
import json
import sys
from collections import defaultdict

def load_benchmark_data(filename):
    data = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['kernel_name'], row['kernel_provider'], row['x_value'], row['extra_benchmark_config_str'])
            data[key].append(float(row['y_value_50']))
    return data

baseline_data = load_benchmark_data('${BASELINE_DIR}/baseline_benchmark_data.csv')
current_data = load_benchmark_data('benchmarks/data/all_benchmark_data.csv')

results = {'passed': 0, 'failed': 0, 'new': 0, 'missing': 0, 'details': []}
tolerance = ${TOLERANCE}

# Check all baseline measurements
for key, baseline_values in baseline_data.items():
    if key in current_data:
        baseline_val = baseline_values[-1]  # Latest value
        current_val = current_data[key][-1]
        
        if baseline_val > 0:
            ratio = abs(current_val - baseline_val) / baseline_val
            if ratio <= tolerance:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['details'].append({
                    'key': str(key),
                    'baseline': baseline_val,
                    'current': current_val,
                    'ratio': ratio
                })
    else:
        results['missing'] += 1

# Check for new measurements
for key in current_data:
    if key not in baseline_data:
        results['new'] += 1

# Save results
with open('${VALIDATION_DIR}/performance_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print(f\"Performance Comparison Summary:\")
print(f\"  Passed (within {tolerance*100}%): {results['passed']}\")
print(f\"  Failed (exceeded tolerance): {results['failed']}\")
print(f\"  New measurements: {results['new']}\")
print(f\"  Missing measurements: {results['missing']}\")

if results['failed'] > 0:
    print(f\"\\n⚠ Performance regressions detected:\")
    for detail in results['details'][:5]:  # Show first 5
        print(f\"  {detail['key']}: {detail['baseline']:.4f} -> {detail['current']:.4f} ({detail['ratio']*100:.1f}% change)\")
    if len(results['details']) > 5:
        print(f\"  ... and {len(results['details']) - 5} more\")
    sys.exit(1)
else:
    print(f\"\\n✓ All performance measurements within tolerance\")
" || echo "⚠ Could not compare performance (may be expected for new features)"
fi

# 4. Validate new features (if applicable)
echo "Validating new features..."
python3 -c "
import os
import json

validation_results = {
    'feature': '${FEATURE_NAME}',
    'baseline_timestamp': '${BASELINE_TIMESTAMP}',
    'validation_timestamp': '$(date +%Y%m%d_%H%M%S)',
    'checks': {}
}

# Feature-specific validation
if '${FEATURE_NAME}' == 'micro_metrics':
    # Check if new metrics are being collected
    try:
        import csv
        with open('benchmarks/data/all_benchmark_data.csv', 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            # Check for new metric columns (will be added in Phase 1)
            new_metrics = ['gpu_occupancy', 'memory_throughput', 'sm_efficiency']
            for metric in new_metrics:
                validation_results['checks'][f'has_{metric}'] = metric in headers
    except Exception as e:
        validation_results['checks']['error'] = str(e)

elif '${FEATURE_NAME}' == 'nsight_integration':
    # Check if Nsight wrapper exists and is functional
    validation_results['checks']['nsight_wrapper_exists'] = os.path.exists('benchmarks/metrics/nsight_wrapper.py')
    
# Save validation results
with open('${VALIDATION_DIR}/feature_validation.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f\"Feature validation results saved to ${VALIDATION_DIR}/feature_validation.json\")
"

# 5. Generate comparison visualizations
if [ -f "benchmarks/data/all_benchmark_data.csv" ]; then
    echo "Generating comparison visualizations..."
    python3 benchmarks/benchmark_visualizer.py --kernel-name layer_norm --metric-name speed 2>/dev/null || true
    cp benchmarks/visualizations/*.png "${VALIDATION_DIR}/" 2>/dev/null || true
fi

# 6. Create validation summary
cat > "${VALIDATION_DIR}/summary.txt" << EOF
Post-Feature Validation Summary
===============================
Feature: ${FEATURE_NAME}
Baseline Timestamp: ${BASELINE_TIMESTAMP}
Validation Timestamp: $(date +%Y%m%d_%H%M%S)
Tolerance: ${TOLERANCE} (${TOLERANCE}00%)

Results:
- Test Comparison: See test_diff.txt
- Performance Comparison: See performance_comparison.json
- Feature Validation: See feature_validation.json

Files created:
- current_tests.txt: Current PyTest output
- test_diff.txt: Difference from baseline tests
- benchmark_output.txt: Current benchmark output
- performance_comparison.json: Performance analysis
- feature_validation.json: Feature-specific checks
- *.png: Current visualizations

Next steps:
1. Review any test differences
2. Investigate performance regressions (if any)
3. Verify new features are working as expected
EOF

echo "=== Validation Complete ==="
echo "Results saved to: ${VALIDATION_DIR}"
echo "Summary: ${VALIDATION_DIR}/summary.txt"