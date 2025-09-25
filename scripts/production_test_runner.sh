#!/bin/bash
# Production Test Runner - SRS NFR-002 Compliance
# Requirement: Complete test suite execution <30s

set -euo pipefail

# Performance constants
readonly MAX_TOTAL_TIME=30
readonly MAX_SINGLE_TEST_TIME=5
readonly PARALLEL_JOBS=4

echo "🚀 PRODUCTION TEST SUITE - SRS NFR-002 COMPLIANCE"
echo "=================================================="
echo "Max total time: ${MAX_TOTAL_TIME}s"
echo "Max single test: ${MAX_SINGLE_TEST_TIME}s"
echo "Parallel jobs: ${PARALLEL_JOBS}"
echo

start_time=$(date +%s)

# Use nextest if available, otherwise fallback to cargo test
if command -v cargo-nextest &> /dev/null; then
    echo "📊 Using nextest for parallel execution"
    
    # Configure nextest for production requirements
    export NEXTEST_PROFILE=ci
    export RUST_BACKTRACE=0
    export RUST_LOG=error
    
    timeout ${MAX_TOTAL_TIME} cargo nextest run \
        --all-features \
        --no-fail-fast \
        --test-threads=${PARALLEL_JOBS} \
        --final-status-level=pass
else
    echo "⚡ Using cargo test with optimizations"
    
    # Configure for performance
    export RUST_BACKTRACE=0 
    export RUST_LOG=error
    
    timeout ${MAX_TOTAL_TIME} cargo test \
        --all-features \
        -- --test-threads=${PARALLEL_JOBS} --nocapture
fi

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo
echo "✅ TEST EXECUTION COMPLETE"
echo "Total time: ${total_time}s (limit: ${MAX_TOTAL_TIME}s)"

if [ ${total_time} -gt ${MAX_TOTAL_TIME} ]; then
    echo "❌ SRS NFR-002 VIOLATION: Test execution exceeded ${MAX_TOTAL_TIME}s limit"
    exit 1
else
    echo "✅ SRS NFR-002 COMPLIANCE: Test execution within limits"
fi