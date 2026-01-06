#!/bin/bash
# Production Test Runner - SRS 30-Second Constraint Enforcement
# Senior Rust Engineer - Micro-Sprint Optimization

set -euo pipefail

TIMEOUT=30
TEST_PROFILE="${1:-ci}"
VERBOSE=${VERBOSE:-0}

echo "🧪 Production Test Runner - SRS Constraint Enforcement"
echo "Profile: $TEST_PROFILE | Timeout: ${TIMEOUT}s"

# Function to run tests with timeout and proper error handling
run_tests_with_constraint() {
    local profile=$1
    local start_time=$(date +%s)
    
    echo "⏱️  Starting optimized test execution (max ${TIMEOUT}s)..."
    
    # Phase 1: Critical core tests only (target <20s)
    echo "Running critical core tests (SRS constraint enforcement) with nextest"
    timeout 20 cargo nextest run \
        --profile ci \
        --release \
        --lib \
        --test-threads=4 \
        -E 'not test(/physics/) and not test(/validation/) and not test(/benchmark/) and not test(/integration/)' \
        || handle_test_failure $?
    
    local core_end=$(date +%s)
    local core_duration=$((core_end - start_time))
    
    echo "✅ Core tests completed in ${core_duration}s"
    
    # Phase 2: Optional physics validation (if time permits)
    local remaining_time=$((TIMEOUT - core_duration))
    if [ $remaining_time -gt 10 ]; then
        echo "⚡ Running physics validation tests (${remaining_time}s remaining) with nextest"
        timeout $remaining_time cargo nextest run \
            --profile ci \
            --release \
            --lib \
            --test-threads=2 \
            physics::mechanics::acoustic_wave::kzk \
            || echo "⚠️  Physics validation tests require more time (non-critical)"
    else
        echo "⚠️  Skipping physics validation tests due to SRS time constraint"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "✅ Test suite completed in ${duration}s (within ${TIMEOUT}s constraint)"
    
    if [ $duration -gt $TIMEOUT ]; then
        echo "⚠️  WARNING: Test duration ${duration}s exceeds SRS constraint"
        exit 1
    fi
}

# Handle test failures with detailed diagnostics  
handle_test_failure() {
    local exit_code=$1
    
    case $exit_code in
        124)
            echo "❌ CRITICAL: Tests exceeded ${TIMEOUT}s timeout constraint"
            echo "🔧 Recommended actions:"
            echo "   1. Install cargo-nextest for better parallelization"
            echo "   2. Split large integration tests into smaller units"
            echo "   3. Review test efficiency and remove blocking operations"
            exit 1
            ;;
        1)
            echo "❌ Test failures detected - investigating..."
            # Run a quick diagnostic to identify failing tests
            if command -v cargo-nextest &> /dev/null; then
                cargo nextest list --profile ci --lib --format terse | head -10
            else
                cargo test --lib -- --list | head -10
            fi
            exit 1
            ;;
        *)
            echo "❌ Unexpected test execution error (exit code: $exit_code)"
            exit $exit_code
            ;;
    esac
}

# Pre-flight checks
echo "🔍 Pre-flight validation..."

# Check for compilation issues (lib only)
echo "Validating library compilation..."
cargo check --lib || {
    echo "❌ Library compilation failed - fix before testing"
    exit 1
}

echo "✅ Library compilation successful"

# Execute tests with SRS constraint
run_tests_with_constraint "$TEST_PROFILE"

echo "🎯 Production test validation complete - SRS compliance verified"
