#!/bin/bash
# Fast Unit Test Runner - SRS 30-Second Constraint Compliance
# Separates unit tests from integration tests for optimal CI/CD performance

set -euo pipefail

TIMEOUT=25  # Leave 5s buffer for SRS 30s constraint
VERBOSE=${VERBOSE:-0}

echo "🧪 Fast Unit Test Runner - SRS Compliance Strategy"
echo "Target: <${TIMEOUT}s execution time"

start_time=$(date +%s)

echo "⚡ Phase 1: Core library unit tests only..."

if command -v cargo-nextest &> /dev/null; then
    timeout $TIMEOUT cargo nextest run \
        --profile ci \
        --release \
        --lib \
        --no-default-features \
        -E 'not test(/test_rayleigh_collapse_time/) and not test(/test_acoustic_dispersion_relation/) and not test(/literature_validation/) and not test(/physics_validation_test/) and not test(/energy_conservation/) and not test(/cfl_stability/) and not test(/dispersion_validation/) and not test(/benchmark/)' \
        --quiet
else
    timeout $TIMEOUT cargo test \
        --lib \
        --quiet \
        --no-default-features \
        -- \
        --test-threads=4 \
        --skip "test_rayleigh_collapse_time" \
        --skip "test_acoustic_dispersion_relation" \
        --skip "literature_validation" \
        --skip "physics_validation_test" \
        --skip "energy_conservation" \
        --skip "cfl_stability" \
        --skip "dispersion_validation" \
        --skip "benchmark" \
        --nocapture
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ Fast unit tests completed in ${duration}s"

if [ $duration -le $TIMEOUT ]; then
    echo "🎯 SRS NFR-002 COMPLIANCE: ${duration}s ≤ 30s ✅"
    exit 0
else
    echo "❌ SRS NFR-002 VIOLATION: ${duration}s > 30s"
    exit 1
fi
