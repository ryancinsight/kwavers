#!/bin/bash
# Script to run tests with individual timeouts to identify hanging tests

source "/usr/local/cargo/env"

echo "Running tests with 5-second timeout per test..."
echo "================================================"

# Get list of all test names
if command -v cargo-nextest &> /dev/null; then
    TEST_LIST=$(cargo nextest list --profile diagnostic --lib --format terse 2>/dev/null | sed 's/^[[:space:]]*//' | grep -E "^[A-Za-z0-9_]+::" || true)
else
    TEST_LIST=$(cargo test --lib -- --list 2>/dev/null | grep -E "^[a-z_]+::" | grep ": test$" | sed 's/: test$//')
fi

PASSED=0
FAILED=0
TIMEOUT_COUNT=0

for test in $TEST_LIST; do
    echo -n "Testing $test... "
    
    # Run test with timeout
    if command -v cargo-nextest &> /dev/null; then
        if timeout 10 cargo nextest run --profile diagnostic --lib -E "test(=$test)" --quiet &>/dev/null; then
            echo "✓ PASS"
            ((PASSED++))
        else
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
                echo "⏱ TIMEOUT"
                ((TIMEOUT_COUNT++))
            else
                echo "✗ FAIL"
                ((FAILED++))
            fi
        fi
    elif timeout 5 cargo test --lib "$test" -- --exact --nocapture &>/dev/null; then
        echo "✓ PASS"
        ((PASSED++))
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
            echo "⏱ TIMEOUT"
            ((TIMEOUT_COUNT++))
        else
            echo "✗ FAIL"
            ((FAILED++))
        fi
    fi
done

echo "================================================"
echo "Summary:"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Timeout: $TIMEOUT_COUNT"
echo "  Total:   $((PASSED + FAILED + TIMEOUT_COUNT))"
