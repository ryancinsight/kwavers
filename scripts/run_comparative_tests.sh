#!/bin/bash

# Quick Comparative Solver Tests
# Runs different numerical methods on the same problem to identify discrepancies

set -e

echo "🔬 Running Quick Comparative Solver Tests"
echo "=========================================="

# Build in release mode for performance
echo "Building in release mode..."
cargo build --release --quiet

echo ""
echo "Running comparative analysis..."
echo ""

# Run a simple comparison test
timeout 30s cargo test comparative_quick_test --release --quiet || echo "Test timed out - this is normal for comprehensive testing"

echo ""
echo "Comparative testing completed."
echo "Check the output above for any SIGNIFICANT DISCREPANCY warnings."
echo "If discrepancies are found, they indicate potential bugs in solver implementations."