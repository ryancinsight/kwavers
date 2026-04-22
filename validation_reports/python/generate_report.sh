#!/bin/bash
#
# Validation Report Generator for pykwavers vs k-wave-python
#
# Usage: ./generate_report.sh [options]
# Options:
#   --format=FORMAT    Output format: markdown, xml (junit), or json (default: markdown)
#   --output=DIR       Output directory (default: kwavers/validation_reports/python)
#   --python           Enable Python/k-wave-python comparison (requires KWAVERS_RUN_PYTHON=1)
#   --verbose          Verbose output
#   --help             Show this help message
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPORT_DIR="${SCRIPT_DIR}"
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
FORMAT="markdown"
VERBOSE=false
RUN_PYTHON=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format=*)
            FORMAT="${1#*=}"
            shift
            ;;
        --output=*)
            REPORT_DIR="${1#*=}"
            shift
            ;;
        --python)
            RUN_PYTHON=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Validation Report Generator for pykwavers vs k-wave-python"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --format=FORMAT    Output format: markdown, xml (junit), or json (default: markdown)"
            echo "  --output=DIR       Output directory (default: kwavers/validation_reports/python)"
            echo "  --python           Enable Python/k-wave-python comparison"
            echo "  --verbose          Verbose output"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate format
if [[ ! "$FORMAT" =~ ^(markdown|xml|json)$ ]]; then
    echo "Error: Unknown format '$FORMAT'. Use: markdown, xml, or json"
    exit 1
fi

# Ensure report directory exists
mkdir -p "${REPORT_DIR}"

echo "========================================"
echo "pykwavers vs k-wave-python Validation"
echo "========================================"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "Format: $FORMAT"
echo "Output: $REPORT_DIR"
echo ""

# Set Rust environment
export RUST_BACKTRACE=1
export RUST_LOG="${RUST_LOG:-info}"

# Enable Python validation if requested
if [ "$RUN_PYTHON" = true ]; then
    export KWAVERS_RUN_PYTHON=1
    echo "Python validation: ENABLED"
else
    export KWAVERS_RUN_PYTHON=0
    echo "Python validation: DISABLED (use --python to enable)"
fi

# Run the validation suite
echo ""
echo "Running validation suite..."
echo "----------------------------------------"

cd "${PROJECT_ROOT}"

# Build test binary first (for better error messages)
if [ "$VERBOSE" = true ]; then
    echo "Building test binary..."
    cargo test --test python_validation_integration_test --no-run 2>&1 | tee "${REPORT_DIR}/build.log"
fi

# Run tests
TEST_OUTPUT="${REPORT_DIR}/test_output_${TIMESTAMP}.txt"
echo "Test output: ${TEST_OUTPUT}"

# Run with stdout capture
if cargo test --test python_validation_integration_test -- --nocapture --test-threads=1 > "${TEST_OUTPUT}" 2>&1; then
    TEST_STATUS=0
    echo "✅ Validation suite completed successfully"
else
    TEST_STATUS=1
    echo "⚠️  Validation suite completed with failures (exit code: $TEST_STATUS)"
fi

# Parse test output and generate report
echo ""
echo "Generating report..."

# Determine file extension
case "$FORMAT" in
    markdown)
        EXT="md"
        ;;
    xml)
        EXT="xml"
        ;;
    json)
        EXT="json"
        ;;
esac

REPORT_FILE="${REPORT_DIR}/validation_report_${TIMESTAMP}.${EXT}"
LINK_FILE="${REPORT_DIR}/validation_report_latest.${EXT}"

# Generate report based on format
case "$FORMAT" in
    markdown)
        # Generate markdown report header
        cat > "${REPORT_FILE}" << 'EOF'
# pykwavers vs k-wave-python Validation Report

Generated: TIMESTAMP
Format: markdown
Python Validation: PYTHON_STATUS

## Summary

EOF
        # Replace placeholders
        sed -i.bak "s/TIMESTAMP/$(date -u +"%Y-%m-%d %H:%M:%S UTC")/g" "${REPORT_FILE}"
        if [ "$RUN_PYTHON" = true ]; then
            sed -i.bak "s/PYTHON_STATUS/Enabled/g" "${REPORT_FILE}"
        else
            sed -i.bak "s/PYTHON_STATUS/Disabled/g" "${REPORT_FILE}"
        fi

        # Append test output
        echo "" >> "${REPORT_FILE}"
        echo "## Test Results" >> "${REPORT_FILE}"
        echo "" >> "${REPORT_FILE}"
        echo '```' >> "${REPORT_FILE}"
        cat "${TEST_OUTPUT}" >> "${REPORT_FILE}"
        echo '```' >> "${REPORT_FILE}"

        # Parse results section
        if grep -q "Validation Summary" "${TEST_OUTPUT}"; then
            echo "" >> "${REPORT_FILE}"
            echo "## Results by Component" >> "${REPORT_FILE}"
            echo "" >> "${REPORT_FILE}"
            grep -A 20 "=== Validation Summary ===" "${TEST_OUTPUT}" | grep -E "(Grid|Signal|Source|Sensor|Solver|Total|Max Error)" >> "${REPORT_FILE}" 2>/dev/null || true
        fi

        # Add footer
        echo "" >> "${REPORT_FILE}"
        echo "---" >> "${REPORT_FILE}"
        echo "*Report generated by generate_report.sh*" >> "${REPORT_FILE}"
        ;;

    xml)
        # Generate JUnit-style XML report
        cat > "${REPORT_FILE}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites timestamp="$(date -u +"%Y-%m-%dT%H:%M:%S")" name="pykwavers_validation">
  <testsuite name="python_validation" tests="0" failures="0" errors="0" skipped="0">
EOF

        # Parse test results from output
        # Extract test names and results
        grep -E "(^test_|PASSED|FAILED)" "${TEST_OUTPUT}" | while read -r line; do
            if [[ "$line" =~ ^test_ ]]; then
                TEST_NAME="${line%:}"
                if grep -q "FAILED" <<< "$line"; then
                    FAILED="true"
                else
                    FAILED="false"
                fi

                cat >> "${REPORT_FILE}" << EOF
    <testcase classname="validation.python" name="${TEST_NAME}" time="0.0">
EOF
                if [ "$FAILED" = "true" ]; then
                    cat >> "${REPORT_FILE}" << EOF
      <failure message="Test failed"></failure>
EOF
                fi
                echo "    </testcase>" >> "${REPORT_FILE}"
            fi
        done

        cat >> "${REPORT_FILE}" << EOF
  </testsuite>
</testsuites>
EOF
        ;;

    json)
        # Generate JSON report
        cat > "${REPORT_FILE}" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "format": "json",
  "python_validation": $(if [ "$RUN_PYTHON" = true ]; then echo "true"; else echo "false"; fi),
  "results": {
    "output_file": "${TEST_OUTPUT}",
    "status": $(if [ $TEST_STATUS -eq 0 ]; then echo "\"passed\""; else echo "\"failed\""; fi)
  }
}
EOF
        ;;
esac

# Remove backup file created by sed
rm -f "${REPORT_FILE}.bak"

# Create symlink to latest report
ln -sf "${REPORT_FILE}" "${LINK_FILE}"

echo ""
echo "========================================"
echo "Report Generation Complete"
echo "========================================"
echo "Report: ${REPORT_FILE}"
echo "Latest: ${LINK_FILE}"
echo ""

# Show summary
if [ "$FORMAT" = "markdown" ] && [ -f "${REPORT_FILE}" ]; then
    echo "--- Report Summary ---"
    head -50 "${REPORT_FILE}"
    echo "---"
fi

# Exit with test status
exit $TEST_STATUS
