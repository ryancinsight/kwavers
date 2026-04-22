# k-Wave Python Parity Validation Reports

This directory contains the validation report infrastructure for documenting parity between kwavers/pykwavers and the k-wave-python reference implementation.

## Overview

The validation report system provides automated quantitative tracking of numerical parity between implementations, with support for multiple output formats:

- **Markdown** - Human-readable reports with detailed error analysis
- **JSON** - Machine-readable metrics for programmatic consumption
- **XML** - JUnit-style format for CI/CD integration

## Directory Structure

```
validation_reports/
├── templates/              # Report templates
│   ├── validation_report.md    # Markdown report template
│   ├── validation_metrics.json # JSON metrics template
│   └── ci_report.xml           # JUnit XML template
├── output/                 # Generated reports (timestamped)
├── generate_report.py      # Report generation script
└── README.md              # This file
```

## Usage

### Generate All Reports

```bash
cd validation_reports
python generate_report.py --format all --output ./output
```

### Generate Specific Format

```bash
# Markdown only
python generate_report.py --format markdown

# JSON only
python generate_report.py --format json

# XML only (for CI/CD)
python generate_report.py --format xml
```

### Run Tests and Generate Report

```bash
# Run cargo tests before generating report
python generate_report.py --run-tests --format all
```

### Filter Specific Tests

```bash
# Generate report only for PSTD tests
python generate_report.py --run-tests --test-filter test_pstd
```

### Use Existing Test Output

```bash
# Parse existing cargo test output
python generate_report.py --cargo-output ./output/cargo_test_output.txt
```

## Report Contents

Each validation report includes:

### Metadata
- Report ID and timestamp
- Version information (kwavers, pykwavers, k-wave-python)
- Git commit hash
- System information (platform, architecture, compiler versions)

### Summary
- Total test count
- Pass/fail statistics
- Pass rate percentage
- Execution duration

### Component Validation
Results grouped by component:
- **Grid** - Grid dimension and spacing tests
- **Source** - Source injection and beam pattern tests
- **Signal** - Signal waveform generation tests
- **Sensor** - Sensor response and recording tests
- **Solver** - PSTD/FMM solver accuracy tests

### Error Metrics
For each test, reports include:
| Metric | Description | Tolerance |
|--------|-------------|-----------|
| L2 (RMS) | Root mean square error | < 2% |
| L∞ (Max) | Maximum absolute error | < 5% |
| Relative | Relative percentage error | < 2% |
| Peak Ratio | Peak pressure deviation | ±5% |

### Tolerance Thresholds

Default tolerances are defined for numerical parity validation:

```python
L2_ERROR_THRESHOLD = 0.02      # 2% RMS error
LINF_ERROR_THRESHOLD = 0.05    # 5% maximum error
RELATIVE_ERROR_THRESHOLD = 0.02 # 2% relative error
PEAK_DEVIATION_THRESHOLD = 0.05 # 5% peak pressure deviation
```

These thresholds ensure numerical parity within discretization uncertainties.

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Validation Tests
  run: |
    cargo test --test validation_suite -- --nocapture 2>&1 | tee cargo_output.txt
    python validation_reports/generate_report.py \
      --format xml \
      --cargo-output cargo_output.txt \
      --output ./validation_reports/output

- name: Publish Test Results
  uses: actions/upload-artifact@v3
  with:
    name: validation-report
    path: validation_reports/output/*.xml
```

### JUnit XML Format

The XML output follows JUnit schema for compatibility with:
- Jenkins JUnit plugin
- GitLab CI test reports
- GitHub Actions test annotations
- Azure DevOps test reporting

## Error Patterns

The parser extracts error metrics from cargo test output using these patterns:

- `L2 Error: <value>` - RMS error relative to reference
- `L∞ Error: <value>` - Maximum pointwise error
- `Relative Error: <value>` - Percentage error
- `Peak Ratio: <value>` - Pressure amplitude ratio

Example cargo test output:
```
test test_pstd_vs_dalembert_1d_homogeneous ... ok
test result: passed. 1 passed; 0 failed
L2 Error: 0.0085
L∞ Error: 0.023
Relative Error: 0.85%
Peak Ratio: 0.987
```

## Mathematical Basis

Validation tests employ rigorous analytical solutions:

### d'Alembert Solution (1D Wave Equation)
For ∂²p/∂t² - c²∇²p = 0 with initial p(x,0) = p₀(x):
```
p(x,t) = ½[p₀(x - ct) + p₀(x + ct)]
```

Error metrics are computed against this exact solution, ensuring:
- Numerical convergence to analytical reference
- Phase accuracy preservation
- Energy conservation properties

### Spectral Accuracy
PSTD methods achieve spectral convergence:
- Exponential decay of truncation error
- Minimal numerical dispersion
- Minimal numerical dissipation

## Troubleshooting

### No test results parsed

Ensure cargo test output format includes test names and results:
```bash
cargo test --test test_pstd_kwave_comparison -- --nocapture
```

### Missing error metrics

Add explicit error metric output to tests:
```rust
println!("L2 Error: {:.6}", l2_error);
println!("L∞ Error: {:.6}", linf_error);
```

### Format version incompatibility

Check `SCHEMA_VERSION` constant in `generate_report.py` matches template versions.

## Contributing

When adding new validation tests:

1. Follow naming convention: `<component>_<specific_validation>`
2. Output error metrics in parseable format
3. Document analytical reference solution
4. Include tolerance justification in comments

## References

- Treeby & Cox, "Modeling ultrasound propagation using the k-space pseudospectral method," J. Acoust. Soc. Am. 127(6), 2010
- d'Alembert, Jean le Rond, 1746, "Recherches sur la courbe que forme une corde tendue mise en vibration"
- k-Wave User Manual: http://www.k-wave.org/

## License

This validation report infrastructure is part of kwavers and follows the same license terms.