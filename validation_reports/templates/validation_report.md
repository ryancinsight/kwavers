# Validation Report: {{report_name}}
**k-Wave Python (k-wave-python) vs kwavers/pykwavers Parity Validation**

---

## Executive Summary

- **Report ID**: {{report_id}}
- **Generation Date**: {{timestamp}}
- **k-wave-python Version**: {{kwave_python_version}}
- **pykwavers Version**: {{pykwavers_version}}
- **kwavers Version**: {{kwavers_version}}
- **Git Commit**: {{git_commit}}

### Test Statistics
- Total Tests: {{total_tests}}
- Passed: {{passed_tests}}
- Failed: {{failed_tests}}
- Skipped: {{skipped_tests}}
- Pass Rate: {{pass_rate}}%
- Overall Status: {{#if overall_pass}}✅ PASS{{else}}❌ FAIL{{/if}}

### Error Summary
- Mean L2 Error: {{mean_l2_error}}
- Max L∞ Error: {{max_linf_error}}
- Mean Relative Error: {{mean_relative_error}}%

---

## Component Validation

{{#each components}}

### {{component_name}} Validation

{{#each tests}}
#### {{test_name}}

**Test ID**: `{{test_id}}`

**Description**: {{description}}

**Configuration**:
- Grid Size: {{grid_size}}
- Medium: {{medium_type}}
- Solver: {{solver_type}}
- Time Steps: {{time_steps}}

**Error Metrics**:
| Metric | Value | Tolerance | Status |
|--------|-------|-----------|--------|
| L2 (RMS) | {{l2_error}}% | {{l2_tolerance}}% | {{#if l2_pass}}✅{{else}}❌{{/if}} |
| L∞ (Max) | {{linf_error}}% | {{linf_tolerance}}% | {{#if linf_pass}}✅{{else}}❌{{/if}} |
| Relative | {{relative_error}}% | {{relative_tolerance}}% | {{#if relative_pass}}✅{{else}}❌{{/if}} |
| Peak Pressure | {{peak_pressure_ratio}} | ±{{peak_tolerance}} | {{#if peak_pass}}✅{{else}}❌{{/if}} |

**Status**: {{#if test_pass}}✅ PASS{{else}}❌ FAIL{{/if}}

---

{{/each}}

{{/each}}

## Detailed Results

### Complete Test Log

| Component | Test Name | L2 Error | L∞ Error | Relative | Peak | Status |
|-----------|-----------|----------|----------|----------|------|--------|
{{#each all_tests}}
| {{component}} | {{name}} | {{l2}}% | {{linf}}% | {{rel}}% | {{peak}} | {{#if pass}}✅{{else}}❌{{/if}} |
{{/each}}

### Error Distribution

#### L2 Error Distribution
- Mean: {{stats.l2.mean}}%
- Std Dev: {{stats.l2.std}}%
- Min: {{stats.l2.min}}%
- Max: {{stats.l2.max}}%
- Median: {{stats.l2.median}}%

#### L∞ Error Distribution
- Mean: {{stats.linf.mean}}%
- Std Dev: {{stats.linf.std}}%
- Min: {{stats.linf.min}}%
- Max: {{stats.linf.max}}%
- Median: {{stats.linf.median}}%

### Failure Analysis

{{#each failures}}

#### ⚠️ Failed: {{component}} - {{test_name}}

**Test ID**: {{test_id}}

**Failure Reason**: {{failure_reason}}

**Expected vs Actual**:
- Expected: {{expected_value}}
- Actual: {{actual_value}}

**Diagnostic Data**:
```json
{{diagnostic_json}}
```

---

{{else}}
*All tests passed. No failures to report.*
{{/each}}

## Solver-Specific Validation

### PSTD (Pseudo-Spectral Time Domain)

| Test Case | Spatial Order | Temporal Order | Grid Points | CFL | Pass |
|-----------|---------------|----------------|-------------|-----|------|
{{#each pstd_tests}}
| {{name}} | {{spatial_order}} | {{temporal_order}} | {{grid_points}} | {{cfl}} | {{#if pass}}✅{{else}}❌{{/if}} |
{{/each}}

### k-Space Corrections

| Correction Type | Energy Conservation | Phase Accuracy | Pass |
|-----------------|---------------------|----------------|------|
{{#each kspace_tests}}
| {{type}} | {{energy_conservation}} | {{phase_accuracy}} | {{#if pass}}✅{{else}}❌{{/if}} |
{{/each}}

## Comparison with Reference Data

### Against k-wave-python

| Metric | kwavers Result | k-wave-python Result | Difference | Within Tolerance |
|--------|----------------|---------------------|------------|------------------|
{{#each reference_comparisons}}
| {{metric}} | {{kwavers_value}} | {{kwave_value}} | {{diff}} | {{#if within_tol}}✅{{else}}❌{{/if}} |
{{/each}}

## System Information

- **Platform**: {{platform}}
- **Architecture**: {{architecture}}
- **Rust Version**: {{rust_version}}
- **Python Version**: {{python_version}}
- **NumPy Version**: {{numpy_version}}
- **SciPy Version**: {{scipy_version}}
- **GPU Available**: {{gpu_available}}
- **GPU Model**: {{gpu_model}}

## Execution Details

- **Start Time**: {{start_time}}
- **End Time**: {{end_time}}
- **Duration**: {{duration}} seconds
- **Parallel Workers**: {{workers}}
- **Test Harness**: {{test_harness}}

## Conclusion

{{#if overall_pass}}
## ✅ Overall Assessment: PARITY ACHIEVED

The validation suite confirms that kwavers achieves numerical parity with k-wave-python 
reference implementations across all tested components. The implementation is ready 
for production use.
{{else}}
## ❌ Overall Assessment: PARITY NOT ACHIEVED

The validation suite identified {{failed_tests}} test(s) failing to meet the required 
tolerance thresholds. Remediation is required before declaring parity with k-wave-python.

### Priority Fixes Required
{{#each critical_failures}}
1. **{{component}}**: {{test_name}} - {{failure_reason}}
{{/each}}
{{/if}}

### Recommendations

{{#each recommendations}}
- {{text}}
{{else}}
- All systems operating within expected parameters. No immediate action required.
{{/each}}

---

## Appendix

### A. Raw Error Data

<details>
<summary>Click to expand raw JSON metrics</summary>

```json
{{raw_metrics_json}}
```

</details>

### B. Configuration File

<details>
<summary>Click to expand validation configuration</summary>

```yaml
{{config_yaml}}
```

</details>

### C. Test Commands Used

```bash
# Cargo test execution
cargo test --test validation_suite -- --nocapture --report-time

# Python reference execution
python -m pytest tests/test_validation_suite.py -v --tb=short
```

---

*Generated by kwavers Validation Report Generator v{{generator_version}}*  
*Report Template Version: 1.0.0*