# pykwavers vs k-wave-python Validation Summary

## Date: 2026-02-06
## Sprint: 218 Session 2

---

## Executive Summary

Validation testing between pykwavers and k-wave-python reveals **significant discrepancies** that require investigation and fixes. While pykwavers' FDTD and PSTD solvers show excellent agreement (99% correlation), both differ substantially from k-wave-python's k-space PSTD implementation.

### Key Findings

| Comparison | L2 Error | L∞ Error | Correlation | Status |
|------------|----------|----------|-------------|--------|
| pykwavers FDTD vs PSTD | 0.90 | 0.90 | 0.994 | [OK] Internal consistency |
| pykwavers FDTD vs k-wave | 3.90 | 2.10 | 0.186 | [X] **MISMATCH** |
| pykwavers PSTD vs k-wave | 38.1 | 20.8 | 0.117 | [X] **MISMATCH** |

**Acceptance Criteria:**
- L2 error < 0.01 (1%)
- L∞ error < 0.05 (5%)
- Correlation > 0.95

---

## Test Configuration

```python
Grid: (64, 64, 64)
Spacing: 0.10 mm
Frequency: 1.0 MHz
Wavelength: 1.50 mm
Points per wavelength: 15.0
Time steps: 500
Medium: Water (c=1500 m/s, ρ=1000 kg/m³)
Source: Plane wave at z=0, amplitude=1e5 Pa
Sensor: Point sensor at center (3.2, 3.2, 3.2) mm
PML: 10 points
```

---

## Analysis

### 1. Internal Consistency [PASS]

pykwavers FDTD and PSTD solvers show 99.4% correlation, indicating:
- Core wave propagation physics is consistent
- Grid and time-stepping are aligned
- Source injection is consistent between solvers

### 2. k-wave-python Discrepancy [FAIL]

Both pykwavers solvers differ significantly from k-wave-python:
- Low correlation (~0.12-0.19)
- High L2 error (3.9-38.1)
- Different amplitude scales observed

**Hypothesized Causes:**

1. **Source Injection Method**
   - k-wave-python uses k-space distributed source (BLI)
   - pykwavers uses direct grid source injection
   - Source amplitude normalization may differ

2. **Source Signal Definition**
   - k-wave: `p = amplitude * sin(2*pi*f*t)`
   - pykwavers: Similar but may have different scaling
   - Time array alignment may differ

3. **PML Implementation**
   - Different PML formulations (CPML vs standard PML)
   - PML inside vs outside domain
   - Reflection coefficients may differ

4. **Temporal Integration**
   - k-wave: 4th-order Runge-Kutta or predictor-corrector
   - pykwavers: Explicit Euler with CFL condition
   - Different stability criteria

5. **Sensor Recording**
   - k-wave: Records at specific grid points with interpolation
   - pykwavers: Direct grid point recording
   - Time array alignment may differ

---

## Recommendations

### Immediate Actions (P0)

1. **Source Injection Audit**
   - Compare source mask handling between implementations
   - Verify source signal normalization
   - Check amplitude scaling factors

2. **Time Array Alignment**
   - Ensure identical time discretization
   - Verify CFL condition calculations
   - Compare temporal integration schemes

3. **Sensor Recording Verification**
   - Standardize sensor position interpolation
   - Align time array indexing
   - Verify record_start_index handling

### Short-term Actions (P1)

4. **Implement Source Modes**
   - Add "mass source" mode (particle velocity source)
   - Add "pressure source" mode (Dirichlet BC)
   - Match k-wave's source injection exactly

5. **PML Alignment**
   - Implement k-wave's CPML formulation
   - Match PML inside/outside behavior
   - Standardize alpha parameter

6. **Temporal Integration**
   - Implement 4th-order Runge-Kutta option
   - Match k-wave's predictor-corrector scheme
   - Add time-staggered grid option

### Medium-term Actions (P2)

7. **Enhanced Comparison Framework**
   - Add source mode parameter to comparison config
   - Implement automatic source scaling calibration
   - Add spectral analysis to validation metrics

8. **Documentation**
   - Document known differences from k-wave
   - Provide migration guide for k-wave users
   - Add validation test suite to CI/CD

---

## Next Steps

1. Fix source injection mismatch (priority: P0)
2. Re-run validation tests
3. If L2 error < 0.1, consider acceptable for beta release
4. Target L2 error < 0.01 for production release

---

## Test Commands

```bash
# Run validation tests
cd pykwavers
.venv/Scripts/python -m pytest test_kwave_comparison.py -v

# Run analysis script
.venv/Scripts/python validation_analysis.py

# Run with different solver types
.venv/Scripts/python -c "
from pykwavers.comparison import run_comparison, SimulationConfig, SimulatorType
config = SimulationConfig(...)
comparison = run_comparison(config, [SimulatorType.PYKWAVERS_PSTD, SimulatorType.KWAVE_PYTHON])
print(comparison.validation_report)
"
```

---

## References

1. Treeby & Cox (2010) - k-Wave paper
2. k-wave-python documentation: https://github.com/waltsims/k-wave-python
3. pykwavers architecture docs: ../kwavers/ARCHITECTURE.md
4. PSTD source injection fix: SESSION_3_FDTD_SOURCE_INJECTION_RESOLUTION.md

---

**Status:** Validation framework operational, discrepancies identified, fixes required.
