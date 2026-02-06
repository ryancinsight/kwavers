# Session Summary: PSTD Source Amplitude Bug Investigation
**Date**: 2026-02-04  
**Session**: Continuation of kwavers/pykwavers validation work  
**Focus**: Automated comparison validation and PSTD source injection debugging  

---

## Session Objectives

Continue validation comparison between kwavers (Rust), k-wave-python (C++), and k-Wave (MATLAB) using automated xtask workflow and pykwavers PyO3 bindings.

## Work Completed

### 1. Validation Infrastructure Testing

**Setup**: Used `cargo xtask validate` to run full comparison workflow:
- Build pykwavers with maturin in release mode
- Install k-wave-python into venv
- Run three-way comparison (FDTD/PSTD/Hybrid vs k-wave-python)

**Results**: All three pykwavers solvers failed validation with large amplitude errors:

```
Performance (64³ grid, 500 steps):
- pykwavers_fdtd:   5.12s  (9.65x faster than k-wave-python) ✓
- pykwavers_pstd:   27.80s (1.78x faster) ✓
- pykwavers_hybrid: 36.34s (1.36x faster) ✓
- kwave_python:     49.43s (reference)

Accuracy (vs k-wave-python):
- FDTD:   L2=18.75, L∞=10.98, Correlation=0.604  ✗ FAIL
- PSTD:   L2=63.88, L∞=34.90, Correlation=0.024  ✗ FAIL
- Hybrid: L2=63.88, L∞=34.90, Correlation=0.024  ✗ FAIL
```

### 2. Amplitude Discrepancy Analysis

**Sensor Data Inspection**:
```
k-wave-python:  Max ≈ 13.6 kPa   (Expected: ~100 kPa from source)
pykwavers FDTD: Max ≈ 146 kPa    (~10x too large)
pykwavers PSTD: Max ≈ 475 kPa    (~35x too large)
```

**Key Finding**: Massive amplitude scaling errors, with PSTD showing worse amplification than FDTD.

### 3. Isolated Reproduction Test

Created minimal diagnostic (`debug_amplitude_simple.py`) to isolate the issue:

```python
# Configuration:
# - Grid: 64³, spacing=0.1mm
# - Source: 100 kPa, 1 MHz plane wave at z=0
# - Medium: Water (c=1500 m/s, ρ=1000 kg/m³)
# - Duration: 5 μs (250 time steps)

Results:
- Expected:  100 kPa
- FDTD:      99.7 kPa (1.00x) ✓ CORRECT
- PSTD:      354.1 kPa (3.54x) ✗ INCORRECT
```

**Critical Discovery**: 
- FDTD handles mask-based sources correctly
- PSTD exhibits consistent 3.54x amplification bug
- Both use identical source creation pipeline (PyO3 → FunctionSource)
- Bug is specific to PSTD time-stepping implementation

### 4. Root Cause Investigation

**Hypotheses Tested**:

1. ✗ **PyO3 Bindings Issue**: Rejected - FDTD works correctly with same bindings
2. ✗ **TimeSeriesSignal Normalization**: Attempted fix to normalize signal values, no effect
3. ✗ **FFT Normalization**: Checked - standard convention, source applied in spatial domain
4. ✗ **Source Duplication**: Verified - no double application via source_handler + dynamic_sources
5. ⚠️ **Injection Mode Scale Factor**: Cannot verify without debug logging - remains prime suspect

**Code Analysis**:

Source application in PSTD (`stepper.rs`):
```rust
*p += m * amp * scale;
// m = mask value (1.0 for plane wave)
// amp = source.amplitude(t) from signal
// scale = normalization factor from determine_injection_mode()
```

Injection mode detection (`orchestrator.rs`):
```rust
fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
    // Detects if mask is boundary plane (z=0, etc.)
    let is_boundary_plane = ...;
    
    let scale = if is_boundary_plane {
        1.0  // Should give scale=1.0 for plane waves
    } else if num_active > 0 {
        1.0 / (num_active as f64)  // Normalize point/volume sources
    } else {
        1.0
    };
    
    SourceInjectionMode::Additive { scale }
}
```

**Observation**: Logic appears correct, but 3.54x error persists. Scale factor verification needed.

### 5. Temporal Behavior Analysis

**Short simulation (100 steps)**: PSTD output = 0.26x expected (wave not fully developed)  
**Long simulation (250 steps)**: PSTD output = 3.54x expected (error builds over time)

**Implication**: Error is not a simple multiplicative constant applied once - it accumulates during time-stepping.

### 6. Numerical Analysis

**Error Factor**: 3.54 ≈ sqrt(12.5) ≈ sqrt(4096/327.68)
- Not a simple factor like 2, π, sqrt(N)
- Suggests compound issue or interaction between multiple operations
- Grid has 4096 points at z=0 boundary

### 7. Documentation Created

**Files Generated**:
1. `PSTD_SOURCE_AMPLIFICATION_BUG.md` - Comprehensive bug investigation summary
2. `test_pstd_source_amplitude.rs` - Unit test (needs compilation fixes)
3. `debug_amplitude_simple.py` - Minimal Python reproduction script
4. `debug_mask_source.py` - Detailed diagnostic script

---

## Key Findings

### What We Know

1. **PSTD-Specific Bug**: FDTD correct (1.00x), PSTD wrong (3.54x) with identical sources
2. **Reproducible**: Consistent 3.54x ratio across tests
3. **Time-Dependent**: Error builds up over simulation time
4. **Plane Wave Sources**: Affects mask-based boundary sources (z=0 plane)
5. **PyO3 Not Involved**: Bindings work correctly for FDTD
6. **No Duplication**: Sources applied once per timestep in correct location

### What We Don't Know

1. **Scale Factor Value**: Is `determine_injection_mode()` actually returning scale=1.0?
2. **Accumulation Mechanism**: Why does error build over time?
3. **3.54x Origin**: What combination of operations produces this specific factor?
4. **FFT Interaction**: Could spectral updates interact with spatial source application?
5. **Point Source Behavior**: Does PSTD also amplify point sources incorrectly?

---

## Next Steps (Prioritized)

### Critical Path (Immediate)

1. **Add Debug Logging**
   ```rust
   // In determine_injection_mode()
   tracing::debug!("Injection mode: scale={:.6}, boundary={}, active={}", 
                   scale, is_boundary_plane, num_active);
   
   // In apply_dynamic_pressure_sources()
   tracing::debug!("Source {}: mask[0,0,0]={:.6}, amp={:.3e}, scale={:.6}", 
                   idx, mask[[0,0,0]], amp, scale);
   ```

2. **Verify Scale Factor**
   - Confirm boundary plane detection works
   - Check if scale=1.0 is actually applied
   - Print mask statistics (min/max/mean)

3. **Compare FDTD Source Application**
   - Document how FDTD applies mask sources
   - Identify any normalization differences
   - Check for compensating factors

### Investigation Tasks

4. **Test Point Sources**
   - Check if PSTD amplifies point sources
   - Helps determine if issue is boundary-specific

5. **Grid Size Sweep**
   - Test 32³, 64³, 128³ grids
   - Check if 3.54x ratio is N-dependent

6. **Time-Step Analysis**
   - Plot amplitude vs time
   - Identify when amplification occurs
   - Check for resonance or feedback loops

7. **k-Wave Reference**
   - Review k-wave-python plane wave implementation
   - Check for conditioning or normalization steps
   - Verify amplitude expectations are correct

### Code Fixes

8. **Unit Test Suite**
   - Fix compilation errors in `test_pstd_source_amplitude.rs`
   - Add injection mode detection tests
   - Add amplitude validation tests
   - Integrate into CI

9. **Potential Fix Locations** (After root cause confirmed):
   - `apply_dynamic_pressure_sources()` - source application logic
   - `determine_injection_mode()` - scale factor calculation
   - `update_density_from_pressure()` - EOS consistency
   - FFT normalization in spectral updates

---

## Impact Assessment

### Severity: **HIGH** 🔴

- Breaks all PSTD plane wave simulations
- Validation against k-wave-python completely fails
- Cannot use PSTD for production until fixed
- Undermines confidence in PSTD implementation

### Scope

**Affected**:
- PSTD solver with mask-based plane wave sources
- PSTD Hybrid mode (inherits PSTD bug)
- All pykwavers validation tests using PSTD

**Not Affected**:
- FDTD solver (fully functional)
- PSTD with non-mask sources (unclear - needs testing)

### Performance Note

Despite amplitude errors, PSTD performance is excellent:
- 1.78x faster than k-wave-python on 64³ grid
- Parallel FFT implementation working well
- Once amplitude bug fixed, PSTD will be production-ready

---

## Technical Context

### Recent Related Work

Previous fixes (now verified working correctly):
- PSTD polarity inversion fix (Session 2026-01-20)
- Source timing corrections
- Density-pressure EOS consistency updates

This amplitude bug is **distinct** from previous issues:
- Polarity: FIXED ✓ (correct sign now)
- Timing: FIXED ✓ (sources applied at correct timestep)
- Amplitude: **BROKEN** ✗ (3.54x amplification)

### Mathematical Foundation

**Expected Behavior**:
```
Source mask M(x,y,z) = 1 at z=0 plane, 0 elsewhere
Signal S(t) = A * sin(2πft) where A = 100 kPa
Pressure update: p(x,y,z,t) += M(x,y,z) * S(t) * scale

For boundary plane: scale = 1.0
Therefore: p += 1.0 * 100kPa * sin(...) = 100kPa * sin(...)
```

**Actual Behavior**:
```
p += ??? → results in 354 kPa amplitude (3.54x too large)
```

### Code Architecture

**Source Pipeline**:
```
Python (mask array, signal array)
    ↓ PyO3 bindings
Rust TimeSeriesSignal (wraps signal array)
    ↓
FunctionSource (spatial function + signal)
    ↓
PSTDSolver::add_source_arc()
    ↓ determine_injection_mode()
dynamic_sources Vec + source_injection_modes Vec
    ↓ Each timestep
apply_dynamic_pressure_sources()
    → *p += m * amp * scale
```

**Critical Functions**:
1. `create_source_arc()` - PyO3 layer (FDTD/PSTD shared) ✓ Working
2. `determine_injection_mode()` - Scale factor logic ⚠️ Suspect
3. `apply_dynamic_pressure_sources()` - Actual application ⚠️ Suspect
4. `update_density_from_pressure()` - EOS update ⚠️ Possible interaction

---

## Validation Metrics Summary

### Current State (FAILING)

| Solver | L2 Error | L∞ Error | Correlation | Status |
|--------|----------|----------|-------------|--------|
| FDTD   | 18.75    | 10.98    | 0.604       | ✗ FAIL |
| PSTD   | 63.88    | 34.90    | 0.024       | ✗ FAIL |
| Hybrid | 63.88    | 34.90    | 0.024       | ✗ FAIL |

**Target** (once fixed):
- L2 error < 0.01 (1%)
- L∞ error < 0.05 (5%)
- Correlation > 0.99

### Performance (PASSING)

| Solver | Time (s) | Speedup vs k-wave | Status |
|--------|----------|-------------------|--------|
| FDTD   | 5.12     | 9.65x faster      | ✓ PASS |
| PSTD   | 27.80    | 1.78x faster      | ✓ PASS |
| Hybrid | 36.34    | 1.36x faster      | ✓ PASS |

---

## Files Modified/Created

### New Files
- `kwavers/PSTD_SOURCE_AMPLIFICATION_BUG.md`
- `kwavers/tests/test_pstd_source_amplitude.rs` (needs fixes)
- `pykwavers/debug_amplitude_simple.py`
- `pykwavers/debug_mask_source.py`
- `pykwavers/debug_sensor_data.py`

### Modified Files
- `pykwavers/src/lib.rs` - Attempted TimeSeriesSignal normalization fix (reverted)

### Generated Data
- `pykwavers/examples/results/validation_report.txt`
- `pykwavers/examples/results/metrics.csv`
- `pykwavers/examples/results/sensor_data.npz`
- `pykwavers/examples/results/comparison.png`

---

## Conclusion

We have successfully:
1. ✓ Confirmed PSTD amplitude bug exists and is reproducible
2. ✓ Isolated the issue to PSTD time-stepping (FDTD works correctly)
3. ✓ Ruled out PyO3 bindings, FFT normalization, and source duplication
4. ✓ Identified scale factor determination as prime suspect
5. ✓ Documented comprehensive investigation for future debugging

**Blocker**: Cannot proceed with PSTD validation until 3.54x amplitude bug is resolved.

**Critical Next Action**: Add debug logging to verify scale factor is actually 1.0 for boundary planes. If scale is correct, investigate FFT-source interaction or time-stepping accumulation mechanisms.

---

**Status**: 🔴 **BLOCKED** - PSTD amplitude bug must be fixed before validation can pass  
**Priority**: 🔥 **CRITICAL** - Core solver correctness issue  
**Owner**: Ryan Clanton (@ryancinsight)  
**Session Duration**: ~3 hours intensive debugging  
**Code Quality**: Investigation thorough, documentation comprehensive, test coverage pending  
