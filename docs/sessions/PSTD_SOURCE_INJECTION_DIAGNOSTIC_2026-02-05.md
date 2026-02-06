# PSTD Source Injection Diagnostic Summary
**Date:** 2026-02-05  
**Author:** Ryan Clanton (@ryancinsight)  
**Context:** Sprint 217 - k-Wave Comparison & Validation via pykwavers

---

## Executive Summary

Added comprehensive tracing infrastructure to diagnose PSTD source injection behavior. 
**Key Finding:** Source injection mechanism is working correctly (`scale=1.0`, boundary plane 
correctly identified), but sensor readings show unexpected attenuation (~85% amplitude loss).

**Status:** Root cause identified as different from initial hypothesis. The ~3.54× amplification 
bug reported in previous session is NOT reproduced with current test configuration. Instead, we 
observe significant attenuation that requires investigation of:
1. PML boundary absorption effects
2. Sensor placement and wave propagation timing
3. Grid resolution and numerical dispersion

---

## Implementation: Tracing Infrastructure

### 1. Added Debug Logging to PSTD Core

**File:** `kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs`

```rust
fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
    // ... existing logic ...
    
    debug!(
        num_active,
        mask_sum,
        mask_min,
        mask_max,
        is_boundary_plane,
        scale,
        "PSTD source injection mode determined"
    );
    debug!(
        "  Mask geometry: all_same_i={}, all_same_j={}, all_same_k={}, first_i={:?}, first_j={:?}, first_k={:?}",
        all_same_i, all_same_j, all_same_k, first_i, first_j, first_k
    );
    
    SourceInjectionMode::Additive { scale }
}
```

**File:** `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs`

```rust
pub(crate) fn apply_dynamic_pressure_sources(&mut self, dt: f64) {
    let t = self.time_step_index as f64 * dt;
    let p_max_before = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));

    for (idx, (source, mask)) in self.dynamic_sources.iter().enumerate() {
        // ... mask statistics ...
        
        match mode {
            SourceInjectionMode::Additive { scale } => {
                debug!(
                    time_step = self.time_step_index,
                    source_idx = idx,
                    amp,
                    mask_active,
                    mask_sum,
                    mask_max,
                    scale,
                    contribution = amp * scale,
                    "PSTD applying additive pressure source"
                );
                // ... injection logic ...
            }
        }
    }

    let p_max_after = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if self.time_step_index % 10 == 0 || self.time_step_index < 5 {
        debug!(
            time_step = self.time_step_index,
            p_max_before,
            p_max_after,
            delta = p_max_after - p_max_before,
            "PSTD pressure field after source injection"
        );
    }
}
```

### 2. Added Tracing Initialization to pykwavers

**File:** `pykwavers/Cargo.toml`
- Added `kwavers` feature: `structured-logging`
- Added dependencies: `tracing`, `tracing-subscriber`

**File:** `pykwavers/src/lib.rs`

```rust
use std::sync::Once;
use tracing_subscriber::fmt;
use tracing_subscriber::EnvFilter;

static TRACING_INIT: Once = Once::new();

#[pyfunction]
fn init_tracing() -> PyResult<()> {
    TRACING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("kwavers=info"));

        fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .with_line_number(true)
            .init();
    });
    Ok(())
}

#[pymodule]
fn _pykwavers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing classes ...
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    Ok(())
}
```

**File:** `pykwavers/python/pykwavers/__init__.py`
- Exported `init_tracing` function to Python API

### 3. Diagnostic Test Script

**File:** `pykwavers/debug_pstd_trace.py`

```python
import os
os.environ["RUST_LOG"] = "kwavers=debug"
import pykwavers as kw

kw.init_tracing()

# 64³ grid, 0.1 mm spacing
# 1 MHz plane wave at z=0, 100 kPa amplitude
# PSTD solver, 50 time steps
# Sensor at center (3.2, 3.2, 3.2) mm
```

---

## Diagnostic Results

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Grid | 64×64×64 points |
| Spacing | 0.1 mm (dx=dy=dz) |
| Medium | Water (c=1500 m/s, ρ=1000 kg/m³) |
| Source | 1 MHz plane wave at z=0 |
| Amplitude | 100 kPa (100,000 Pa) |
| Solver | PSTD with default config |
| Time Steps | 50 (dt ≈ 34.6 ns, CFL=0.52) |
| Sensor | Point at (3.2, 3.2, 3.2) mm |

### Tracing Output Analysis

#### ✅ Source Injection Mode Determination (Correct)

```
DEBUG kwavers::solver::forward::pstd::implementation::core::orchestrator: 331: 
  PSTD source injection mode determined 
  num_active=4096 
  mask_sum=4096.0 
  mask_min=1.0 
  mask_max=1.0 
  is_boundary_plane=true 
  scale=1.0

DEBUG kwavers::solver::forward::pstd::implementation::core::orchestrator: 340: 
  Mask geometry: all_same_i=false, all_same_j=false, all_same_k=true, 
  first_i=Some(0), first_j=Some(0), first_k=Some(0)
```

**Interpretation:**
- ✅ Plane wave mask correctly identified as boundary plane
- ✅ All 4096 points (64×64) on z=0 plane are active
- ✅ Scale = 1.0 (no normalization) — correct for plane waves
- ✅ Mask geometry correctly shows k=0 for all active points

#### ✅ Source Application Per Time Step (Correct)

Sample from steps 1-10:

```
step=1:  amp=21594 Pa,  contribution=21594 Pa,  p_max_after=21594 Pa
step=2:  amp=42169 Pa,  contribution=42169 Pa,  p_max_after=47693 Pa
step=3:  amp=60755 Pa,  contribution=60755 Pa,  p_max_after=65745 Pa
step=4:  amp=76473 Pa,  contribution=76473 Pa,  p_max_after=78396 Pa
step=7:  amp=99889 Pa,  contribution=99889 Pa,  (peak of sine wave)
step=10: amp=82207 Pa,  contribution=82207 Pa,  p_max_after=109069 Pa
```

**Interpretation:**
- ✅ Source amplitude follows expected sine wave (peaks ~100 kPa)
- ✅ Contribution = amp × scale = amp × 1.0 (correct)
- ✅ Pressure field maxima reach ~109 kPa (reasonable for wave formation)
- ✅ No evidence of 3.54× amplification bug from previous context

#### ❌ Sensor Reading vs Field Maximum (Unexpected Attenuation)

```
Pressure field maxima during simulation: ~109 kPa
Sensor recorded pressure maximum:       14.8 kPa
Attenuation:                            ~85% loss
Expected sensor reading:                ~100 kPa
```

**Discrepancy:** Sensor reads only 14.8 kPa despite field maxima of 109 kPa.

---

## Root Cause Analysis

### Hypothesis 1: Sensor Placement & Wave Propagation Timing ⚠️ LIKELY

**Issue:** Sensor at (3.2, 3.2, 3.2) mm may be too close to source plane (z=0).

- Wave needs time to form and propagate
- Distance from source: 3.2 mm = 32 grid points
- At c=1500 m/s: arrival time ≈ 2.13 μs
- Simulation duration: 50 steps × 34.6 ns = 1.73 μs

**Conclusion:** Wave may not have reached sensor or is still forming.

**Evidence:**
- p_max in field reaches 109 kPa (wave exists)
- Sensor at center records only 14.8 kPa (wave hasn't arrived or is attenuated)

### Hypothesis 2: PML Absorption Effects ⚠️ POSSIBLE

**Issue:** Default PML boundaries (thickness=20 points) may be absorbing energy.

- PML extends 20 points from each boundary
- Sensor at (32, 32, 32) is inside active domain but near PML
- PML absorption could attenuate wave before reaching sensor

**Evidence:**
- Default PSTD config uses CPML boundaries
- No explicit PML-free region verification

### Hypothesis 3: Numerical Dispersion & Grid Resolution ⚠️ UNLIKELY

**Issue:** Grid resolution may be insufficient.

- Wavelength λ = c/f = 1500/1e6 = 1.5 mm = 15 grid points
- Rule of thumb: 10-15 PPW (points per wavelength) for PSTD
- Current: 15 PPW (borderline acceptable)

**Evidence:**
- Field maxima are correct (~109 kPa)
- PSTD is nearly dispersion-free for this resolution
- Issue is sensor reading, not field evolution

### Hypothesis 4: Sensor Recording Implementation 🔍 NEEDS VERIFICATION

**Issue:** Sensor may be recording at wrong location or interpolation is incorrect.

**Evidence:**
- Need to verify sensor mask creation
- Need to verify interpolation from grid to sensor point

---

## Conclusion: Amplification Bug CONFIRMED

**CRITICAL UPDATE (2026-02-05 03:52 UTC):** After running quick diagnostic with proper 
configuration (64³ grid, 8 μs duration, sensor at 60% of domain), the amplification bug 
is **CONFIRMED** and reproduced.

### Quick Diagnostic Results

**Test Configuration:**
- Grid: 64×64×64, spacing 0.1 mm
- Duration: 8 μs (692 time steps)
- Source: 1 MHz plane wave, 100 kPa amplitude
- Sensor: (3.2, 3.2, 3.8) mm (60% along z-axis)
- Expected arrival: 2.53 μs

**Results:**

| Metric | Expected | FDTD | PSTD | Status |
|--------|----------|------|------|--------|
| **Amplitude** | 100 kPa | 136.4 kPa (+36.4%) | **623.2 kPa (+523%)** | ❌ CRITICAL |
| **Arrival Time** | 2.53 μs | 1.47 μs (-42%) | 0.10 μs (-96%) | ❌ FAIL |
| **Amplification Factor** | 1.0× | 1.36× | **6.23×** | ❌ CRITICAL |

**Key Findings:**
1. ✅ Source injection mechanism is correct (scale=1.0, boundary plane identified)
2. ❌ PSTD produces **6.23× amplification** (623 kPa vs 100 kPa expected)
3. ❌ FDTD also shows 1.36× amplification (less severe but still present)
4. ❌ Both solvers show premature wave arrival (suggests numerical issues)

**Root Cause Hypothesis:**
The initial diagnostic (section above) showed low amplitude because:
1. Short simulation (50 steps) didn't allow full wave propagation
2. Sensor placement too close to PML boundaries
3. Different test configuration masked the amplification

The amplification bug exists and is **reproducible** with proper test setup.

---

## Next Steps

### Priority 1: Root Cause Analysis of Amplification ⚠️ CRITICAL

**Confirmed Issue:** PSTD amplifies plane wave sources by 6.23×, FDTD by 1.36×

**Investigation Required:**
1. **Step-by-step field evolution tracing**
   - Add detailed logging to `update_pressure()`, `update_density()`, `update_velocity()`
   - Track field maxima and energy at each substep
   - Identify which operation causes amplification

2. **Source injection timing and accumulation**
   - Verify source is applied only once per timestep
   - Check if `apply_dynamic_pressure_sources()` is called multiple times
   - Review interaction between source injection and spectral updates

3. **Spectral operator analysis**
   - Verify FFT normalization (forward vs inverse)
   - Check k-space operators (kappa, gradient operators)
   - Compare with k-Wave spectral operator implementation

4. **Density-pressure consistency**
   - After `apply_dynamic_pressure_sources()`, `update_density_from_pressure()` is called
   - Verify p = c²ρ relationship is maintained
   - Check if this creates feedback loop

### Priority 2: Compare with k-Wave Reference Implementation

1. Run identical test case in k-Wave (MATLAB or k-wave-python)
2. Compare source injection method (additive vs Dirichlet)
3. Review k-Wave source code for plane wave boundary handling
4. Verify spectral operator implementations match

### Priority 3: Minimal Reproducible Test Case

Create isolated test for source injection only:
1. Single time step with source injection
2. No spectral operators, no updates
3. Verify pressure field = source amplitude
4. Gradually add operations to identify culprit

### Priority 4: FDTD Amplification Investigation

FDTD also shows 1.36× amplification (less severe but still incorrect):
1. Review FDTD source injection code
2. Check if issue is in PyO3 binding layer (affects both solvers)
3. Verify plane wave source creation and mask generation

### Priority 5: Temporary Mitigation

Until root cause is fixed:
1. Add amplitude correction factor (divide by observed amplification)
2. Document the issue prominently in API docs
3. Add validation tests to CI to catch regressions

---

## Verification Test Specification

To properly validate PSTD source injection, create a test with:

```python
# Grid: Large enough for wave propagation
nx, ny, nz = 128, 128, 128
dx = dy = dz = 0.1e-3  # 0.1 mm

# Duration: Long enough for multiple wavelengths
duration = 20e-6  # 20 μs (13 wavelengths)

# Sensor: Far from source, outside PML
sensor_pos = (6.4e-3, 6.4e-3, 10e-3)  # (64, 64, 100) in grid coords

# Expected result:
# - Sensor should record ~100 kPa peak pressure
# - Arrival time: t ≈ 10 mm / 1500 m/s = 6.67 μs
# - Allow 3-4 periods for steady state
```

---

## Files Modified

### Core kwavers
- `kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs`
- `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs`

### pykwavers Bindings
- `pykwavers/Cargo.toml`
- `pykwavers/src/lib.rs`
- `pykwavers/python/pykwavers/__init__.py`

### Diagnostic Tools
- `pykwavers/debug_pstd_trace.py` (new)

---

## References

1. Previous Investigation: `PSTD_SOURCE_AMPLIFICATION_BUG.md`
2. Previous Session: `SESSION_SUMMARY_2026-02-04_PSTD_AMPLITUDE_BUG.md`
3. Sprint Context: Thread `71f2db49-a50f-4c29-a847-32147117665a`
4. k-Wave Documentation: Source injection and PML boundaries
5. Treeby & Cox (2010): k-Wave MATLAB Toolbox, J. Biomed. Opt.

---

## Appendix: Full Tracing Output Sample

```
2026-02-05T03:43:32.780378Z DEBUG kwavers::solver::forward::pstd::implementation::core::orchestrator: 331: 
  PSTD source injection mode determined 
  num_active=4096 mask_sum=4096.0 mask_min=1.0 mask_max=1.0 is_boundary_plane=true scale=1.0

2026-02-05T03:43:32.829010Z DEBUG kwavers::solver::forward::pstd::implementation::core::stepper: 254: 
  PSTD applying additive pressure source 
  time_step=1 source_idx=0 amp=21594.144754930778 
  mask_active=4096 mask_sum=4096.0 mask_max=1.0 scale=1.0 contribution=21594.144754930778

2026-02-05T03:43:32.829010Z DEBUG kwavers::solver::forward::pstd::implementation::core::stepper: 279: 
  PSTD pressure field after source injection 
  time_step=1 p_max_before=0.0 p_max_after=21594.144754930778 delta=21594.144754930778

[... continues for 50 time steps ...]

Final Result:
- Sensor pressure max: 14.8 kPa (14848 Pa)
- Expected: 100 kPa
- Error: -85.2%
```

---

## Session Summary

**Date:** 2026-02-05  
**Duration:** ~3 hours  
**Outcome:** Amplification bug confirmed and reproduced

### Accomplishments
1. ✅ Added comprehensive tracing infrastructure to PSTD core
2. ✅ Created `init_tracing()` function in pykwavers for runtime diagnostics
3. ✅ Verified source injection mode determination is correct (scale=1.0, boundary plane detection)
4. ✅ Created reproducible test case (`quick_pstd_diagnostic.py`)
5. ✅ Confirmed PSTD amplification bug: 6.23× (623 kPa vs 100 kPa expected)
6. ✅ Identified FDTD also has amplification: 1.36× (less severe)

### Artifacts Created
- `kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs` - tracing added
- `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs` - tracing added
- `pykwavers/src/lib.rs` - `init_tracing()` function
- `pykwavers/Cargo.toml` - added `structured-logging` feature
- `pykwavers/debug_pstd_trace.py` - initial diagnostic script
- `pykwavers/quick_pstd_diagnostic.py` - **reproducible test case**
- `pykwavers/validate_pstd_propagation.py` - comprehensive validation (too slow)
- `PSTD_SOURCE_INJECTION_DIAGNOSTIC_2026-02-05.md` - this document

### Bug Characteristics
- **Severity:** CRITICAL - blocks production use of PSTD solver
- **Scope:** Affects plane wave sources; potentially all additive sources
- **Magnitude:** 6.23× amplification in PSTD (523% error)
- **Reproducibility:** 100% reproducible with `quick_pstd_diagnostic.py`
- **Related Issue:** FDTD shows 1.36× amplification (may share root cause)

### Hypotheses
1. **FFT normalization error** - forward/inverse FFT scaling incorrect
2. **Source accumulation** - source applied multiple times per timestep
3. **Density-pressure feedback loop** - `update_density_from_pressure()` after source injection
4. **Spectral operator error** - k-space operators amplify instead of propagate
5. **PyO3 binding issue** - signal amplitude scaling in Python→Rust conversion

### Next Session Goals
1. Add step-by-step field evolution tracing
2. Verify FFT normalization constants
3. Check for source injection duplication
4. Compare with k-Wave reference implementation
5. Fix the bug and add regression tests

---

**Mathematical Verification Status:** ❌ FAILED  
**Production Readiness:** ❌ BLOCKED  
**Action Required:** Immediate investigation of amplification mechanism (Priority 1)  
**Estimated Fix Time:** 1-2 sessions (4-8 hours)