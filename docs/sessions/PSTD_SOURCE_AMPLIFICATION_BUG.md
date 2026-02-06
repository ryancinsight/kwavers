# PSTD Source Amplification Bug Investigation Summary

**Date**: 2026-02-04  
**Author**: Ryan Clanton (@ryancinsight)  
**Status**: Bug Identified, Root Cause Under Investigation  

## Executive Summary

PSTD solver exhibits a consistent 3.54x amplitude amplification bug when applying plane wave sources from masks. FDTD solver handles identical sources correctly, confirming the issue is specific to PSTD implementation.

## Problem Statement

When running a plane wave simulation with:
- Source: 100 kPa amplitude, 1 MHz sine wave
- Mask: Full z=0 plane (64×64 = 4096 points, all values = 1.0)
- Grid: 64³, 0.1 mm spacing
- Medium: Water (c=1500 m/s, ρ=1000 kg/m³)

**Expected Result**: Max pressure amplitude ≈ 100 kPa  
**FDTD Result**: 99.7 kPa (1.00x) ✓ CORRECT  
**PSTD Result**: 354.1 kPa (3.54x) ✗ INCORRECT  

## Reproduction

### Python Test (pykwavers)
```python
import numpy as np
import pykwavers as kw

grid_size = 64
spacing = 0.1e-3
c0, rho0 = 1500.0, 1000.0
freq, amplitude = 1e6, 1e5
duration, cfl = 5e-6, 0.3

dt = cfl * spacing / c0
nt = int(duration / dt)

grid = kw.Grid(nx=grid_size, ny=grid_size, nz=grid_size, 
               dx=spacing, dy=spacing, dz=spacing)
medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

# Plane wave mask at z=0
mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
mask[:, :, 0] = 1.0

t = np.arange(nt) * dt
signal = amplitude * np.sin(2 * np.pi * freq * t)

source = kw.Source.from_mask(mask, signal, frequency=freq)
sensor = kw.Sensor.point(position=(grid_size//2 * spacing, 
                                    grid_size//2 * spacing, 
                                    grid_size//2 * spacing))

# FDTD: Correct
sim_fdtd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD)
result_fdtd = sim_fdtd.run(time_steps=nt, dt=dt)
# Max amplitude: 99.7 kPa ✓

# PSTD: Incorrect (3.54x amplification)
sim_pstd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
result_pstd = sim_pstd.run(time_steps=nt, dt=dt)
# Max amplitude: 354.1 kPa ✗
```

## Investigation History

### Hypotheses Tested & Rejected

1. **PyO3 Bindings Issue** ✗
   - Both FDTD and PSTD use identical `create_source_arc()` path
   - FDTD works correctly, so bindings are not the issue

2. **TimeSeriesSignal Normalization** ✗
   - Initially suspected signal was double-counting amplitude
   - Attempted fix: Normalize signal and scale mask by max_amplitude
   - Result: No change (still 3.54x)
   - Conclusion: FDTD uses same TimeSeriesSignal correctly, so this is not the root cause

3. **FFT Normalization** ✗
   - Checked FFT forward/inverse normalization
   - kwavers uses standard convention: no normalization in forward, divide by N in inverse
   - Source applied in spatial domain, not frequency domain
   - FFT operations should not affect source amplitude

4. **Source Duplication** ✗
   - Checked if sources applied via both `source_handler` and `dynamic_sources`
   - Confirmed: `source_handler` initialized with empty `GridSource`
   - Sources only in `dynamic_sources` list
   - No duplication occurring

5. **Injection Mode Scale Factor** (Suspected but unverified)
   - `determine_injection_mode()` should detect z=0 plane and return scale=1.0
   - Cannot easily verify without making function public or adding debug logging
   - This remains a prime suspect

## Code Locations

### PSTD Source Application
**File**: `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs`

```rust
pub(crate) fn apply_dynamic_pressure_sources(&mut self, dt: f64) {
    let t = self.time_step_index as f64 * dt;
    for (idx, (source, mask)) in self.dynamic_sources.iter().enumerate() {
        let amp = source.amplitude(t);
        if amp.abs() < 1e-12 {
            continue;
        }

        if source.source_type() == SourceField::Pressure {
            let mode = self.source_injection_modes[idx];

            match mode {
                SourceInjectionMode::Boundary => {
                    Zip::from(&mut self.fields.p).and(mask).for_each(|p, &m| {
                        if m.abs() > 1e-12 {
                            *p += m * amp;  // scale=1.0 implicit
                        }
                    });
                }
                SourceInjectionMode::Additive { scale } => {
                    Zip::from(&mut self.fields.p).and(mask).for_each(|p, &m| {
                        if m.abs() > 1e-12 {
                            *p += m * amp * scale;  // ← KEY LINE
                        }
                    });
                }
            }
        }
    }
}
```

### Injection Mode Detection
**File**: `kwavers/src/solver/forward/pstd/implementation/core/orchestrator.rs`

```rust
fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
    let shape = mask.dim();
    let mut num_active = 0;
    let mut all_same_k = true;
    let mut first_k = None;
    // ... (determines if all points share same z-index)

    let is_boundary_plane = (all_same_i && (first_i == Some(0) || first_i == Some(shape.0 - 1)))
        || (all_same_j && (first_j == Some(0) || first_j == Some(shape.1 - 1)))
        || (all_same_k && (first_k == Some(0) || first_k == Some(shape.2 - 1)));

    let scale = if is_boundary_plane {
        1.0  // Plane wave: no normalization
    } else if num_active > 0 {
        1.0 / (num_active as f64)  // Point/volume source: normalize
    } else {
        1.0
    };

    SourceInjectionMode::Additive { scale }
}
```

## Key Observations

1. **Magnitude of Error**: 3.54x is not a simple factor like 2x, π, or sqrt(N)
   - 3.54 ≈ sqrt(12.5) ≈ sqrt(4096/327.68)
   - Could indicate multiple compounding issues

2. **Consistency**: Error is reproducible across different configurations
   - Same 3.54x ratio with 64³ grid
   - Suggests systematic bug, not numerical instability

3. **FDTD Correctness**: Proves the issue is PSTD-specific
   - Same source creation pipeline
   - Same PyO3 bindings
   - Different time-stepping implementation

4. **Time-Dependent Build-Up**: With short simulations (100 steps), amplitude is 0.26x
   - Suggests error accumulates or builds over time
   - Not a simple multiplicative factor applied once

## Recommended Next Steps

### Immediate Actions

1. **Add Debug Logging to `determine_injection_mode()`**
   ```rust
   // Temporarily make function public or add tracing
   tracing::debug!("Source injection mode: scale={:.6}, is_boundary={}, num_active={}", 
                   scale, is_boundary_plane, num_active);
   ```

2. **Verify Scale Factor Application**
   - Confirm that boundary planes actually get scale=1.0
   - Check if Additive vs Boundary enum variant matters

3. **Compare FDTD Source Application**
   - Examine how FDTD applies mask-based sources
   - Identify any normalization differences

4. **Test with Different Grid Sizes**
   - Check if 3.54x ratio changes with grid size
   - Would help identify if N-dependent

### Medium-Term Fixes

1. **Create Unit Test for `determine_injection_mode()`**
   ```rust
   #[test]
   fn test_plane_wave_detection() {
       let mut mask = Array3::zeros((64, 64, 64));
       mask.slice_mut(s![.., .., 0]).fill(1.0);
       
       let mode = PSTDSolver::determine_injection_mode(&mask);
       match mode {
           SourceInjectionMode::Additive { scale } => {
               assert_eq!(scale, 1.0, "Plane wave should have scale=1.0");
           }
           _ => panic!("Expected Additive mode"),
       }
   }
   ```

2. **Add Amplitude Validation Test**
   - Compare PSTD vs FDTD on identical setup
   - Assert amplitude within 10% tolerance
   - Add to CI regression tests

3. **Review k-Wave Source Implementation**
   - Check how k-wave-python handles plane wave sources
   - May reveal missing normalization or conditioning

### Root Cause Analysis Priorities

1. **Scale factor verification** (Highest priority)
2. **Time-stepping interaction** (FFT → source → FFT pipeline)
3. **Density-pressure consistency updates** (Recent changes in this area)
4. **Spectral operator interactions** (PML, anti-aliasing filters)

## Impact

### Severity: HIGH

- Breaks all PSTD plane wave simulations
- Validation against k-wave-python fails completely
- Affects production use of PSTD solver

### Scope

- **Affected**: PSTD solver with mask-based sources
- **Not Affected**: FDTD solver (works correctly)
- **Possibly Affected**: PSTD with point sources (needs testing)

## References

- **Validation Results**: `pykwavers/examples/results/validation_report.txt`
- **Earlier Fixes**: 
  - `PSTD_POLARITY_FIX_SUMMARY.md`
  - `SOURCE_INJECTION_FIX_SUMMARY.md`
- **Test Scripts**:
  - `pykwavers/debug_amplitude_simple.py`
  - `pykwavers/debug_mask_source.py`

## Notes

- This bug was discovered during validation against k-wave-python
- PSTD polarity and timing issues were previously fixed
- This is a distinct amplitude scaling issue
- The 3.54x factor remains unexplained and is the key to solving this bug

---

**Action Required**: Add debug logging to `determine_injection_mode()` and verify scale factor is actually 1.0 for boundary planes. If scale is correct, investigate time-stepping accumulation or FFT pipeline interactions.