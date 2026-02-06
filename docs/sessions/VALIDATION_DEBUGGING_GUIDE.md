# Validation Debugging Quick Reference Guide

**Purpose**: Fast reference for debugging pykwavers validation failures  
**Date**: 2025-01-20  
**Status**: Active Investigation

---

## Quick Diagnostics

### 1. Run Full Validation
```bash
cd kwavers
cargo xtask validate
```

### 2. Check Individual Solver
```bash
cd kwavers/pykwavers
.venv/Scripts/python examples/compare_all_simulators.py
```

### 3. Run Minimal Test
```bash
cd kwavers/pykwavers
.venv/Scripts/python examples/debug_comparison.py
```

### 4. Unit Tests Only
```bash
cd kwavers
cargo test --test test_plane_wave_injection_fixed
```

---

## Current Failure Symptoms

### PSTD: Negative Correlation (-0.7782)
**Meaning**: Signals are 180° out of phase (inverted)

**Check List**:
- [ ] Signal polarity: `amplitude` should be positive in source creation
- [ ] FFT normalization: Check `inverse_into()` scaling
- [ ] Time indexing: Verify `t = time_step_index * dt` not off-by-one
- [ ] Phase convention: sin(2πft) vs cos(2πft)
- [ ] Pressure sign convention: positive compression vs negative

**Debug Commands**:
```rust
// In stepper.rs, add to apply_dynamic_pressure_sources():
println!("DEBUG: t={:.3e}, amp={:.3e}, p_before={:.3e}", 
         t, amp, self.fields.p[[32,32,0]]);
```

### FDTD: Poor Correlation (0.6035)
**Meaning**: Signals are partially aligned but with large errors

**Check List**:
- [ ] CFL condition: `dt < dx/c0`
- [ ] Source timing: Applied before velocity update?
- [ ] Boundary conditions: PML/CPML properly configured?
- [ ] Grid size: Sufficient points per wavelength (>10)?
- [ ] Amplitude normalization: Check `scale = 1.0/N` logic

**Debug Commands**:
```rust
// In solver.rs, add to step_forward():
if self.time_step_index < 5 {
    println!("FDTD step {}: p_max={:.3e}", 
             self.time_step_index, self.fields.p.iter().fold(0.0, |m,&v| m.max(v.abs())));
}
```

---

## Common Root Causes

### 1. Time Indexing Off-By-One
**Symptom**: Phase shift, early/late arrival

**Fix**: Ensure `amplitude(t)` uses current timestep:
```rust
let t = self.time_step_index as f64 * dt;  // Not (time_step_index + 1)
```

### 2. FFT Normalization Mismatch
**Symptom**: Amplitude scaling wrong, negative correlation

**Check**: Forward FFT and inverse FFT scaling
```rust
// Should be:
// forward: no scaling (or 1/N)
// inverse: 1/N (or no scaling)
// Product must equal 1/N total
```

### 3. Source Mask Normalization
**Symptom**: Amplitude 10×-100× wrong

**Fix**: Always normalize additive sources:
```rust
let scale = 1.0 / (mask.iter().filter(|&&m| m.abs() > 1e-12).count() as f64);
```

### 4. Pressure vs Density Confusion (PSTD)
**Symptom**: Large errors, wrong timing

**Fix**: Apply sources to pressure field, not density:
```rust
// CORRECT (after update_pressure):
Zip::from(&mut self.fields.p).and(mask).for_each(|p, &m| {
    *p += m * amp * scale;
});

// WRONG (modifying rho before pressure update):
// Zip::from(&mut self.rho).and(mask).for_each(...)
```

---

## Validation Test Configuration

### Standard Test (compare_all_simulators.py)
```python
config = SimulationConfig(
    grid_shape=(64, 64, 64),
    grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),  # 0.1 mm
    sound_speed=1500.0,                       # m/s
    density=1000.0,                           # kg/m³
    source_frequency=1e6,                     # 1 MHz
    source_amplitude=1e5,                     # 100 kPa
    source_position=None,                     # Plane wave at z=0
    sensor_position=(3.2e-3, 3.2e-3, 3.2e-3), # Center of domain
    duration=10e-6,                           # 10 μs
    pml_size=10,
    cfl=0.3,
)
```

### Expected Results (Passing)
```
L2 error:     < 0.01  (1%)
Linf error:   < 0.05  (5%)
Correlation:  > 0.99
RMSE:         < 1000 Pa (1% of 100 kPa)
```

### Current Results (Failing)
```
FDTD:   L2=18.8,  Linf=11.0,  correlation=0.6035
PSTD:   L2=1.00,  Linf=1.00,  correlation=-0.7782
Hybrid: L2=1.00,  Linf=1.00,  correlation=-0.7782
```

---

## Debugging Workflow

### Step 1: Verify Adapter Configuration
```python
# In pykwavers/python/pykwavers/comparison.py
# Add debug output in config_to_pykwavers():

print(f"DEBUG: Creating source")
print(f"  mask shape: {mask.shape}")
print(f"  mask sum: {np.sum(mask)}")
print(f"  signal shape: {signal.shape}")
print(f"  signal[0:5]: {signal[0:5]}")
print(f"  signal min/max: {np.min(signal)}, {np.max(signal)}")
```

### Step 2: Compare Source Application
```rust
// In kwavers: stepper.rs
pub(crate) fn apply_dynamic_pressure_sources(&mut self, dt: f64) {
    let t = self.time_step_index as f64 * dt;
    
    if self.time_step_index < 10 {
        println!("DEBUG apply_dynamic_pressure_sources:");
        println!("  time_step_index: {}", self.time_step_index);
        println!("  t: {:.6e} s", t);
        println!("  num sources: {}", self.dynamic_sources.len());
    }
    
    for (idx, (source, mask)) in self.dynamic_sources.iter().enumerate() {
        let amp = source.amplitude(t);
        
        if self.time_step_index < 10 {
            println!("  source {}: amp={:.3e}", idx, amp);
            let mode = self.source_injection_modes[idx];
            println!("  source {}: mode={:?}", idx, mode);
        }
        
        // ... rest of function
    }
}
```

### Step 3: Verify Field Values
```rust
// After source application:
if self.time_step_index < 10 {
    let p_boundary = self.fields.p[[32, 32, 0]];
    let p_center = self.fields.p[[32, 32, 32]];
    println!("  After sources: p_boundary={:.3e}, p_center={:.3e}", 
             p_boundary, p_center);
}
```

### Step 4: Check k-Wave Equivalence
```python
# Create minimal k-wave-python script
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC

# Exact same parameters as pykwavers test
grid = kWaveGrid([64, 64, 64], [0.1e-3, 0.1e-3, 0.1e-3])
medium = kWaveMedium(sound_speed=1500.0, density=1000.0)

# Plane wave source
source = kSource()
source.p_mask = np.zeros((64,64,64), dtype=bool)
source.p_mask[:, :, 0] = True
t = np.arange(500) * grid.dt
source.p = 1e5 * np.sin(2*np.pi*1e6*t)

# Sensor at center
sensor = kSensor()
sensor.mask = np.zeros((64,64,64), dtype=bool)
sensor.mask[32,32,32] = True

# Run
sensor_data = kspaceFirstOrder3DC(grid, medium, source, sensor, ...)

# Compare with pykwavers
print(f"k-Wave signal[0:10]: {sensor_data.p[0:10]}")
# Should match pykwavers sensor output
```

---

## File Locations

### Core Solver Code
```
kwavers/src/solver/forward/pstd/implementation/core/stepper.rs
  - apply_dynamic_pressure_sources()  [Line ~177]
  - step_forward()                    [Line ~12]

kwavers/src/solver/forward/fdtd/solver.rs
  - apply_dynamic_pressure_sources()  [Line ~260]
  - step_forward()                    [Line ~226]
```

### Test Code
```
kwavers/tests/test_plane_wave_injection_fixed.rs
  - test_plane_wave_boundary_injection_pstd()  [Line ~127]
  - test_plane_wave_boundary_injection_fdtd()  [Line ~36]
```

### Validation Code
```
kwavers/pykwavers/python/pykwavers/comparison.py
  - config_to_pykwavers()     [Line ~226]
  - config_to_kwave_python()  [Line ~300]

kwavers/pykwavers/examples/compare_all_simulators.py
  - main()                    [Line ~330]
```

---

## Error Pattern Recognition

### Pattern 1: All Solvers Fail Similarly
**Likely Cause**: Adapter/configuration issue (not solver bug)
**Action**: Focus on `comparison.py` configuration matching

### Pattern 2: PSTD Fails, FDTD Works
**Likely Cause**: PSTD-specific issue (FFT, spectral, timing)
**Action**: Check PSTD source application and FFT normalization

### Pattern 3: Negative Correlation Only
**Likely Cause**: Sign/phase inversion somewhere
**Action**: Check all sign conventions (amplitude, FFT, pressure definition)

### Pattern 4: Magnitude Only (Good Phase)
**Likely Cause**: Normalization factor wrong
**Action**: Check `scale = 1.0/N` calculation and application

### Pattern 5: Time Shift (Good Shape, Wrong Position)
**Likely Cause**: Time indexing off-by-one
**Action**: Verify `t = time_step_index * dt` everywhere

---

## Analytical Test Cases

### 1D Plane Wave (Simplest)
```rust
// Expected: p(z,t) = A*sin(2πf(t - z/c))
// At sensor z=L/2, expect arrival at t = L/(2c)

let L = 64 * 0.1e-3;  // 6.4 mm
let c = 1500.0;       // m/s
let expected_arrival = L / (2.0 * c);  // 2.13 μs

// Validate: first arrival should be within 10% of expected
```

### Point Source (Spherical)
```rust
// Expected: p(r,t) = A/(4πr) * δ(t - r/c)
// Peak amplitude scales as 1/r

let r = distance_to_sensor;
let expected_peak = amplitude / r;

// Validate: observed peak within 2× of expected
```

### Energy Conservation
```rust
// Total energy should increase only from source injection
let E_field = integrate_energy(&p, &ux, &uy, &uz);
let E_source = integrate_source_power() * dt;

assert!((E_field - E_prev - E_source).abs() < 0.01 * E_source);
```

---

## Quick Fixes Checklist

If validation fails, try these in order:

1. [ ] Check source amplitude sign (should be positive)
2. [ ] Verify time indexing: `t = time_step_index * dt`
3. [ ] Check FFT normalization in `inverse_into()`
4. [ ] Verify source applied after `update_pressure()` (PSTD)
5. [ ] Verify source applied before `update_velocity()` (FDTD)
6. [ ] Check mask normalization: `scale = 1.0/N`
7. [ ] Verify sensor position matches k-Wave exactly
8. [ ] Check dt calculation matches k-Wave CFL
9. [ ] Verify grid spacing in meters (not mm)
10. [ ] Check source signal phase: sin(2πft) starting at t=0

---

## Contact/Resources

**Documentation**:
- Full analysis: `PSTD_SOURCE_TIMING_ANALYSIS.md`
- Session summary: `SESSION_SUMMARY_PSTD_VALIDATION.md`
- Previous fixes: `SOURCE_INJECTION_FIX_SUMMARY.md`

**k-Wave Resources**:
- Manual: http://www.k-wave.org/manual/
- Forum: http://www.k-wave.org/forum/
- GitHub: https://github.com/ucl-bug/k-wave

**Next Steps**: See `PSTD_SOURCE_TIMING_ANALYSIS.md` → Action Items

---

**Last Updated**: 2025-01-20  
**Status**: Active debugging, awaiting k-Wave source analysis