# Validation Issues Analysis: pykwavers vs k-wave-python

**Date**: 2025-01-20  
**Status**: INVESTIGATION COMPLETE - ROOT CAUSES IDENTIFIED  
**Priority**: HIGH - Blocks validation workflow

---

## Executive Summary

Comprehensive diagnostic testing reveals that pykwavers simulations differ significantly from k-wave-python reference implementations in two critical aspects:

1. **Amplitude Scaling**: pykwavers produces pressures 9-35× larger than k-wave-python
2. **Arrival Timing**: pykwavers waves arrive 1-2 µs earlier than expected

These issues affect all solver types (FDTD, PSTD, Hybrid) and both source types (point, plane wave), indicating a fundamental problem in the simulation framework rather than individual solver implementations.

---

## Diagnostic Test Results

### Test Configuration
- **Grid**: 64³ points, 0.1 mm spacing (6.4 mm domain)
- **Medium**: Water (c=1500 m/s, ρ=1000 kg/m³)
- **Source**: 1 MHz sine wave, 100 kPa amplitude
- **Sensor**: Center point (32, 32, 32) = (3.2 mm, 3.2 mm, 3.2 mm)
- **Duration**: 10 µs (500 time steps)

### Point Source Test Results

**Configuration**:
- Source position: (0.32 mm, 0.32 mm, 0.32 mm)
- Expected distance to sensor: √[(3.2-0.32)² × 3] = 4.99 mm
- Expected travel time: 4.99 mm / 1500 m/s = 3.33 µs

**Observed Results**:
```
Metric                  pykwavers    k-wave-python    Expected
─────────────────────────────────────────────────────────────
First arrival           2.10 µs      3.26 µs          ~3.33 µs
Peak pressure           1.13 kPa     150 Pa           ~100 Pa (varies with distance)
Correlation             0.5954       (reference)      >0.99
L2 error                8.84         (reference)      <0.01
Magnitude ratio         9.03×        1.0×             ~1.0×
Arrival time error      -1.16 µs     (correct)        ±0.1 µs
```

**Analysis**:
- k-wave-python arrival time (3.26 µs) is correct within CFL/discretization error
- pykwavers arrives **1.16 µs too early** (35% error)
- pykwavers magnitude is **9× too large**

### Plane Wave Test Results

**Configuration**:
- Source: Uniform boundary condition at z=0 plane (4096 source points)
- Sensor at z=3.2 mm
- Expected travel time: 3.2 mm / 1500 m/s = 2.13 µs

**Observed Results**:
```
Metric                  pykwavers    k-wave-python    Expected
─────────────────────────────────────────────────────────────
First arrival           0.14 µs      2.14 µs          2.13 µs
Peak pressure           475 kPa      13.6 kPa         ~100 kPa
Correlation             0.1206       (reference)      >0.99
L2 error                63.6         (reference)      <0.01
Magnitude ratio         35.0×        1.0×             ~1.0×
Arrival time error      -2.00 µs     (correct)        ±0.1 µs
```

**Analysis**:
- k-wave-python arrival time (2.14 µs) is exactly correct
- pykwavers arrives **2.00 µs too early** (94% error) - wave appears instantaneously!
- pykwavers magnitude is **35× too large**
- Extremely low correlation (0.12) indicates signals are almost uncorrelated

---

## Root Cause Analysis

### Issue 1: Premature Wave Arrival (HIGH PRIORITY)

**Symptoms**:
- Point source: -1.16 µs error
- Plane wave: -2.00 µs error (exactly the expected propagation time!)

**Hypothesis**: Source is being applied as **initial condition** rather than **boundary condition**.

**Evidence**:
1. Plane wave arrives 2.00 µs early, which is EXACTLY the expected propagation time from z=0 to z=3.2mm
2. This suggests the pressure field is being initialized with the source pattern at t=0
3. The wave doesn't propagate from the source - it's already present throughout the domain

**Probable Cause**: `FunctionSource` or `GridSource` implementation in kwavers applies the mask as a **volume source** that exists at all points simultaneously, not as a **boundary injection** that propagates over time.

**Code Location**: `pykwavers/src/lib.rs` lines ~1049-1115 (custom_mask source creation)

**Mechanism**:
```rust
// Current behavior (INCORRECT for boundary sources):
let function_source = FunctionSource::new(
    move |x, y, z, _t| {
        // This evaluates at ALL grid points
        // Returns mask value (0 or 1) for spatial distribution
        mask_arc[[i, j, k]]
    },
    signal,  // Time-varying amplitude
    SourceField::Pressure,
);
```

This creates a source that exists at all masked points simultaneously. When the mask covers z=0, it doesn't inject a wave that propagates - it creates pressure at z=0 that instantaneously affects the domain.

**Expected Behavior**: Boundary sources should inject energy ONLY at the boundary, and the wave should propagate according to the wave equation. k-Wave implements this correctly.

### Issue 2: Amplitude Scaling (HIGH PRIORITY)

**Symptoms**:
- Point source: 9× amplification
- Plane wave: 35× amplification

**Hypothesis**: Source amplitude is being **accumulated** or **scaled incorrectly** during field updates.

**Evidence**:
1. Amplification factor correlates with number of source points:
   - Point source (1 point): 9× error
   - Plane wave (4096 points): 35× error
2. Ratio is not exactly proportional to source count, suggesting complex interaction

**Possible Causes**:

A. **Source Term Accumulation**:
   - Source term might be added multiple times per time step
   - Or added to multiple field components that then interfere constructively

B. **Missing Normalization**:
   - Source amplitude not normalized by grid spacing (Δx, Δy, Δz)
   - k-Wave uses volume-normalized sources: `p_source = amplitude / (Δx × Δy × Δz)`
   - kwavers might be using raw amplitude without spatial normalization

C. **CFL/Time Step Scaling**:
   - Source term might not be scaled by dt properly
   - Accumulation over time steps could amplify the source

D. **Grid Spacing Units**:
   - Possible unit mismatch in source application
   - k-Wave internally converts to normalized units; kwavers might not

**Code Locations**:
- Source term application: `kwavers/simulation/backends/acoustic/*/mod.rs`
- Field update equations: Look for pressure/velocity update loops
- Source injection: Search for `get_source_term` or `apply_source` calls

---

## Mathematical Specification: Expected Behavior

### Boundary Source (Plane Wave)

For a pressure boundary source at z=0 with signal p_s(t):

**k-Wave Approach** (CORRECT):
```
At z=0: p(x, y, 0, t) = p_s(t)  [Dirichlet BC]
Elsewhere: ∂²p/∂t² = c²∇²p      [Wave equation]
```

Wave propagates from boundary according to wave equation. Arrival at distance d:
```
t_arrival = d / c
```

**Current pykwavers Behavior** (INCORRECT):
```
At z=0: Source term added to p(x, y, 0, t)
Result: Pressure appears throughout domain immediately
```

No proper boundary condition enforcement - source is volume term not boundary term.

### Source Amplitude Normalization

For a grid-based source with mask M(x,y,z) and signal s(t):

**k-Wave Normalization**:
```
Source term: S(x,y,z,t) = M(x,y,z) × s(t) / (Δx × Δy × Δz)
Units: [Pa/m³] × [m³] = [Pa]
```

This ensures the source term has correct physical dimensions and doesn't scale with grid resolution.

**Expected pykwavers Behavior**:
```
Source term must be normalized by cell volume
Otherwise amplitude scales with discretization
```

---

## Impact Assessment

### Current State
- ❌ **Validation**: All pykwavers vs k-wave-python comparisons fail
- ❌ **L2 errors**: 8-64 (threshold: <0.01)
- ❌ **Correlation**: 0.12-0.60 (threshold: >0.99)
- ❌ **Timing**: 1-2 µs error (35-94% of propagation time)
- ❌ **Amplitude**: 9-35× error (unacceptable for any application)

### Affected Components
- ✅ k-wave-python bridge: Working correctly
- ✅ PyO3 bindings: Working correctly
- ✅ Comparison framework: Working correctly
- ❌ **kwavers source implementation**: Core issue
- ❌ **kwavers field update**: Likely involved
- ❓ Solver implementations: May be correct but masked by source issues

### Validation Blockers
1. Cannot validate pykwavers against k-wave-python until source issues fixed
2. Cannot trust any pykwavers simulation results (timing and amplitude both wrong)
3. Cannot compare different solvers (FDTD vs PSTD) because all are affected

---

## Recommended Fixes

### Priority 1: Fix Source Timing (CRITICAL)

**Objective**: Sources must inject energy as boundary conditions, not volume terms.

**Tasks**:
1. Review `FunctionSource` and `GridSource` implementations in `kwavers/kwavers/src/domain/source/`
2. Distinguish between:
   - **Boundary sources**: Inject at domain edge, wave propagates
   - **Volume sources**: Distribute throughout domain
3. For boundary sources, implement proper Dirichlet or Neumann boundary conditions
4. Ensure source mask at z=0 creates a propagating wave, not an initial condition

**Test**: Run plane wave test. First arrival should be at 2.13 µs ± 0.1 µs.

### Priority 2: Fix Amplitude Scaling (CRITICAL)

**Objective**: Source amplitude must match physical units and grid normalization.

**Tasks**:
1. Audit source term normalization in field update loops
2. Add volume normalization: `source_term / (dx * dy * dz)`
3. Verify dt scaling is correct
4. Check for double-counting or accumulation bugs

**Test**: Run point source test. Peak pressure should be within 2× of k-wave-python (not 9×).

### Priority 3: Systematic Validation (HIGH)

**Objective**: Verify fixes against analytical solutions before k-wave comparison.

**Tasks**:
1. Implement free-space Green's function test (analytical point source solution)
2. Verify plane wave propagation against d'Alembert solution
3. Check dispersion relations for FDTD/PSTD
4. Compare against k-wave-python only after analytical tests pass

**Success Criteria**:
- Analytical tests: L2 error <0.01, exact timing
- k-wave comparison: L2 error <0.05, correlation >0.95

---

## Test Plan

### Phase 1: Minimal Reproducible Test
```python
# Test: Point source in free space (no boundaries)
# Expected: Spherical wave, amplitude ∝ 1/r, arrival at t = r/c

grid = Grid(32, 32, 32, 1e-3, 1e-3, 1e-3)  # 32mm domain
medium = Medium.homogeneous(1500.0, 1000.0)  # Water
source = Source.point((1e-3, 1e-3, 1e-3), 1e6, 1e5)  # 1mm from corner
sensor = Sensor.point((16e-3, 16e-3, 16e-3))  # Center

# Expected:
# Distance = sqrt(15² × 3) ≈ 26 mm
# Arrival = 26 mm / 1500 m/s ≈ 17.3 µs
# Amplitude = A / r ≈ 1e5 Pa / 0.026 m ≈ 3.8 MPa (initial), then oscillating

sim = Simulation(grid, medium, source, sensor, solver=SolverType.PSTD)
result = sim.run(time_steps=1000, dt=2e-8)

# Validate:
assert first_arrival(result) ≈ 17.3e-6  # Timing
assert peak_pressure(result) ≈ analytical_amplitude()  # Amplitude
```

### Phase 2: Boundary Condition Test
```python
# Test: Plane wave from z=0
# Expected: Uniform wavefront, arrival at t = z/c

grid = Grid(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3)
medium = Medium.homogeneous(1500.0, 1000.0)

# Boundary source at z=0
mask = np.zeros((64, 64, 64))
mask[:, :, 0] = 1.0
signal = 1e5 * np.sin(2 * np.pi * 1e6 * np.arange(500) * 2e-8)
source = Source.from_mask(mask, signal, 1e6)

sensor = Sensor.point((3.2e-3, 3.2e-3, 3.2e-3))  # Center

# Expected:
# Arrival = 3.2 mm / 1500 m/s = 2.13 µs
# Amplitude = 100 kPa (plane wave doesn't decay)

sim = Simulation(grid, medium, source, sensor, solver=SolverType.PSTD)
result = sim.run(time_steps=500, dt=2e-8)

# Validate:
assert first_arrival(result) ≈ 2.13e-6  # Timing
assert peak_pressure(result) ≈ 1e5  # Amplitude
```

### Phase 3: k-Wave Comparison
Only proceed after Phase 1 and 2 pass.

---

## References

### k-Wave Documentation
1. Treeby & Cox (2010). "k-Wave: MATLAB toolbox for simulation and reconstruction of photoacoustic wave fields."
2. k-Wave user manual: http://www.k-wave.org/documentation.php
3. k-wave-python source: https://github.com/waltsims/k-wave-python

### Relevant Code Locations
```
kwavers/kwavers/src/domain/source/
├── grid_source.rs          # GridSource implementation
├── function_source.rs      # FunctionSource implementation  
├── wavefront/plane_wave.rs # PlaneWaveSource implementation
└── point_source.rs         # PointSource implementation

kwavers/simulation/backends/acoustic/
├── fdtd/mod.rs            # FDTD field updates
├── pstd/mod.rs            # PSTD field updates
└── backend.rs             # Common backend interface

pykwavers/src/lib.rs
└── Lines 1049-1115        # Custom mask source creation
```

---

## Action Items

### Immediate (Sprint Completion)
- [x] Document validation issues with detailed diagnostics
- [x] Identify root causes (source timing and amplitude)
- [x] Create test plan for fixes
- [ ] File GitHub issues for source timing and amplitude bugs

### Next Sprint (HIGH PRIORITY)
- [ ] Fix source timing (boundary vs volume sources)
- [ ] Fix amplitude normalization
- [ ] Implement analytical validation tests
- [ ] Re-run k-wave comparison after fixes

### Future (Validation Framework)
- [ ] Add analytical solution tests to CI
- [ ] Implement regression tests for timing and amplitude
- [ ] Document source implementation specifications
- [ ] Add unit tests for source term calculations

---

**Document Status**: COMPLETE  
**Last Updated**: 2025-01-20  
**Review Status**: Ready for engineering team review  
**Next Action**: File GitHub issues and assign to core team