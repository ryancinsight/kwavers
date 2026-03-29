# Kwavers to k-wave-python Parity Implementation Roadmap

**Document Version:** 1.0  
**Date:** February 13, 2026  
**Status:** Implementation Phase

## Executive Summary

This document provides a comprehensive roadmap for implementing missing features in kwavers to achieve functional parity with k-wave-python. All implementations must be done in **kwavers (Rust)**, with pykwavers serving only as thin PyO3 wrappers.

## Current Status

✅ **Completed:**
- Clean architecture audit and refactoring
- Zero build warnings (kwavers and pykwavers)
- Basic functionality working (Grid, Medium, Source, Sensor, Simulation)
- Core solvers (FDTD, PSTD) operational
- Validation tests passing

⚠️ **Missing for k-wave parity:**
- Heterogeneous medium support
- Multiple sensor recording modes
- Per-dimension PML configuration
- Complete 2D transducer array workflows

## Implementation Priorities

### Phase 1: Critical Features (Required for 80% of Examples)

#### 1.1 Heterogeneous Medium Support
**Priority:** P0 - Critical  
**Status:** Not Implemented  
**Impact:** Blocks all realistic tissue modeling

**Current State:**
```rust
// Only HomogeneousMedium exists
pub struct HomogeneousMedium {
    sound_speed: f64,
    density: f64,
    // ...
}
```

**Required Implementation:**
```rust
// New: HeterogeneousMedium with spatially varying properties
pub struct HeterogeneousMedium {
    sound_speed: Array3<f64>,  // 3D array matching grid
    density: Array3<f64>,
    absorption: Array3<f64>,   // Optional: spatially varying absorption
    nonlinearity: Option<Array3<f64>>, // Optional: B/A parameter
}
```

**Implementation Details:**
- Location: `kwavers/src/domain/medium/heterogeneous.rs`
- Must implement `Medium` trait
- Support all existing medium operations (sound_speed(x,y,z), density(x,y,z), etc.)
- Efficient memory layout for spatial lookups
- Integration with solvers (FDTD, PSTD)

**API Design:**
```rust
impl HeterogeneousMedium {
    /// Create from 3D arrays
    pub fn from_arrays(
        sound_speed: Array3<f64>,
        density: Array3<f64>,
        absorption: Option<Array3<f64>>,
    ) -> Result<Self, MediumError>;
    
    /// Create from functions
    pub fn from_functions<F, G>(
        grid: &Grid,
        sound_speed_fn: F,
        density_fn: G,
    ) -> Self
    where
        F: Fn(f64, f64, f64) -> f64,
        G: Fn(f64, f64, f64) -> f64;
    
    /// Validate consistency with grid
    pub fn validate(&self, grid: &Grid) -> Result<(), ValidationError>;
}
```

**Python API:**
```python
# Create from numpy arrays
sound_speed_map = np.load('tissue_sound_speed.npy')
density_map = np.load('tissue_density.npy')
medium = kw.Medium.heterogeneous(
    sound_speed=sound_speed_map,
    density=density_map,
)

# Or from functions
def sound_speed_fn(x, y, z):
    if z < 0.02:  # Skin layer
        return 1600.0
    else:  # Soft tissue
        return 1540.0

medium = kw.Medium.from_function(
    grid=grid,
    sound_speed=sound_speed_fn,
    density=lambda x,y,z: 1000.0,
)
```

**Files to Modify:**
- `kwavers/src/domain/medium/mod.rs` - Add module
- `kwavers/src/domain/medium/traits.rs` - Ensure trait supports arrays
- `kwavers/src/solver/forward/fdtd/mod.rs` - Update medium access
- `kwavers/src/solver/forward/pstd/mod.rs` - Update medium access

**Testing:**
- Unit tests: Property lookups, edge cases
- Parity tests: Compare with k-wave-python heterogeneous medium
- Example: `us_bmode_phased_array` with realistic tissue

---

#### 1.2 Sensor Recording Modes
**Priority:** P0 - Critical  
**Status:** Partially Implemented (tracking exists, selection doesn't)  
**Impact:** Only pressure time-series returned; missing p_max, p_min, p_rms, p_final

**Current State:**
- Recording infrastructure exists in `domain/sensor/recorder/statistics.rs`
- `PointSensor` has `rms_pressure()` method
- But: No way to select which recordings to output

**Required Implementation:**

**Step 1: Recording Mode Selection**
```rust
// In domain/sensor/mod.rs or config
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecordingMode {
    Pressure,       // Time-series pressure (default)
    PressureMax,    // Maximum pressure over time
    PressureMin,    // Minimum pressure over time
    PressureRms,    // RMS pressure over time
    PressureFinal,  // Final pressure field
    Velocity,       // Particle velocity components
    VelocityMax,
    VelocityMin,
    VelocityRms,
}

pub struct SensorConfig {
    pub recording_modes: Vec<RecordingMode>,
    pub record_start_index: Option<usize>,  // Delayed recording
}
```

**Step 2: SimulationResult Enhancement**
```rust
pub struct SimulationResult {
    // Current fields
    pub sensor_data: Array2<f64>,  // Time-series (for backward compat)
    
    // New fields for multiple recordings
    pub recordings: HashMap<RecordingMode, Array>,
    
    // Specific accessors
    pub pressure_max: Option<Array3<f64>>,
    pub pressure_min: Option<Array3<f64>>,
    pub pressure_rms: Option<Array3<f64>>,
    pub pressure_final: Option<Array3<f64>>,
}
```

**Step 3: Integration with Solvers**
- Modify FDTD and PSTD solvers to track statistics during simulation
- Update at each time step: max, min, accumulate for RMS
- Store final field at simulation end

**Python API:**
```python
# Configure sensor with multiple recording modes
sensor = kw.Sensor.from_mask(
    mask,
    record=['p', 'p_max', 'p_rms', 'p_final']  # Select outputs
)

# Run simulation
result = sim.run(...)

# Access different recordings
pressure_time = result.sensor_data  # Default time-series
pressure_max = result.pressure_max   # Maximum over time
pressure_rms = result.pressure_rms   # RMS over time
```

**Files to Modify:**
- `kwavers/src/domain/sensor/config.rs` - Add RecordingMode
- `kwavers/src/domain/sensor/recorder/statistics.rs` - Enhance tracking
- `kwavers/src/simulation/mod.rs` - Update SimulationResult
- `kwavers/src/solver/forward/fdtd/solver.rs` - Track statistics
- `kwavers/src/solver/forward/pstd/orchestrator.rs` - Track statistics
- `pykwavers/src/lib.rs` - Expose recording configuration

**Testing:**
- Test each recording mode independently
- Compare p_max, p_min, p_rms with k-wave-python
- Test delayed recording (record_start_index)

---

#### 1.3 Complete 2D Transducer Array Support
**Priority:** P0 - Critical  
**Status:** Partially Implemented  
**Impact:** Missing elevation focusing, advanced beamforming

**Current Implementation:**
- Basic `TransducerArray2D` exists
- Electronic steering and focusing implemented
- Apodization support added

**Missing Features:**

**A. Elevation Focusing (Cylindrical Arrays)**
```rust
impl TransducerArray2D {
    /// Set elevation focus distance
    pub fn set_elevation_focus_distance(&mut self, distance: f64);
    
    /// Model cylindrical curvature in elevation
    pub fn with_elevation_curvature(
        self,
        radius: f64,
    ) -> Self;
}
```

**B. kWaveArray (Flexible Geometry) - Expose to Python**
```rust
// Already exists in kwavers: FlexibleTransducerArray
// Just needs PyO3 bindings

#[pyclass]
pub struct KWaveArray {
    inner: kwavers::domain::source::flexible::FlexibleTransducerArray,
}

#[pymethods]
impl KWaveArray {
    #[new]
    fn new() -> Self;
    
    fn add_arc_element(&mut self, position: (f64, f64, f64), radius: f64, diameter: f64);
    fn add_rect_element(&mut self, position: (f64, f64, f64), width: f64, height: f64);
    fn add_disc_element(&mut self, position: (f64, f64, f64), diameter: f64);
    
    fn get_array_binary_mask(&self, grid: &Grid) -> Py<PyArray3<bool>>;
    fn get_distributed_source_signal(&self, signal: &PyArray1<f64>) -> Py<PyArray2<f64>>;
}
```

**Python API:**
```python
# Create flexible array
array = kw.KWaveArray()
array.add_arc_element((0, 0, 0), radius=0.05, diameter=0.02)
array.add_disc_element((0.01, 0, 0), diameter=0.005)

# Get source mask
source_mask = array.get_array_binary_mask(grid)
source = kw.Source.from_mask(source_mask, signal)
```

**Files to Modify:**
- `pykwavers/src/lib.rs` - Add KWaveArray class
- `kwavers/src/domain/source/flexible/mod.rs` - Ensure complete API
- Documentation and examples

---

### Phase 2: High Priority Features

#### 2.1 Per-Dimension PML Configuration
**Priority:** P1 - High  
**Status:** Basic pml_size only  
**Impact:** Cannot optimize PML for non-cubic domains

**Current:**
```rust
pub struct PMLConfig {
    pub size: usize,  // Uniform PML size
}
```

**Required:**
```rust
pub struct PMLConfig {
    pub x_size: Option<usize>,
    pub y_size: Option<usize>,
    pub z_size: Option<usize>,
    pub x_alpha: Option<f64>,
    pub y_alpha: Option<f64>,
    pub z_alpha: Option<f64>,
}
```

**Implementation:**
- Extend CPML implementation
- Update PML boundary application per-dimension
- Python bindings for configuration

---

#### 2.2 Data Type Casting (f32/f64)
**Priority:** P1 - High  
**Status:** Hardcoded f64  
**Impact:** Memory usage and performance optimization

**Options:**
1. **Generic Types** (Preferred):
```rust
pub struct Medium<T: Float> {
    sound_speed: Array3<T>,
    // ...
}
```

2. **Runtime Selection**:
```rust
pub enum Precision {
    F32,
    F64,
}

pub struct SimulationConfig {
    pub precision: Precision,
}
```

**Challenges:**
- Requires changes throughout codebase
- Solver implementations need to be generic
- Testing matrix doubles

**Recommendation:** Start with Runtime Selection approach for minimal disruption

---

### Phase 3: Medium Priority

#### 3.1 Smoothing Options
**Priority:** P2 - Medium  
**Status:** Implemented in Rust, needs Python exposure

**Already exists:**
- `SmoothingMethod` enum
- `BoundarySmoothingConfig`
- Just needs to be added to Simulation options

**Quick Fix:**
```rust
// In pykwavers Simulation class
#[pyo3(signature = (..., smoothing=None))]
fn new(..., smoothing: Option<&str>) -> PyResult<Self> {
    let smoothing_method = match smoothing {
        Some("subgrid") => SmoothingMethod::Subgrid,
        Some("ghost_cell") => SmoothingMethod::GhostCell,
        None => SmoothingMethod::None,
        _ => return Err(PyValueError::new_err("Unknown smoothing method")),
    };
    // ...
}
```

---

#### 3.2 Source Scaling Control
**Priority:** P2 - Medium  
**Status:** Not Implemented

**k-wave-python:**
```python
simulation_options = SimulationOptions(
    scale_source_terms=True  # Apply k-space source scaling
)
```

**Implementation:**
Add flag to control k-space correction in source injection

---

### Phase 4: Nice-to-Have

#### 4.1 Stream to Disk
**Priority:** P3 - Low  
**Status:** Not Implemented

For large simulations, stream data to disk instead of memory

#### 4.2 Axisymmetric Solver
**Priority:** P3 - Low  
**Status:** Not Implemented

Specialized 2.5D solver for rotationally symmetric problems

#### 4.3 Elastic Wave Solver
**Priority:** P3 - Low  
**Status:** Not Implemented

Support for shear waves and mode conversion

## Implementation Order

### Week 1-2: Heterogeneous Medium
1. Implement `HeterogeneousMedium` struct
2. Implement `Medium` trait methods
3. Add Python bindings
4. Update solvers to handle heterogeneous media
5. Write comprehensive tests
6. Validate against k-wave-python

### Week 3-4: Sensor Recording Modes
1. Define `RecordingMode` enum
2. Enhance `SimulationResult` struct
3. Implement statistics tracking in solvers
4. Add Python API for recording selection
5. Test each mode independently
6. Compare with k-wave-python

### Week 5-6: 2D Transducer Arrays
1. Complete elevation focusing implementation
2. Expose `FlexibleTransducerArray` as `KWaveArray`
3. Add comprehensive examples
4. Validate beam patterns match k-wave-python

### Week 7-8: Polish and Advanced Features
1. Per-dimension PML
2. Data type casting (if time permits)
3. Smoothing options exposure
4. Performance optimization
5. Documentation

## Success Metrics

**Goal:** Run k-wave-python examples with >0.90 correlation

**Track:**
- `us_beam_patterns` - Requires p_max/p_rms recording
- `at_focused_bowl_3D` - Requires kWaveArray (flexible geometry)
- `ivp_photoacoustic_waveforms` - Requires heterogeneous medium
- `us_bmode_phased_array` - Requires complete 2D array support

## Files Structure

All implementations follow Clean Architecture:

```
kwavers/src/
├── domain/
│   ├── medium/
│   │   ├── heterogeneous.rs     # NEW: HeterogeneousMedium
│   │   └── mod.rs
│   ├── sensor/
│   │   ├── config.rs            # UPDATE: RecordingMode
│   │   └── recorder/
│   │       └── statistics.rs    # UPDATE: Tracking
│   └── source/
│       └── flexible/
│           └── mod.rs           # EXISTING: Needs PyO3 bindings
├── solver/
│   ├── forward/
│   │   ├── fdtd/
│   │   │   └── solver.rs        # UPDATE: Track statistics
│   │   └── pstd/
│   │       └── orchestrator.rs  # UPDATE: Track statistics
│   └── mod.rs
└── lib.rs                       # Re-exports

pykwavers/src/
├── lib.rs                       # UPDATE: Add KWaveArray
└── utils_bindings.rs            # EXISTING: Utility functions
```

## Testing Strategy

1. **Unit Tests:** Each new component in isolation
2. **Integration Tests:** Full simulation workflows
3. **Parity Tests:** Direct comparison with k-wave-python
4. **Performance Tests:** Benchmark against baseline

## Documentation Requirements

- Rust docstrings for all public APIs
- Python docstrings (via PyO3)
- Migration guide from k-wave-python
- Architecture decision records (ADRs)

## Risk Mitigation

**Risk:** Heterogeneous medium performance degradation
**Mitigation:** Profile and optimize spatial lookups, consider spatial indexing

**Risk:** Generic types breaking existing code
**Mitigation:** Feature-gate generic types, maintain f64 as default

**Risk:** Breaking changes to public API
**Mitigation:** Semantic versioning, deprecation warnings, migration guide

## Conclusion

This roadmap provides a structured approach to achieving k-wave-python parity. Focus on Phase 1 (Heterogeneous Medium, Recording Modes, 2D Arrays) to enable 80% of k-wave-python examples. All work stays in kwavers (Rust), with pykwavers as thin wrappers.

**Next Steps:**
1. Implement heterogeneous medium (Week 1-2)
2. Daily standup to track progress
3. Weekly parity validation tests
4. Document architectural decisions

---

## Layer Boundaries — SOC / SRP Architecture Debt

*Added: Phase 3 Audit (2026-03-26)*

This section documents known Separation of Concerns (SOC) and Single Responsibility
Principle (SRP) violations in the codebase, together with the concrete migration path
for each. The violations do **not** block functionality but should be resolved in a
future refactor to improve testability and maintainability.

### Violation 1: ODE Integration Logic in Physics Layer

**Violating modules:**
- `kwavers/src/physics/acoustics/bubble_dynamics/adaptive_integration/` — Adaptive
  Runge-Kutta time-stepping for bubble ODEs
- `kwavers/src/physics/acoustics/bubble_dynamics/imex_integration/` — IMEX
  (Implicit-Explicit) schemes for stiff bubble dynamics

**Rule violated:** Physics layer should define *equations*; the solver layer should
define *integration algorithms*.

**Migration plan:**
1. Define `trait BubbleOde` in `physics/acoustics/bubble_dynamics/` (equations only)
2. Create `solver/forward/ode/` module with generic `AdaptiveRkSolver<E: BubbleOde>`
   and `ImexSolver<E: BubbleOde>`
3. Move `adaptive_integration` and `imex_integration` to `solver/forward/ode/`
4. Callers in `clinical/therapy/` and `analysis/` import from new solver-layer path
5. Remove `pub use` re-exports from physics layer

**Effort estimate:** ~2 days; no user-visible API changes

---

### Violation 2: Thermal-Acoustic Coupling in Physics Layer

**Violating modules:**
- `kwavers/src/physics/foundations/coupling/acoustic_thermal.rs` — computes
  bio-heat (Pennes) equation coupling coefficients

**Rule violated:** Cross-physics coupling belongs in `solver/forward/coupled/`,
which already has `thermal_acoustic.rs` for the leapfrog coupled integrator.

**Migration plan:**
1. Move `acoustic_thermal.rs` coupling helpers into `solver/forward/coupled/`
2. Have the physics layer expose `PennesBioHeatParameters` struct only
3. Coupling calculation performed by solver during time-step

**Effort estimate:** ~0.5 days; internal change only

---

### Violation 3: Monolithic Solver Impls (SRP)

**Violating modules:**
- `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs` — 370 lines,
  handles velocity, density, pressure, source injection, and recording in one impl block
- `kwavers/src/solver/forward/fdtd/solver.rs` — 668 lines, similar scope

**Rule violated:** SRP — a struct should have one reason to change.

**Migration plan for PSTDSolver (stepper.rs):**
1. Extract `PSTDVelocityUpdater`, `PSTDDensityUpdater`, `PSTDPressureUpdater` structs
2. Each takes a `&mut PSTDFields` and implements a single-method trait
3. `PSTDSolver::step_forward` orchestrates the updaters
4. Source injection logic stays in `SourceHandler` (already partially extracted)

**Effort estimate:** ~3 days; refactor only, no physics changes

---

### Verification of Layer Compliance

To check for new SOC violations:

```bash
# Physics layer must not import from solver layer
grep -r "use crate::solver" kwavers/src/physics/ --include="*.rs"

# Solver layer may import from physics (one-way dependency)
# This is correct: solver uses physics equations

# Physics layer must not contain time-stepping loops
grep -rn "for.*step\|while.*time\|integrate.*dt" kwavers/src/physics/ \
    --include="*.rs" | grep -v "test\|#\["
```
