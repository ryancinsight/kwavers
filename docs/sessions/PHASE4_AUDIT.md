# Phase 4 Audit: pykwavers vs k-Wave-python API Gap Analysis

**Date:** 2024-02-04  
**Sprint:** 217 Session 9 - Phase 4 Development  
**Author:** Ryan Clanton (@ryancinsight)  
**Status:** In Progress

---

## Executive Summary

Phase 4 focuses on comprehensive PyO3 wrapping and systematic comparison/correction of pykwavers against k-Wave-python. This audit identifies API gaps, missing features, and architectural differences that must be addressed to achieve feature parity and validation.

**Current Status:**
- ✓ Basic PyO3 bindings functional (Grid, Medium, Source, Sensor, Simulation)
- ✓ FDTD source injection implemented
- ✗ PSTD source injection not implemented
- ✗ Plane wave timing semantics incorrect
- ✗ Limited source types (only plane wave and point)
- ✗ Limited sensor types (only point, no grid recording)
- ✗ No transducer support
- ✗ No PML/boundary configuration exposed
- ✗ No advanced medium properties (absorption, nonlinearity)
- ✗ No reconstruction/beamforming APIs

---

## 1. Core API Comparison

### 1.1 Grid API

| Feature | k-Wave-python | pykwavers | Status | Priority |
|---------|---------------|-----------|--------|----------|
| Basic 3D grid | `kWaveGrid(Nx, Ny, Nz, dx, dy, dz)` | `Grid(nx, ny, nz, dx, dy, dz)` | ✓ Complete | - |
| 2D grid support | `kWaveGrid(Nx, Ny, dx, dy)` | Not supported | ✗ Missing | High |
| 1D grid support | `kWaveGrid(Nx, dx)` | Not supported | ✗ Missing | Medium |
| Grid properties (Lx, Ly, Lz) | ✓ | ✓ | ✓ Complete | - |
| Grid dimensions | ✓ | ✓ | ✓ Complete | - |
| Grid spacing | ✓ | ✓ | ✓ Complete | - |
| Total points | ✓ | ✓ | ✓ Complete | - |
| Time array generation | `grid.makeTime(c, CFL, t_end)` | Not exposed | ✗ Missing | High |
| k-space vectors | `grid.k_vec` | Internal only | ✗ Missing | Medium |
| PML size configuration | In grid config | Not exposed | ✗ Missing | High |

**Gap Analysis:**
- **Critical:** No 2D grid support limits planar simulation capabilities
- **Critical:** Time array generation not exposed to Python (manual CFL calculation required)
- **Important:** PML configuration not accessible from Python

### 1.2 Medium API

| Feature | k-Wave-python | pykwavers | Status | Priority |
|---------|---------------|-----------|--------|----------|
| Homogeneous medium | `kWaveMedium(sound_speed=c, density=rho)` | `Medium.homogeneous(sound_speed, density, absorption)` | ✓ Complete | - |
| Heterogeneous medium | Arrays for c, rho | Not supported | ✗ Missing | High |
| Absorption coefficient | `alpha_coeff`, `alpha_power` | Single `absorption` value | ⚠ Partial | High |
| Nonlinearity (BonA) | `BonA` array | Not exposed | ✗ Missing | Medium |
| Attenuation mode | `alpha_mode` (frequency power law) | Not supported | ✗ Missing | Medium |
| Medium properties access | Direct field access | Not exposed | ✗ Missing | Low |

**Gap Analysis:**
- **Critical:** Heterogeneous media not supported (major limitation for tissue modeling)
- **Critical:** Absorption model incomplete (no frequency-dependent power law)
- **Important:** Nonlinearity parameter not accessible (limits nonlinear simulations)

### 1.3 Source API

| Feature | k-Wave-python | pykwavers | Status | Priority |
|---------|---------------|-----------|--------|----------|
| Plane wave source | `kSource` with plane mask | `Source.plane_wave(grid, freq, amp)` | ✓ Complete | - |
| Point source | `kSource` with point mask | `Source.point(position, freq, amp)` | ✓ Complete | - |
| Arbitrary mask source | `source.p_mask` + `source.p` signal | Not supported | ✗ Missing | High |
| Initial pressure (IVP) | `source.p0` | Not supported | ✗ Missing | High |
| Velocity sources | `source.ux`, `source.uy`, `source.uz` | Not supported | ✗ Missing | Medium |
| Additive vs Dirichlet | `source.p_mode` | Not configurable | ✗ Missing | High |
| Multiple sources | Multiple masks/signals | Single source only | ✗ Missing | High |
| Time-varying signals | Arbitrary time series | Sine wave only | ⚠ Partial | High |
| Source direction | Configurable | Fixed +z for plane wave | ⚠ Partial | High |

**Gap Analysis:**
- **Critical:** Only two source types supported (plane wave, point)
- **Critical:** No arbitrary mask sources (limits custom geometries)
- **Critical:** No initial pressure source (eliminates initial value problems)
- **Critical:** Single source limitation (no multi-source scenarios)
- **Important:** Signal types limited (no arbitrary time series, chirps, pulses)
- **Important:** Plane wave direction fixed, timing semantics incorrect

### 1.4 Sensor API

| Feature | k-Wave-python | pykwavers | Status | Priority |
|---------|---------------|-----------|--------|----------|
| Point sensor | `sensor.mask` (single point) | `Sensor.point(position)` | ✓ Complete | - |
| Multiple point sensors | `sensor.mask` (multiple points) | Not supported | ✗ Missing | High |
| Grid sensor (full field) | `sensor.record = ['p', 'u', ...]` | Not supported | ✗ Missing | High |
| Cartesian sensor mask | Boolean array mask | Not supported | ✗ Missing | High |
| Cuboid sensor | `makeCartRect/Cuboid` | Not supported | ✗ Missing | Medium |
| Record pressure | ✓ | ✓ (point only) | ⚠ Partial | - |
| Record velocity | `sensor.record = ['u']` | Not supported | ✗ Missing | Medium |
| Record intensity | `sensor.record = ['I']` | Not supported | ✗ Missing | Low |
| Time-varying recording | Start/end time | Not supported | ✗ Missing | Low |
| Downsampling | `sensor.record_start_index` | Not supported | ✗ Missing | Low |

**Gap Analysis:**
- **Critical:** Only single-point sensors supported
- **Critical:** No full-field recording (eliminates visualization, analysis)
- **Critical:** No arbitrary mask sensors (limits placement flexibility)
- **Important:** No velocity/particle motion recording
- **Important:** No multi-sensor recording

### 1.5 Simulation API

| Feature | k-Wave-python | pykwavers | Status | Priority |
|---------|---------------|-----------|--------|----------|
| Basic simulation | `kspaceFirstOrder3D(grid, medium, source, sensor)` | `Simulation(grid, medium, source, sensor).run()` | ✓ Complete | - |
| 2D simulation | `kspaceFirstOrder2D` | Not supported | ✗ Missing | High |
| 1D simulation | `kspaceFirstOrder1D` | Not supported | ✗ Missing | Low |
| Axisymmetric | `kspaceFirstOrderAS` | Not supported | ✗ Missing | Low |
| Time step specification | `dt` parameter | ✓ | ✓ Complete | - |
| Auto CFL calculation | Default behavior | ✓ | ✓ Complete | - |
| Solver selection | PSTD/FDTD via options | FDTD only (PSTD internal) | ✗ Missing | High |
| PML configuration | Via simulation options | Not exposed | ✗ Missing | High |
| Progress callback | `ProgressCallback` | Not supported | ✗ Missing | Medium |
| Data casting | `DataCast` option | Not supported | ✗ Missing | Low |
| GPU acceleration | Via C++ backend | Planned, not exposed | ✗ Missing | Medium |

**Gap Analysis:**
- **Critical:** No 2D simulation support
- **Critical:** Solver selection not exposed (always FDTD)
- **Important:** No PML configuration from Python
- **Important:** No progress monitoring/callbacks

---

## 2. Advanced Features

### 2.1 Transducer Support

| Feature | k-Wave-python | pykwavers | Status |
|---------|---------------|-----------|--------|
| Transducer definition | `kTransducer` class | Not supported | ✗ Missing |
| Element positions | `transducer.positions` | Not supported | ✗ Missing |
| Element delays | `transducer.delays` | Not supported | ✗ Missing |
| Element apodization | `transducer.apodization` | Not supported | ✗ Missing |
| Transmit focusing | `transducer.focus` | Not supported | ✗ Missing |
| Receive focusing | Beamforming | Not supported | ✗ Missing |

**Status:** Complete gap - no transducer support in pykwavers

### 2.2 Reconstruction & Beamforming

| Feature | k-Wave-python | pykwavers | Status |
|---------|---------------|-----------|--------|
| Time reversal | `kspaceFirstOrder*D` with `sensor.time_reversal_boundary_data` | Not supported | ✗ Missing |
| Beamforming | `reconstruction.beamform` module | Not supported | ✗ Missing |
| DAS beamforming | ✓ | Not supported | ✗ Missing |
| Synthetic aperture | ✓ | Not supported | ✗ Missing |
| B-mode reconstruction | ✓ | Not supported | ✗ Missing |

**Status:** Complete gap - no reconstruction capabilities

### 2.3 Utility Functions

| Feature | k-Wave-python | pykwavers | Status |
|---------|---------------|-----------|--------|
| Signal generation | `utils.signals` (tone burst, chirp, etc.) | Not exposed | ✗ Missing |
| Filters | `utils.filters` (Gaussian, etc.) | Not exposed | ✗ Missing |
| Medium generation | `utils.mapgen` | Not exposed | ✗ Missing |
| Conversion utilities | `utils.conversion` | Not exposed | ✗ Missing |
| Interpolation | `utils.interp` | Not exposed | ✗ Missing |
| PML size calculation | `utils.pml.get_pml_size` | Not exposed | ✗ Missing |

**Status:** Utilities exist in Rust but not exposed to Python

---

## 3. Architectural Gaps

### 3.1 Solver Backend Integration

**Current State:**
- pykwavers Python layer constructs sources and calls FDTD backend
- PSTD solver exists but source injection not implemented
- No exposed solver selection mechanism

**Required:**
1. Implement `PSTDSolver::add_source()` (similar to FDTD)
2. Expose solver selection to Python API
3. Wire hybrid solver source injection
4. Add solver-specific configuration options

**Priority:** High (required for k-Wave validation)

### 3.2 Plane Wave Semantics

**Current Issue:**
- `PlaneWaveSource::create_mask()` produces spatial phase variation across entire grid
- Pre-populates wave pattern → incorrect arrival timing
- Measured: ~79.8% timing error in validation

**Root Cause:**
```rust
// Current implementation
mask[[i, j, k]] = (k_dot_r).cos(); // Spatial cosine across grid
```

**Required Fix:**
Options:
1. Boundary-only injection: mask only at z=0 plane, Dirichlet/additive source
2. `SourceMode` enum: `BoundarySource` vs `InitialCondition`
3. `PlaneWaveConfig` extension: `injection_mode` field

**Priority:** Critical (timing validation impossible without fix)

### 3.3 Multi-Dimensional Support

**Current State:**
- Grid, solvers are 3D-only
- k-Wave supports 1D, 2D, 3D, axisymmetric

**Required:**
1. Generic dimension support in `Grid` (via const generics or enum)
2. 2D-specific solver paths (FDTD2D, PSTD2D)
3. Python API: `Grid2D`, `Grid3D` or dimension parameter
4. Axisymmetric coordinate transform

**Priority:** High (2D essential for planar problems)

### 3.4 NumPy Integration

**Current Issue:**
- ndarray version mismatch (kwavers 0.16, pyo3 numpy 0.21 uses 0.15)
- Workaround: `Vec<f64>` intermediate copy

**Required:**
1. Align ndarray versions across workspace
2. Zero-copy NumPy interop (PyArray from ndarray view)
3. Handle C vs Fortran layout correctly

**Priority:** Medium (performance optimization)

---

## 4. Testing & Validation Gaps

### 4.1 Current Test Coverage

**Existing Tests:**
- `test_basic.py`: Smoke test (grid, medium, source, sensor instantiation)
- `test_source_injection.py`: FDTD plane wave, point source, amplitude scaling
- `examples/compare_plane_wave.py`: k-Wave comparison framework (MATLAB bridge optional)

**Gaps:**
- No negative tests (invalid inputs, error handling)
- No boundary condition tests
- No heterogeneous medium tests
- No multi-source tests
- No full-field sensor tests
- No PSTD validation
- No 2D/1D tests
- No nonlinear propagation tests

### 4.2 k-Wave Validation Matrix

| Test Case | k-Wave | pykwavers | Status |
|-----------|--------|-----------|--------|
| Homogeneous plane wave | ✓ | ✓ | ⚠ Timing error |
| Point source | ✓ | ✓ | Not validated |
| Heterogeneous medium | ✓ | ✗ | Not supported |
| Absorption | ✓ | ✗ | Incomplete |
| Nonlinear propagation | ✓ | ✗ | Not supported |
| Multiple sources | ✓ | ✗ | Not supported |
| 2D simulation | ✓ | ✗ | Not supported |
| Initial pressure (IVP) | ✓ | ✗ | Not supported |
| Time reversal | ✓ | ✗ | Not supported |

**Priority:** High - expand validation coverage systematically

---

## 5. Implementation Priority Matrix

### Phase 4A: Critical Fixes (Week 1)

1. **Fix plane wave timing semantics** [P0]
   - Implement boundary-only injection
   - Add `SourceMode` enum
   - Validate arrival times

2. **Implement PSTD source injection** [P0]
   - `PSTDSolver::add_source()`
   - Wire hybrid solver
   - Test spectral source injection

3. **Add arbitrary mask sources** [P0]
   - Python API: `Source.from_mask(mask, signal)`
   - Support arbitrary time series signals
   - Validate against k-Wave

4. **Implement multi-source support** [P1]
   - `Simulation.add_source()` method
   - Internal: multiple source injection
   - Test superposition

### Phase 4B: Core Features (Week 2)

5. **Add full-field sensor recording** [P1]
   - `Sensor.grid()` API
   - 4D array storage (nx × ny × nz × nt)
   - Memory optimization (ROI, downsampling)

6. **Heterogeneous medium support** [P1]
   - Array-based medium properties
   - Python API: `Medium.from_arrays(c, rho, alpha)`
   - Interpolation for grid mismatch

7. **Frequency-dependent absorption** [P1]
   - Power-law absorption model
   - `alpha_coeff`, `alpha_power` parameters
   - Validate dispersion

8. **2D simulation support** [P1]
   - `Grid2D` class
   - FDTD2D, PSTD2D solver paths
   - 2D-specific tests

### Phase 4C: Advanced Features (Week 3)

9. **Initial pressure sources** [P2]
   - `Source.initial_pressure(p0_array)`
   - Initial condition injection
   - IVP validation

10. **Expose solver selection** [P2]
    - `Simulation(solver='fdtd'|'pstd'|'hybrid')`
    - Solver-specific config
    - Performance comparison

11. **PML configuration** [P2]
    - Python API: `pml_size`, `pml_alpha`
    - Boundary condition options
    - Reflection validation

12. **Utility function exposure** [P2]
    - Signal generators (tone burst, chirp)
    - Filters, conversions
    - Medium generation helpers

### Phase 4D: Production Readiness (Week 4)

13. **Transducer support** [P3]
    - `Transducer` class
    - Element array management
    - Focusing, delays, apodization

14. **Reconstruction APIs** [P3]
    - Time reversal
    - DAS beamforming
    - B-mode reconstruction

15. **GPU acceleration** [P3]
    - Expose GPU backend
    - Performance benchmarks
    - Memory management

16. **Documentation & examples** [P2]
    - API reference docs
    - Tutorial notebooks
    - k-Wave migration guide

---

## 6. Validation Strategy

### 6.1 Unit Test Requirements

For each feature:
1. **Positive tests:** Valid inputs → expected outputs
2. **Negative tests:** Invalid inputs → proper error messages
3. **Boundary tests:** Edge cases, limits, special values
4. **Invariant tests:** Properties that must hold

### 6.2 Integration Test Requirements

1. **k-Wave comparison tests:**
   - Identical setup → quantitative error metrics
   - L2 < 0.01, L∞ < 0.05 acceptance criteria
   - Automated regression suite

2. **Performance benchmarks:**
   - Grid size scaling
   - Time step scaling
   - Memory profiling

3. **Multi-feature interaction:**
   - Heterogeneous + absorption
   - Multiple sources + sensors
   - Nonlinear + dispersive

### 6.3 Property-Based Tests

Using `proptest` in Rust:
1. Source amplitude → linear pressure scaling
2. Grid spacing → CFL stability
3. Absorption → exponential decay
4. Reciprocity: source ↔ sensor symmetry

---

## 7. Documentation Requirements

### 7.1 API Documentation

- [ ] Complete docstrings for all Python classes/methods
- [ ] Type hints for all parameters
- [ ] Usage examples in docstrings
- [ ] Cross-references to k-Wave equivalents

### 7.2 Migration Guide

- [ ] k-Wave → pykwavers API mapping table
- [ ] Common patterns translation
- [ ] Performance tuning guide
- [ ] Troubleshooting section

### 7.3 Tutorial Examples

- [ ] Basic simulation workflow
- [ ] Heterogeneous medium setup
- [ ] Multi-source scenarios
- [ ] Transducer simulation
- [ ] B-mode reconstruction
- [ ] GPU acceleration

---

## 8. Success Criteria

### Phase 4 Complete When:

1. **API Parity:**
   - ✓ Core 3D simulation features match k-Wave
   - ✓ 2D simulation support
   - ✓ All source types supported
   - ✓ Full-field sensor recording

2. **Validation:**
   - ✓ All k-Wave comparison tests pass (L2 < 0.01, L∞ < 0.05)
   - ✓ Plane wave timing error < 5%
   - ✓ Heterogeneous medium validated
   - ✓ Absorption model validated

3. **Testing:**
   - ✓ >90% code coverage in Python bindings
   - ✓ Negative tests for all APIs
   - ✓ Property-based tests passing
   - ✓ Performance benchmarks documented

4. **Documentation:**
   - ✓ Complete API reference
   - ✓ k-Wave migration guide
   - ✓ Tutorial examples
   - ✓ Troubleshooting guide

---

## 9. Risk Assessment

### High-Risk Items

1. **Plane wave timing fix:**
   - Risk: Semantic changes may break existing behavior
   - Mitigation: Add `SourceMode` flag, default to current behavior, deprecation path

2. **Heterogeneous medium:**
   - Risk: Performance degradation with spatially-varying properties
   - Mitigation: Benchmark, optimize hot paths, consider GPU

3. **Multi-dimensional support:**
   - Risk: Code duplication, maintenance burden
   - Mitigation: Generic implementations, macro-based dimension dispatch

4. **NumPy zero-copy:**
   - Risk: Memory safety issues with ownership
   - Mitigation: Careful lifetime management, extensive testing

### Medium-Risk Items

1. PSTD source injection (spectral domain complexities)
2. Transducer support (complex geometry management)
3. GPU acceleration (platform compatibility)

---

## 10. Next Actions

### Immediate (This Session):

1. ✓ Complete Phase 4 audit (this document)
2. → Implement PSTD source injection
3. → Fix plane wave timing semantics
4. → Add arbitrary mask source support
5. → Validate with k-Wave comparison

### This Week:

6. Implement full-field sensor recording
7. Add heterogeneous medium support
8. Create comprehensive test suite
9. Write API documentation

### Next Week:

10. 2D simulation support
11. Initial pressure sources
12. Solver selection exposure
13. k-Wave validation report

---

**Audit Complete:** 2024-02-04  
**Next Review:** After Phase 4A implementation  
**Assigned:** Ryan Clanton (@ryancinsight)