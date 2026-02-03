# Sprint 217 Session 8: k-Wave Comparison Framework

**Session ID**: SPRINT_217_SESSION_8  
**Date**: 2026-02-04  
**Focus**: Establish mathematically rigorous k-Wave comparison framework and implement core acoustic solver validation  
**Status**: PLANNING  
**Phase**: Foundation (0-10% - Audit/Planning/Gap Analysis)

---

## Executive Summary

### Objective
Establish a mathematically rigorous comparison framework between kwavers and k-Wave (MATLAB toolbox), focusing on acoustic solver validation with formal specifications, empirical benchmarks, and analytical verification. This session transitions from unsafe documentation to implementation validation against the de facto standard in ultrasound simulation.

### Rationale
- **Mathematical Validation**: k-Wave represents 15+ years of peer-reviewed acoustic simulation research
- **Clinical Acceptance**: k-Wave is the industry standard for ultrasound simulation validation
- **Gap Identification**: Systematic comparison reveals missing features and correctness issues
- **Research Integration**: PRD identifies k-Wave as primary integration target for research excellence
- **Correctness First**: "Math > Working Code" - validate against analytical solutions, not "no crashes"

### Success Criteria
1. ✅ **Formal Mathematical Specifications**: Document k-Wave acoustic wave equations with literature references
2. ✅ **Comparison Infrastructure**: Python bridge to k-Wave/kwavepython for automated validation
3. ✅ **Analytical Test Suite**: 5+ analytical solutions (plane wave, Gaussian beam, spherical wave, etc.)
4. ✅ **Empirical Benchmarks**: Performance and accuracy comparisons on standard test cases
5. ✅ **Gap Analysis**: Document feature gaps, correctness issues, and implementation priorities
6. ✅ **Zero Regressions**: Maintain 2009/2009 test pass rate and zero compilation errors

---

## Background Context

### Current State (From Sprint 217 Sessions 1-7)

**Architecture Health (Session 1)**:
- 98/100 Architecture Health Score
- Zero circular dependencies across 1,303 source files
- 100% layer compliance verified
- 2009/2009 tests passing
- Foundation validated for research integration

**Unsafe Documentation (Sessions 2-7)**:
- 64/116 unsafe blocks documented (55.2% coverage)
- SAFETY/INVARIANTS/ALTERNATIVES/PERFORMANCE template established
- SIMD (AVX-512, AVX2, NEON) and arena allocators documented
- Critical FieldArena unsoundness identified and documented

**Implementation Status**:
- FDTD solver: 2nd/4th/6th/8th order accuracy options
- PSTD solver: Spectral accuracy with FFT
- DG solver: Shock capturing for nonlinear acoustics
- CPML boundaries: Roden & Gedney 2000 implementation
- Existing validation: Internal analytical solutions only (no external comparison)

### k-Wave Overview

**k-Wave**: Open-source MATLAB toolbox for acoustic and elastic wave propagation simulation
- **Authors**: Bradley Treeby, Ben Cox (University College London)
- **Citations**: 1000+ publications, clinical acceptance
- **Core Methods**: k-space pseudospectral time domain (k-space PSTD)
- **Key Features**: 
  - Perfectly matched layers (PML) for boundaries
  - Nonlinear acoustics (Westervelt, KZK)
  - Heterogeneous and absorbing media
  - Advanced source/sensor configurations
  - Extensive validation against analytical solutions

**kwavepython**: Python bindings for k-Wave via MATLAB Engine API
- Provides programmatic access to k-Wave from Python
- Enables automated comparison workflows
- Reference implementation for validation

---

## Mathematical Specifications

### 1. Linear Acoustic Wave Equation (k-Wave Core)

**Governing Equations** (Treeby & Cox 2010):
```
∂p/∂t + ρ₀c₀²∇·u = 0           (1) Pressure evolution
∂u/∂t + (1/ρ₀)∇p = 0            (2) Velocity evolution
```

Where:
- `p(x,t)` = acoustic pressure [Pa]
- `u(x,t)` = particle velocity [m/s]
- `ρ₀` = ambient density [kg/m³]
- `c₀` = sound speed [m/s]

**k-Space Operator** (k-Wave's key innovation):
```
∇ → ik    (Fourier space representation)
```

Transforms to:
```
∂p̂/∂t + ρ₀c₀²(ik)·û = 0
∂û/∂t + (1/ρ₀)(ik)p̂ = 0
```

**Exact Dispersion**: k-space method eliminates numerical dispersion by computing spatial derivatives exactly in Fourier space.

### 2. Absorbing Media (Power Law)

**Frequency-Dependent Absorption** (Szabo 1994):
```
α(ω) = α₀|ω|^y                   (3) Absorption coefficient

α₀ = absorption coefficient [Np/(m·MHz^y)]
y = power law exponent (typically 1-2)
```

**k-Wave Implementation** (Treeby et al. 2010):
```
∂p/∂t + ρ₀c₀²∇·u = -L{p}        (4) With absorption operator

L{·} = causal absorption operator (convolutional form)
```

### 3. Nonlinear Acoustics (Westervelt Equation)

**Full Westervelt** (Westervelt 1963):
```
∇²p - (1/c₀²)∂²p/∂t² = (β/ρ₀c₀⁴)∂²p²/∂t² + (δ/c₀⁴)∂³p/∂t³    (5)

β = coefficient of nonlinearity (B/A parameter)
δ = diffusivity of sound
```

**k-Wave's k-space Nonlinear** (Treeby & Cox 2010):
```
∂p/∂t + ρ₀c₀²∇·u = -βρ₀c₀²(u·∇)u + L{p}    (6) k-space nonlinear form
```

### 4. Perfectly Matched Layer (PML)

**CPML Formulation** (Roden & Gedney 2000):
```
∂ψ/∂t + σψ = α∂u/∂x             (7) Auxiliary field evolution
∂u/∂t = (1/ρ₀)∂p/∂x + ψ          (8) Velocity with PML term

σ(x) = σ_max(x/d)^n              (9) PML conductivity profile
```

**k-Wave Parameters**:
- `PMLSize`: Number of PML grid points (typically 10-20)
- `PMLAlpha`: PML frequency scaling factor (default 2.0)

---

## Implementation Comparison Matrix

### Feature Parity Analysis

| Feature | k-Wave | kwavers | Status | Priority |
|---------|--------|---------|--------|----------|
| **Core Methods** |
| k-space PSTD | ✅ Yes | ✅ Yes (PSTD module) | VERIFY | P0 |
| FDTD | ❌ No | ✅ Yes (2/4/6/8th order) | ADVANTAGE | - |
| DG | ❌ No | ✅ Yes (shock capturing) | ADVANTAGE | - |
| **Boundaries** |
| PML (CPML) | ✅ Yes | ✅ Yes (Roden & Gedney) | VERIFY | P0 |
| Periodic | ✅ Yes | ❓ Unknown | GAP | P1 |
| Dirichlet | ✅ Yes | ✅ Yes | VERIFY | P1 |
| **Physics** |
| Linear acoustics | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Nonlinear (Westervelt) | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Absorption (power law) | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Heterogeneous media | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Anisotropic media | ❌ No | ✅ Yes | ADVANTAGE | - |
| **Sources** |
| Point source | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Plane wave | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Focused | ✅ Yes | ✅ Yes | VERIFY | P1 |
| Transducer arrays | ✅ Yes | ✅ Yes | VERIFY | P1 |
| **Sensors** |
| Grid sensors | ✅ Yes | ✅ Yes | VERIFY | P0 |
| Point sensors | ✅ Yes | ❓ Unknown | GAP | P1 |
| Line sensors | ✅ Yes | ❓ Unknown | GAP | P2 |
| **Advanced** |
| Elastic waves | ✅ Yes | ✅ Yes | VERIFY | P1 |
| Thermal coupling | ❌ Limited | ✅ Yes | ADVANTAGE | - |
| GPU acceleration | ✅ Yes (CUDA) | ✅ Yes (wgpu) | VERIFY | P2 |

**Legend**:
- ✅ Yes: Feature implemented
- ❌ No: Feature not implemented
- ❓ Unknown: Implementation status unclear
- VERIFY: Requires correctness validation
- GAP: Missing feature
- ADVANTAGE: kwavers exceeds k-Wave capabilities

---

## Validation Test Suite Design

### Analytical Solutions (Exact Comparisons)

#### Test 1: Plane Wave Propagation
**Mathematical Specification**:
```
p(x,t) = A sin(k·x - ωt)         (10) Plane wave solution

k = ω/c₀ = 2πf/c₀                (11) Wave number
ω = 2πf                          (12) Angular frequency
```

**Validation Metrics**:
- Phase velocity error: `|c_numerical - c₀|/c₀`
- Amplitude decay: `|A_final/A_initial - 1|`
- Dispersion error: `∫|p_numerical - p_analytical|²dx / ∫|p_analytical|²dx`

**Acceptance Criteria**:
- Phase velocity error < 0.1% (k-Wave baseline)
- Amplitude preservation > 99.9% (no unphysical dissipation)
- L2 dispersion error < 0.01

#### Test 2: Gaussian Beam Propagation
**Mathematical Specification** (Goodman 2005):
```
p(r,z,t) = A₀(w₀/w(z))exp(-r²/w(z)²)exp(i(kz - ωt + φ(z)))    (13)

w(z) = w₀√(1 + (z/z_R)²)         (14) Beam width
z_R = πw₀²/λ                     (15) Rayleigh range
φ(z) = arctan(z/z_R)             (16) Gouy phase
```

**Validation Metrics**:
- Beam width error at z = z_R: `|w_numerical - w_analytical|/w_analytical`
- Focal intensity: `I_numerical/I_analytical` at focal point
- On-axis phase error: `|φ_numerical - φ_analytical|`

**Acceptance Criteria**:
- Beam width error < 1% at Rayleigh range
- Focal intensity within 95-105% of analytical
- Phase error < π/20 radians

#### Test 3: Spherical Wave
**Mathematical Specification**:
```
p(r,t) = (A/r)sin(kr - ωt)       (17) Spherical wave

r = |x - x₀|                     (18) Distance from source
```

**Validation Metrics**:
- Geometric spreading: `p(r)·r / A`
- Wavefront curvature: `∇²p / k²p`
- Energy conservation: `∫p²r²dΩ` over spherical shells

**Acceptance Criteria**:
- Geometric spreading error < 1%
- Wavefront curvature error < 2%
- Energy conservation within 99% (accounting for boundaries)

#### Test 4: Standing Wave (Resonance)
**Mathematical Specification**:
```
p(x,t) = 2A sin(kx)cos(ωt)       (19) Standing wave

Boundary conditions: p(0,t) = p(L,t) = 0    (20)
Resonance: k = nπ/L, n ∈ ℕ       (21)
```

**Validation Metrics**:
- Node locations: `|x_node,numerical - x_node,analytical|`
- Antinode amplitudes: `A_numerical/A_analytical`
- Temporal phase: `cos(ωt_numerical) - cos(ωt_analytical)`

**Acceptance Criteria**:
- Node location error < λ/100
- Antinode amplitude within 98-102%
- Temporal phase error < π/50

#### Test 5: Acoustic Absorption Decay
**Mathematical Specification** (Beer-Lambert Law):
```
p(x) = p₀ exp(-αx)               (22) Exponential decay

α = α₀f^y                        (23) Power law absorption
```

**Validation Metrics**:
- Absorption coefficient: `α_numerical` vs `α_analytical`
- Decay rate: `d(ln p)/dx`
- Total attenuation: `20 log₁₀(p_final/p_initial)` [dB]

**Acceptance Criteria**:
- Absorption coefficient error < 2%
- Decay rate error < 1%
- Total attenuation within 0.5 dB

---

## Implementation Strategy

### Phase 1: Infrastructure Setup (Hours 0-2)

**Task 1.1**: Python Bridge to k-Wave/kwavepython
- **Action**: Create `kwavers/scripts/kwave_bridge.py`
- **Requirements**:
  - MATLAB Engine API integration (kwavepython wrapper)
  - Data exchange: NumPy ↔ MATLAB arrays
  - Configuration: Grid, medium, source, sensor mapping
  - Error handling: MATLAB exceptions, licensing checks
- **Deliverable**: Working Python script that runs k-Wave simulations
- **Verification**: Run k-Wave example_ivp_homogeneous_medium.m

**Task 1.2**: Rust-Python Data Bridge
- **Action**: Create `kwavers/scripts/rust_kwave_comparison.py`
- **Requirements**:
  - Load kwavers simulation results (HDF5/NPY format)
  - Run equivalent k-Wave simulation
  - Compute comparison metrics (L2 error, phase error, etc.)
  - Generate comparison plots (matplotlib)
- **Deliverable**: Automated comparison pipeline
- **Verification**: Compare trivial case (zero initial conditions)

**Task 1.3**: Analytical Solutions Module
- **Action**: Create `kwavers/src/solver/validation/kwave_comparison/analytical.rs`
- **Requirements**:
  - Implement Eqs (10-23) as Rust functions
  - Grid-based evaluation of analytical solutions
  - Error metric computation (L2, Linf, phase, amplitude)
  - Literature references in doc comments
- **Deliverable**: Analytical solution library
- **Verification**: Unit tests for each analytical solution

### Phase 2: Test Case Implementation (Hours 2-4)

**Task 2.1**: Plane Wave Test (Test 1)
- **Action**: Create `tests/validation/kwave_plane_wave.rs`
- **Setup**:
  - Grid: 128×128×128, dx=0.5mm
  - Medium: Water (c₀=1500 m/s, ρ₀=1000 kg/m³)
  - Source: Plane wave, f=1MHz, A=100kPa
  - Time: 1000 steps, dt=50ns (CFL=0.3)
- **Comparison**:
  - kwavers PSTD vs k-Wave k-space PSTD
  - Phase velocity, amplitude, dispersion
- **Deliverable**: Passing test with documented comparison

**Task 2.2**: Gaussian Beam Test (Test 2)
- **Action**: Create `tests/validation/kwave_gaussian_beam.rs`
- **Setup**:
  - Grid: 256×256×128, dx=0.25mm
  - Medium: Water
  - Source: Gaussian beam, f=2MHz, w₀=5mm
  - Time: Propagate to z=50mm (2× Rayleigh range)
- **Comparison**:
  - Beam width evolution
  - Focal intensity
  - Gouy phase shift
- **Deliverable**: Passing test with beam profile plots

**Task 2.3**: Spherical Wave Test (Test 3)
- **Action**: Create `tests/validation/kwave_spherical_wave.rs`
- **Setup**:
  - Grid: 128×128×128, dx=0.5mm (cubic domain)
  - Medium: Water
  - Source: Point source at center, f=1MHz
  - Time: Propagate to steady state
- **Comparison**:
  - Geometric spreading (1/r decay)
  - Wavefront curvature
  - Energy conservation
- **Deliverable**: Passing test with radial profiles

**Task 2.4**: Standing Wave Test (Test 4)
- **Action**: Create `tests/validation/kwave_standing_wave.rs`
- **Setup**:
  - Grid: 256×1×1, dx=0.25mm (1D cavity)
  - Medium: Air (c₀=343 m/s)
  - Boundary: Dirichlet (p=0) at both ends
  - Source: Initial Gaussian pulse
  - Time: 5000 steps to reach steady state
- **Comparison**:
  - Node/antinode locations
  - Modal frequencies
  - Temporal phase
- **Deliverable**: Passing test with mode analysis

**Task 2.5**: Absorption Test (Test 5)
- **Action**: Create `tests/validation/kwave_absorption.rs`
- **Setup**:
  - Grid: 512×1×1, dx=0.1mm (long 1D domain)
  - Medium: Soft tissue (α₀=0.5 dB/cm/MHz, y=1.5)
  - Source: Continuous wave, f=1MHz
  - Time: 2000 steps to steady state
- **Comparison**:
  - Absorption coefficient extraction
  - Exponential decay rate
  - Total attenuation [dB]
- **Deliverable**: Passing test with decay curves

### Phase 3: Performance Benchmarking (Hours 4-5)

**Task 3.1**: Grid Size Scaling Study
- **Action**: Run Tests 1-5 with varying grid sizes
- **Grid Sizes**: 32³, 64³, 128³, 256³
- **Metrics**:
  - Execution time (kwavers vs k-Wave)
  - Memory usage
  - Accuracy degradation with coarser grids
- **Deliverable**: Scaling plots and performance table

**Task 3.2**: Method Comparison (FDTD vs PSTD vs k-space)
- **Action**: Run plane wave test with all methods
- **Methods**:
  - kwavers FDTD (4th order)
  - kwavers PSTD
  - k-Wave k-space PSTD
- **Metrics**:
  - Dispersion error vs points-per-wavelength
  - CFL stability limits
  - Computational efficiency
- **Deliverable**: Method comparison table

**Task 3.3**: Nonlinear Acoustics Benchmark
- **Action**: Run Westervelt equation test
- **Setup**:
  - High-amplitude source (p₀=1MPa)
  - Track harmonic generation
  - Compare kwavers Westervelt solver vs k-Wave
- **Metrics**:
  - Second harmonic amplitude
  - Shock formation distance
  - Energy transfer to harmonics
- **Deliverable**: Nonlinear validation report

### Phase 4: Gap Analysis and Documentation (Hours 5-6)

**Task 4.1**: Feature Gap Documentation
- **Action**: Create `docs/kwave_comparison_gaps.md`
- **Content**:
  - Missing features (from comparison matrix)
  - Correctness issues identified
  - Performance bottlenecks
  - Implementation priorities
- **Deliverable**: Prioritized gap analysis

**Task 4.2**: Mathematical Verification Report
- **Action**: Create `docs/kwave_mathematical_verification.md`
- **Content**:
  - Analytical solution comparisons (Tests 1-5)
  - Error metrics and acceptance criteria
  - Physics validation (conservation laws, dispersion, stability)
  - Literature references for all equations
- **Deliverable**: Formal verification document

**Task 4.3**: Session Documentation
- **Action**: Complete `SPRINT_217_SESSION_8_PROGRESS.md`
- **Content**:
  - All test results
  - Comparison plots
  - Gap analysis
  - Next session priorities
- **Deliverable**: Complete session record

---

## Acceptance Criteria

### Must Have (Blocking for Session Completion)
1. ✅ **Working k-Wave Bridge**: Python script successfully runs k-Wave simulations
2. ✅ **3+ Analytical Tests Passing**: Tests 1-3 implemented and passing
3. ✅ **Mathematical Specifications**: All equations (10-23) documented with references
4. ✅ **Gap Analysis**: Feature comparison matrix complete
5. ✅ **Zero Regressions**: 2009/2009 tests passing, zero compilation errors

### Should Have (High Priority)
6. ✅ **5 Analytical Tests Complete**: All Tests 1-5 implemented
7. ✅ **Performance Benchmarks**: Grid scaling and method comparison
8. ✅ **Comparison Plots**: Visual validation for all test cases
9. ✅ **Documentation**: Mathematical verification report

### Nice to Have (Future Sessions)
10. ⏳ **Nonlinear Validation**: Westervelt equation comparison
11. ⏳ **3D Heterogeneous Media**: Complex tissue models
12. ⏳ **Clinical Test Cases**: Ultrasound imaging scenarios
13. ⏳ **GPU Performance**: CUDA vs wgpu comparison

---

## Risk Assessment

### Technical Risks

**Risk 1: MATLAB Licensing / kwavepython Availability**
- **Impact**: Cannot run k-Wave comparisons
- **Probability**: Medium (depends on environment)
- **Mitigation**: 
  - Use pre-computed k-Wave reference data
  - Implement k-Wave algorithms directly in Rust (from papers)
  - Focus on analytical solutions (no k-Wave required)

**Risk 2: Numerical Differences Due to Implementation Details**
- **Impact**: False positive "errors" from different discretization choices
- **Probability**: High (expected for complex physics)
- **Mitigation**:
  - Compare against analytical solutions first (ground truth)
  - Document implementation differences (e.g., PML parameters)
  - Accept small numerical differences (<1%) as tolerable

**Risk 3: Performance Bottlenecks in Python Bridge**
- **Impact**: Slow comparisons, long session duration
- **Probability**: Medium
- **Mitigation**:
  - Use small test cases for initial validation
  - Parallelize independent comparisons
  - Cache k-Wave results for repeated tests

### Schedule Risks

**Risk 4: Scope Creep (Too Many Test Cases)**
- **Impact**: Session extends beyond 6 hours
- **Probability**: High (enthusiasm for validation)
- **Mitigation**:
  - Strict prioritization (Tests 1-3 minimum, 4-5 stretch goals)
  - Time-box each task (max 1 hour per test)
  - Defer advanced tests to Session 9

**Risk 5: Unexpected Correctness Issues**
- **Impact**: Major bugs discovered, require fixing before validation
- **Probability**: Medium (untested code paths)
- **Mitigation**:
  - Document issues as gaps, don't fix immediately
  - Create GitHub issues for follow-up
  - Focus on validation, not implementation

---

## Dependencies and Prerequisites

### Software Requirements
- ✅ Rust toolchain (installed)
- ✅ Python 3.8+ with NumPy, SciPy, Matplotlib
- ❓ MATLAB R2020b+ with k-Wave toolbox (optional)
- ❓ kwavepython package (optional)

### Knowledge Requirements
- ✅ Acoustic wave equations (team expertise)
- ✅ Numerical methods (FDTD, PSTD, spectral methods)
- ✅ Analytical solutions for wave equations
- ⏳ k-Wave implementation details (learn as needed)

### Input Artifacts
- ✅ Sprint 217 Session 1-7 results
- ✅ Current solver implementations (FDTD, PSTD, DG)
- ✅ PRD research integration goals
- ⏳ k-Wave documentation and papers

---

## Success Metrics

### Quantitative Metrics
1. **Test Coverage**: ≥3 analytical tests implemented (target: 5)
2. **Accuracy**: All tests meet acceptance criteria (L2 error < 0.01, phase error < 0.1%, etc.)
3. **Performance**: Document kwavers vs k-Wave runtime (no specific target, information gathering)
4. **Gap Count**: Identify and document ≥5 feature gaps
5. **Regression Freedom**: Maintain 2009/2009 test pass rate

### Qualitative Metrics
1. **Mathematical Rigor**: All equations documented with literature references
2. **Reproducibility**: Comparison pipeline can be re-run by others
3. **Clarity**: Gap analysis provides clear implementation priorities
4. **Foundation**: Framework ready for ongoing validation in future sessions

---

## Next Session Preview (Session 9)

### Priorities
1. **Address Critical Gaps**: Implement top P0 features from gap analysis
2. **Advanced Physics**: Nonlinear acoustics, heterogeneous media validation
3. **Clinical Test Cases**: Ultrasound imaging scenarios
4. **GPU Validation**: Compare wgpu vs CUDA performance

### Prerequisites from Session 8
- Working k-Wave comparison pipeline
- Baseline analytical validation complete
- Gap analysis documented
- No blocking correctness issues

---

## References

### k-Wave Literature
1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields." *Journal of Biomedical Optics*, 15(2), 021314.
2. Treeby, B. E., Jaros, J., Rendell, A. P., & Cox, B. T. (2012). "Modeling nonlinear ultrasound propagation in heterogeneous media with power law absorption using a k-space pseudospectral method." *The Journal of the Acoustical Society of America*, 131(6), 4324-4336.
3. Roden, J. A., & Gedney, S. D. (2000). "Convolution PML (CPML): An efficient FDTD implementation of the CFS-PML for arbitrary media." *Microwave and optical technology letters*, 27(5), 334-339.

### Analytical Solutions
4. Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.). Roberts and Company Publishers. (Gaussian beam theory)
5. Szabo, T. L. (1994). "Time domain wave equations for lossy media obeying a frequency power law." *The Journal of the Acoustical Society of America*, 96(1), 491-500.
6. Westervelt, P. J. (1963). "Parametric acoustic array." *The Journal of the Acoustical Society of America*, 35(4), 535-537.

### Kwavers Documentation
7. `kwavers/README.md` - Architecture and capabilities
8. `kwavers/docs/PRD.md` - Product requirements and research integration goals
9. Sprint 217 Session 1-7 progress documents

---

**Document Version**: 1.0  
**Author**: Ryan Clanton (@ryancinsight)  
**Last Updated**: 2026-02-04  
**Status**: PLANNING - Ready for Session 8 Execution