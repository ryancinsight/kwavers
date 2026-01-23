# Reference Implementation Review: Research Integration Analysis for Kwavers

**Date**: 2026-01-23  
**Review Scope**: 11 leading ultrasound simulation and optimization repositories  
**Objective**: Identify features, algorithms, and best practices for kwavers integration  
**Reviewer**: Analysis of k-Wave, jwave, fullwave25, mSOUND, BabelBrain, and related projects

---

## Executive Summary

This comprehensive review analyzes 11 leading research repositories in ultrasound simulation, therapeutic planning, and optimization. The analysis identifies critical gaps in kwavers, prioritizes integration opportunities, and provides a roadmap for incorporating state-of-the-art methods.

### Critical Findings

| Repository | Primary Innovation | Integration Priority | Estimated Effort |
|-----------|-------------------|---------------------|------------------|
| **jwave** | JAX-based differentiable simulations | 🔴 P0 Critical | 120-160 hours |
| **k-Wave** | k-space pseudospectral method refinements | 🔴 P0 Critical | 80-120 hours |
| **BabelBrain** | MRI-guided HIFU planning workflow | 🔴 P0 Critical | 140-180 hours |
| **fullwave25** | Multi-GPU nonlinear FDTD | 🟡 P1 High | 100-140 hours |
| **mSOUND** | Mixed-domain methods (TMDM/FSMDM) | 🟡 P1 High | 80-100 hours |
| **k-wave-python** | Python API design patterns | 🟡 P1 High | 60-80 hours |
| **dbua** | Differentiable beamforming | 🟢 P2 Medium | 40-60 hours |
| **Sound Speed Estimation** | Spatial coherence optimization | 🟢 P2 Medium | 30-40 hours |
| **HITU Simulator** | Thermal dose calculation | 🟢 P2 Medium | 40-50 hours |
| **Kranion** | Transcranial planning visualization | 🟢 P3 Low | 80-120 hours |

**Total Integration Effort**: 750-1,050 hours (19-26 weeks at full-time equivalent)

---

## Repository-by-Repository Analysis

---

## 1. jwave (JAX-Based Differentiable Acoustic Simulation)

**Repository**: https://github.com/ucl-bug/jwave  
**Language**: Python/JAX  
**Citation**: arXiv 2207.01499  
**Stars**: 190+ (active development)

### Key Algorithms and Methods

1. **Differentiable Wave Propagation**
   - Full automatic differentiation through simulation
   - Gradient-based inverse problem solving
   - Physics-informed loss functions
   - End-to-end optimization pipelines

2. **JAX Compilation Strategy**
   - JIT compilation with XLA backend
   - Device-agnostic code (CPU/GPU/TPU)
   - Implicit parallelization via vectorization
   - Graph-level optimization

3. **Modular Simulation Architecture**
   - `Domain` + `Medium` + `TimeAxis` + Solver composition
   - Functional programming patterns
   - Immutable data structures
   - Pure functions for reliable differentiation

### Data Structures and Architecture

```python
# jwave architectural pattern
simulate_wave_propagation(
    medium: Medium,           # Material properties
    time_axis: TimeAxis,      # Temporal discretization (CFL-aware)
    domain: Domain,           # Geometry and boundaries
    initial_conditions: FourierSeries  # Spectral representation
) -> PressureFieldEvolution
```

**Key Design Principles**:
- **Composability**: Small, focused components
- **Functional purity**: No side effects, enables caching/memoization
- **Differentiability-first**: Every operation differentiable by default
- **Device abstraction**: Single codebase for all hardware

### Features Missing in Kwavers

1. **✅ Partial - Differentiable Simulations**
   - Kwavers has: Burn-based PINN infrastructure with autodiff
   - Missing: Full forward solver differentiation (FDTD, PSTD)
   - Gap: Cannot compute gradients through time-domain solvers
   - Impact: Inverse problems limited to PINN approaches

2. **❌ Missing - Fourier Series Initial Conditions**
   - jwave: Efficient spectral representation with automatic CFL handling
   - Kwavers: Direct spatial arrays only
   - Gap: No frequency-domain IC specification
   - Impact: Inefficient for bandlimited sources, manual frequency management

3. **❌ Missing - Device-Agnostic Compilation**
   - jwave: Single codebase compiles to CPU/GPU/TPU via XLA
   - Kwavers: Separate GPU kernels (WGPU shaders), conditional compilation
   - Gap: Code duplication across backends
   - Impact: Maintenance burden, feature parity challenges

4. **✅ Partial - Modular Functional Architecture**
   - Kwavers has: Clean Architecture with domain separation
   - Missing: Functional composition patterns, immutable simulation state
   - Gap: Imperative mutation-based state management
   - Impact: Harder to reason about, cache, or parallelize

### Best Practices to Adopt

1. **Functional Simulation Builder Pattern**
   ```rust
   // Proposed kwavers functional API (inspired by jwave)
   SimulationBuilder::new()
       .with_medium(Medium::homogeneous(c0, rho0))
       .with_domain(Domain::new(nx, ny, nz, dx, dy, dz))
       .with_time_axis(TimeAxis::from_cfl(cfl_number, t_end))
       .with_source(Source::gaussian(center, sigma))
       .build()?
       .run_differentiable()  // Returns gradients
   ```

2. **Automatic CFL Stability Management**
   - TimeAxis computes stable dt from grid spacing + sound speed
   - User specifies CFL number (0.1-0.5), library handles discretization
   - Eliminates manual stability analysis

3. **Spectral Initial Conditions**
   - FourierSeries representation for bandlimited sources
   - Automatic padding and windowing
   - Efficient for narrowband transducers

4. **Composable Physics Operators**
   - Small, pure functions for gradient, divergence, curl
   - Chain via function composition
   - Enables symbolic differentiation

### Integration Recommendations

**Priority 1 (P0 - Critical)**: Differentiable Forward Solvers
- **Scope**: Extend FDTD/PSTD solvers with Burn autodiff
- **Approach**: 
  1. Replace in-place array updates with functional transformations
  2. Implement StateTransition<T> trait for differentiable state evolution
  3. Integrate with Burn's backward pass
- **Effort**: 80-120 hours
- **Files**: `src/solver/forward/{fdtd,pstd}/`, `src/solver/inverse/pinn/`
- **Validation**: Gradient check against finite differences (relative error < 1e-6)
- **Benefit**: Full-waveform inversion, source localization, material parameter estimation

**Priority 2 (P1 - High)**: Device-Agnostic Backend Abstraction
- **Scope**: Unified solver trait with CPU/GPU/TPU implementations
- **Approach**:
  1. Define `ComputeBackend` trait with device discovery
  2. Implement `CPUBackend`, `GPUBackend` (WGPU), `TPUBackend` (future)
  3. Automatic dispatch based on hardware availability
- **Effort**: 40-60 hours
- **Files**: `src/solver/interface/backend.rs`, all solver implementations
- **Benefit**: Single algorithm implementation, automatic parallelization

**Priority 3 (P2 - Medium)**: Fourier Series Initial Conditions
- **Scope**: Spectral representation for bandlimited sources
- **Approach**:
  1. Create `FourierSeries` type with frequency + amplitude vectors
  2. Implement IFFT to spatial domain with automatic padding
  3. Integrate with existing `Source` trait
- **Effort**: 20-30 hours
- **Files**: `src/domain/source/spectral.rs`, `src/math/fft/`
- **Benefit**: Efficient transducer modeling, automatic frequency management

---

## 2. k-Wave (MATLAB Ultrasound Simulation Toolbox)

**Repository**: https://github.com/ucl-bug/k-wave  
**Language**: MATLAB (with C++/CUDA backends)  
**Citations**: 4 key papers (Treeby & Cox 2010, Treeby 2012, Wise 2019, Treeby 2020)

### Key Algorithms and Methods

1. **k-Space Pseudospectral Method**
   - **Spatial gradients**: Fourier collocation (spectral accuracy)
   - **Temporal gradients**: k-space corrected finite difference (exact for linear homogeneous)
   - **Advantage**: 4-8× fewer grid points than FDTD for same accuracy
   - **Dispersion**: Minimal numerical dispersion via k-space correction

2. **Fractional Laplacian Absorption**
   - Power law attenuation: α(f) = α₀ · f^γ
   - Implemented via linear integro-differential operator
   - Supports spatially-varying exponent γ(x,y,z)
   - Causal time-domain convolution

3. **Split-Field Perfectly Matched Layer (PML)**
   - Absorbs outgoing waves at domain boundaries
   - Reflection coefficient < -40 dB typical
   - Split-field formulation for stability
   - Supports complex geometries

4. **kWaveArray Class (Arbitrary Source/Sensor)**
   - Defines off-grid source/sensor positions
   - Spatial integration over element surfaces
   - Supports arbitrary array geometries
   - Efficient Fourier-space implementation

### Features Missing in Kwavers

1. **✅ Implemented - k-Space Pseudospectral (PSTD)**
   - Kwavers has: `src/solver/forward/pstd/` with k-space operators
   - Status: Core implementation complete
   - Gap: k-space temporal correction not fully validated
   - Note: See gap_audit.md Sprint 209 Phase 1 (pseudospectral derivatives complete)

2. **✅ Implemented - Fractional Laplacian Absorption**
   - Kwavers has: Power law absorption in medium models
   - Status: Implemented in `src/physics/acoustics/absorption/`
   - Gap: Spatially-varying exponent support needs validation

3. **✅ Implemented - CPML Boundaries**
   - Kwavers has: Convolutional PML (CPML) implementation
   - Status: `src/domain/boundary/cpml/` (Sprint 208 refactored)
   - Note: CPML is equivalent to split-field PML

4. **❌ Missing - kWaveArray Off-Grid Source/Sensor**
   - k-Wave: Arbitrary element positions with surface integration
   - Kwavers: Grid-aligned sources only
   - Gap: Cannot model realistic transducer element positions
   - Impact: Inaccurate near-field modeling, limited to rectilinear arrays

5. **❌ Missing - Axisymmetric Coordinate Support**
   - k-Wave: Specialized 2D axisymmetric solver (10-100× faster than 3D)
   - Kwavers: Full 3D only
   - Gap: Computational waste for cylindrically symmetric problems
   - Impact: Focused transducers, cylindrical phantoms inefficient

### Data Structures and Architecture

```matlab
% k-Wave simulation pattern
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
medium.sound_speed = c0;
medium.density = rho0;
medium.alpha_coeff = alpha0;
medium.alpha_power = gamma;

source.p_mask = source_mask;
source.p = source_signal;

sensor.mask = sensor_mask;

sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor);
```

**Key Design**: Flat namespace with struct-based configuration, clear separation of geometry/physics/sources/sensors.

### Best Practices to Adopt

1. **Off-Grid Source/Sensor Integration**
   - Distributed source model: integrate over element surface
   - Fourier-space implementation for efficiency
   - Exact integration via Gaussian quadrature

2. **Axisymmetric Solver Specialization**
   - Reduce 3D → 2D (r, z) for cylindrical symmetry
   - Radial FFT for azimuthal derivatives
   - 10-100× speedup for focused transducers

3. **Enhanced Temporal k-Space Correction**
   - Exact dispersion relation for linear propagation
   - Stability-preserving corrections for nonlinear terms
   - Adaptive time stepping

4. **Medium Property Validation**
   - Automatic stability checks (CFL, diffusion limits)
   - Physical range validation (c > 0, ρ > 0, α ≥ 0)
   - Warning system for unusual parameters

### Integration Recommendations

**Priority 1 (P0 - Critical)**: Off-Grid Source/Sensor (kWaveArray Equivalent)
- **Scope**: Arbitrary element positions with spatial integration
- **Approach**:
  1. Extend `Source` trait with `ElementArray` type
  2. Implement Gaussian quadrature surface integration
  3. Fourier-space distributed source computation
- **Effort**: 60-80 hours
- **Files**: `src/domain/source/array.rs`, `src/solver/forward/pstd/implementation/`
- **Validation**: Compare near-field patterns with k-Wave for phased array
- **Benefit**: Realistic transducer modeling, accurate near-field predictions

**Priority 2 (P1 - High)**: Axisymmetric Solver Specialization
- **Scope**: 2D axisymmetric PSTD solver for cylindrical symmetry
- **Approach**:
  1. Create `solver/forward/pstd/axisymmetric/` module
  2. Implement radial coordinate transformations
  3. Use Hankel transforms for radial derivatives
- **Effort**: 40-60 hours
- **Files**: `src/solver/forward/pstd/axisymmetric/`
- **Validation**: Compare bowl transducer focus with 3D solver
- **Benefit**: 10-100× speedup for focused transducers, therapeutic planning

**Priority 3 (P1 - High)**: Enhanced k-Space Temporal Correction
- **Scope**: Exact dispersion relation enforcement
- **Approach**:
  1. Implement exact phase velocity correction: ω/k vs c₀
  2. Add adaptive time stepping based on local CFL
  3. Validate dispersion relation preservation
- **Effort**: 20-30 hours
- **Files**: `src/solver/forward/pstd/implementation/core/stepper.rs`
- **Validation**: Plane wave propagation (dispersion error < 1e-10)
- **Benefit**: Spectral accuracy preservation, long-distance propagation

---

## 3. BabelBrain (MRI-Guided HIFU Planning)

**Repository**: https://github.com/ProteusMRIgHIFU/BabelBrain  
**Language**: Python  
**Focus**: Transcranial focused ultrasound treatment planning

### Key Algorithms and Methods

1. **Multi-Modal Medical Imaging Pipeline**
   - T1-weighted MRI for anatomical reference
   - CT/ZTE/PETRA for bone characterization
   - Automatic tissue segmentation (skull, scalp, brain)
   - Hash-based caching to avoid reprocessing

2. **Skull Acoustic Modeling**
   - CT Hounsfield Units → acoustic properties
   - Cortical/trabecular bone distinction (80% trabecular default)
   - Density maps for heterogeneous bone
   - Air region masking (critical for transcranial)

3. **Coordinate System Management**
   - Patient anatomical coordinates (MRI space)
   - Transducer mechanical coordinates (device space)
   - Automatic transformation matrix handling
   - 3DSlicer/Brainsight integration

4. **Three-Stage Planning Workflow**
   - **Stage 1**: Domain preparation (segmentation, voxelization, GPU filtering)
   - **Stage 2**: Acoustic field simulation with mechanical positioning
   - **Stage 3**: Thermal effect modeling (Pennes bioheat transfer)

### Data Structures and Architecture

**Workflow Architecture**:
```
Medical Images (DICOM/NIFTI)
  ↓ (Segmentation + Voxelization)
Tissue Masks + Acoustic Property Maps
  ↓ (BEM or FDTD Simulation)
Pressure Field (normalized, NIfTI)
  ↓ (Bioheat Transfer)
Temperature Map + Thermal Dose
  ↓ (Export to Neuronavigation)
Treatment Plan (Brainsight/3DSlicer)
```

### Features Missing in Kwavers

1. **❌ Missing - Medical Imaging Pipeline**
   - BabelBrain: Full DICOM/NIFTI loading, segmentation, coordinate transforms
   - Kwavers: Basic NIFTI/DICOM support, no automatic segmentation
   - Gap: Manual tissue mask creation, no HU→acoustic property conversion
   - Impact: Cannot process clinical imaging data for planning

2. **❌ Missing - Transcranial Skull Modeling**
   - BabelBrain: CT-based heterogeneous skull with cortical/trabecular distinction
   - Kwavers: Homogeneous skull models only (`src/physics/acoustics/skull/`)
   - Gap: No CT density mapping, no aberration correction
   - Impact: Inaccurate transcranial focus prediction (clinical safety issue)

3. **❌ Missing - Neuronavigation Integration**
   - BabelBrain: Direct export to Brainsight, 3DSlicer, coordinate alignment
   - Kwavers: Standalone simulation, no clinical workflow integration
   - Gap: Cannot integrate with surgical navigation systems
   - Impact: Research tool only, not clinically deployable

4. **❌ Missing - Multi-Stage Treatment Planning**
   - BabelBrain: Acoustic simulation → thermal model → dose calculation → safety limits
   - Kwavers: Separate acoustic/thermal solvers, no integrated planning workflow
   - Gap: Manual chaining of simulations, no safety validation
   - Impact: Inefficient clinical planning, no automated safety checks

5. **✅ Partial - Bioheat Transfer**
   - Kwavers has: Pennes equation solver (`src/solver/forward/thermal_diffusion/`)
   - Missing: Thermal dose calculation (CEM43), tissue damage models
   - Gap: No cumulative thermal dose tracking
   - Impact: Cannot assess therapeutic efficacy or safety margins

### Best Practices to Adopt

1. **Hash-Based Caching**
   - Avoid recomputing expensive segmentation/voxelization
   - Content-addressed storage for intermediate results
   - Automatic cache invalidation on input changes

2. **GPU-Accelerated Preprocessing**
   - Filtering and voxelization on GPU
   - 10-50× speedup for large medical images
   - Memory-efficient tiled processing

3. **Coordinate System Validation**
   - Automatic checks for transformation matrix validity
   - Visual overlays to verify alignment (MRI + CT)
   - Units and handedness checking

4. **Stage-Based Workflow with Checkpointing**
   - Save intermediate results (masks, pressure fields)
   - Enable iterative refinement (adjust focus without re-segmentation)
   - Parallel what-if analysis (multiple targets)

### Integration Recommendations

**Priority 1 (P0 - Critical)**: CT-Based Skull Modeling
- **Scope**: Hounsfield Unit → acoustic property conversion
- **Approach**:
  1. Implement HU→(c, ρ, α) piecewise linear maps (literature-based)
  2. Add cortical/trabecular bone classification (threshold-based)
  3. Create heterogeneous skull medium from CT volume
- **Effort**: 40-60 hours
- **Files**: `src/physics/acoustics/skull/ct_based.rs` (TODO already exists, see backlog.md)
- **Validation**: Compare aberration patterns with published transcranial data
- **Benefit**: Clinical-grade transcranial ultrasound planning

**Priority 2 (P0 - Critical)**: Medical Imaging Pipeline
- **Scope**: DICOM/NIFTI loading, automatic segmentation, coordinate transforms
- **Approach**:
  1. Enhance existing DICOM/NIFTI loaders with metadata parsing
  2. Integrate tissue segmentation (threshold-based or ML-based)
  3. Implement coordinate transformation infrastructure
- **Effort**: 60-80 hours
- **Files**: `src/clinical/imaging/preprocessing/`, `src/domain/medium/from_imaging.rs`
- **Validation**: Process BabelBrain test datasets, compare masks
- **Benefit**: Automated clinical workflow, no manual preprocessing

**Priority 3 (P0 - Critical)**: Integrated Treatment Planning Workflow
- **Scope**: End-to-end planning with acoustic + thermal + dose calculation
- **Approach**:
  1. Create `TreatmentPlanner` orchestrator
  2. Chain acoustic solver → bioheat transfer → thermal dose
  3. Implement safety validation (max temperature, max dose, skull heating)
- **Effort**: 40-60 hours
- **Files**: `src/clinical/therapy/planning/hifu_planner.rs`
- **Validation**: Reproduce BabelBrain example cases
- **Benefit**: Clinical-ready therapeutic planning

---

## 4. fullwave25 (Full-Wave Acoustic Simulation with GPU)

**Repository**: https://github.com/pinton-lab/fullwave25  
**Language**: C/CUDA  
**Focus**: High-order FDTD for nonlinear acoustics with multi-GPU scaling

### Key Algorithms and Methods

1. **High-Order Staggered-Grid FDTD**
   - **Spatial accuracy**: 8th order
   - **Temporal accuracy**: 4th order
   - **Staggering**: Pressure-velocity formulation (Yee-like)
   - **Benefit**: Minimal numerical dispersion, accurate phase velocity

2. **Heterogeneous Power-Law Attenuation**
   - α(x,y,z,f) = α₀(x,y,z) · f^γ(x,y,z)
   - Spatially-varying exponent γ(x,y,z)
   - Multiple relaxation mechanisms (frequency-dependent Q)
   - Causal convolution via stretched coordinates

3. **Multi-GPU Domain Decomposition**
   - Linear scaling with number of GPUs (tested up to 8 GPUs)
   - Depth-wise domain splitting for efficient memory transfer
   - Minimal halo exchange (ghost zone synchronization)
   - Supports very large 3D simulations (>> 1000³ grid points)

4. **Perfectly Matched Layers with Relaxation**
   - PML with frequency-dependent absorption
   - Compatible with multiple relaxation mechanisms
   - Reflection coefficient < -60 dB achievable

### Features Missing in Kwavers

1. **❌ Missing - High-Order Temporal Accuracy (4th order)**
   - fullwave25: 4th-order Runge-Kutta time integration
   - Kwavers: 2nd-order leapfrog (FDTD standard)
   - Gap: 16× fewer time steps needed for same accuracy
   - Impact: Long simulations computationally expensive

2. **❌ Missing - Spatially-Varying Attenuation Exponent**
   - fullwave25: γ(x,y,z) heterogeneous
   - Kwavers: Single exponent γ for entire domain
   - Gap: Cannot model tissue interfaces with different dispersion
   - Impact: Inaccurate multi-tissue simulations (liver/muscle/fat)

3. **❌ Missing - Multi-GPU Domain Decomposition**
   - fullwave25: Linear scaling across 8 GPUs
   - Kwavers: Single GPU only
   - Gap: Limited to GPU memory (~16-48 GB)
   - Impact: Cannot simulate large 3D volumes (> 500³ typical limit)

4. **✅ Partial - Staggered Grid FDTD**
   - Kwavers has: Staggered grid operator (`src/math/numerics/operators/differential/staggered_grid.rs`)
   - Status: Implemented with Yee scheme for electromagnetic
   - Gap: Not fully validated for high-order acoustic nonlinearity

### Best Practices to Adopt

1. **High-Order Time Integration**
   - 4th-order Runge-Kutta for time stepping
   - Adaptive time step control based on local CFL
   - Improved stability for nonlinear terms

2. **Spatially-Varying Attenuation Exponent**
   - Store γ(x,y,z) as 3D array
   - Tissue-specific dispersion relations
   - Critical for heterogeneous phantoms

3. **Efficient Multi-GPU Synchronization**
   - Overlap computation + communication
   - Asynchronous GPU streams
   - Pinned memory for fast host-device transfer

4. **Stretched-Coordinate Absorption**
   - Unified framework for attenuation + PML
   - Complex-valued coordinate stretching
   - Single implementation for all loss mechanisms

### Integration Recommendations

**Priority 1 (P1 - High)**: Spatially-Varying Attenuation Exponent
- **Scope**: γ(x,y,z) heterogeneous absorption
- **Approach**:
  1. Extend `Medium` trait to store `alpha_power: Array3<f64>`
  2. Modify absorption operators to use per-voxel exponent
  3. Update relaxation mechanism computation
- **Effort**: 20-30 hours
- **Files**: `src/domain/medium/`, `src/physics/acoustics/absorption/`
- **Validation**: Multi-layer phantom with γ₁ ≠ γ₂
- **Benefit**: Accurate heterogeneous tissue modeling

**Priority 2 (P1 - High)**: 4th-Order Time Integration
- **Scope**: Runge-Kutta 4 time stepper for FDTD
- **Approach**:
  1. Implement RK4 integrator in `src/solver/integration/time_integration/`
  2. Add adaptive time stepping based on local nonlinearity
  3. Benchmark vs 2nd-order leapfrog
- **Effort**: 30-40 hours
- **Files**: `src/solver/integration/time_integration/rk4.rs`
- **Validation**: Convergence study (4th order temporal accuracy)
- **Benefit**: 4-16× faster for long simulations

**Priority 3 (P1 - High)**: Multi-GPU Domain Decomposition
- **Scope**: Linear scaling across multiple GPUs
- **Approach**:
  1. Implement domain decomposition along Z-axis
  2. Add GPU-to-GPU halo exchange (CUDA peer access or NCCL)
  3. Overlapped computation + communication
- **Effort**: 50-70 hours
- **Files**: `src/gpu/multi_device/`, `src/solver/forward/fdtd/gpu.rs`
- **Validation**: Weak scaling study (constant work per GPU)
- **Benefit**: 10-100× larger simulations, clinical 3D volumes

---

## 5. mSOUND (Multi-Domain Acoustic Simulation)

**Repository**: https://github.com/m-SOUND/mSOUND  
**Language**: MATLAB  
**Focus**: Mixed time-frequency domain methods

### Key Algorithms and Methods

1. **Transient Mixed-Domain Method (TMDM)**
   - **Time domain**: Pulsed-wave simulation
   - **Frequency domain**: Spatial propagation per frequency component
   - **Benefit**: Efficient for broadband transient sources
   - **Handles**: Arbitrary nonlinearity (no weak nonlinearity assumption)

2. **Frequency-Specific Mixed-Domain Method (FSMDM)**
   - **Steady-state**: Direct frequency-domain solution
   - **Harmonics**: Computes fundamental + 2nd harmonic simultaneously
   - **Limitation**: Linear or weakly nonlinear only
   - **Benefit**: 10-100× faster than TMDM for CW sources

3. **Generalized Westervelt Equation**
   - Spatially-varying c(x,y,z), ρ(x,y,z), B/A(x,y,z)
   - Arbitrary heterogeneity (no homogeneous medium assumption)
   - Power-law attenuation α(x,y,z,f)
   - Optional reflection at interfaces

4. **Hybrid Time-Frequency Approach**
   - FFT transforms between domains
   - Time-domain nonlinearity, frequency-domain attenuation
   - Adaptive domain selection per operator

### Features Missing in Kwavers

1. **❌ Missing - Transient Mixed-Domain Method (TMDM)**
   - mSOUND: Hybrid time-frequency with per-frequency propagation
   - Kwavers: Pure time-domain (FDTD) or pure frequency-domain (HAS)
   - Gap: No hybrid approach combining advantages
   - Impact: Inefficient for broadband sources in attenuating media

2. **❌ Missing - Frequency-Specific Mixed-Domain Method (FSMDM)**
   - mSOUND: Direct steady-state solution for CW sources
   - Kwavers: Must time-step to steady state
   - Gap: 10-100× computational waste for CW problems
   - Impact: Slow therapeutic ultrasound simulations (MHz CW)

3. **❌ Missing - Interface Reflection Option**
   - mSOUND: Toggle reflection on/off at material interfaces
   - Kwavers: Always includes reflections (no option to disable)
   - Gap: Cannot isolate transmission-only effects
   - Impact: Difficult to separate forward/backward waves

4. **✅ Partial - Westervelt Equation**
   - Kwavers has: Westervelt solver (`src/solver/forward/nonlinear/westervelt_spectral/`)
   - Status: Spectral implementation exists
   - Gap: Not integrated with mixed-domain approach

### Best Practices to Adopt

1. **Problem-Specific Domain Selection**
   - Pulsed waves → TMDM (transient)
   - Continuous waves → FSMDM (steady-state)
   - User specifies source type, library selects method
   - Automatic performance optimization

2. **Configurable Interface Physics**
   - Option to include/exclude reflections
   - Useful for debugging and validation
   - Enables comparative studies (with/without reflections)

3. **Efficient Harmonic Generation**
   - FSMDM computes fundamental + harmonics simultaneously
   - Avoids iterative nonlinear solve
   - Useful for contrast imaging (2nd harmonic)

4. **Adaptive Domain Switching**
   - Automatically switch time ↔ frequency based on operation
   - Nonlinearity in time, attenuation in frequency
   - Optimal efficiency per operator

### Integration Recommendations

**Priority 1 (P1 - High)**: Transient Mixed-Domain Method (TMDM)
- **Scope**: Hybrid time-frequency for broadband pulsed sources
- **Approach**:
  1. Create `solver/forward/mixed_domain/tmdm.rs` module
  2. Implement per-frequency propagation via angular spectrum
  3. Time-domain nonlinearity via Westervelt operator
- **Effort**: 50-70 hours
- **Files**: `src/solver/forward/mixed_domain/`
- **Validation**: Compare with full time-domain FDTD (should match exactly)
- **Benefit**: 5-10× speedup for attenuating broadband simulations

**Priority 2 (P1 - High)**: Frequency-Specific Mixed-Domain Method (FSMDM)
- **Scope**: Direct steady-state solution for CW sources
- **Approach**:
  1. Implement frequency-domain Helmholtz solver with nonlinear iteration
  2. Add 2nd harmonic generation via perturbation theory
  3. Integrate with existing angular spectrum methods
- **Effort**: 30-50 hours
- **Files**: `src/solver/forward/mixed_domain/fsmdm.rs`
- **Validation**: CW focused transducer (compare with time-domain at steady state)
- **Benefit**: 10-100× faster HIFU simulations

**Priority 3 (P2 - Medium)**: Configurable Interface Reflections
- **Scope**: Optional reflection at material boundaries
- **Approach**:
  1. Add `include_reflections: bool` to solver configuration
  2. Modify boundary condition application to skip reflection terms
  3. Add transmission-only mode for debugging
- **Effort**: 10-20 hours
- **Files**: `src/domain/boundary/coupling.rs`, solver modules
- **Validation**: Two-layer medium (compare with/without reflections)
- **Benefit**: Simplified validation, educational tool

---

## 6. k-wave-python (Python Interface Patterns)

**Repository**: https://k-wave-python.readthedocs.io  
**Language**: Python wrapper for k-Wave  
**Focus**: Pythonic API design, modern visualization

### Key Design Patterns

1. **Object-Oriented Configuration**
   ```python
   grid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
   medium = kWaveMedium(sound_speed=c0, density=rho0, alpha_coeff=alpha0)
   source = kSource()
   source.p_mask = source_mask
   source.p = source_signal
   sensor = kSensor(mask=sensor_mask)
   ```

2. **Modular Reconstruction Pipeline**
   - Beamforming algorithms (DAS, DMAS, SLSC)
   - Unit conversion utilities (pressure → intensity → dB)
   - Shifted Fourier transforms for efficiency
   - Post-processing workflow automation

3. **Specialized Transducer Class**
   - `kTransducer` for common array geometries
   - Built-in library of clinical transducers
   - Automatic beamforming parameter calculation

4. **Four-Component Simulation Pattern**
   - Every simulation requires: Grid, Medium, Source, Sensor
   - Explicit separation of geometry, physics, excitation, detection
   - Clear mental model for users

### Best Practices to Adopt

1. **Builder Pattern for Complex Objects**
   ```rust
   // Proposed kwavers API (inspired by k-wave-python)
   let simulation = Simulation::builder()
       .grid(Grid::new(nx, ny, nz, dx, dy, dz)?)
       .medium(Medium::homogeneous(c0, rho0, alpha0))
       .source(Source::phased_array(center, num_elements, pitch))
       .sensor(Sensor::plane(position, normal))
       .build()?;
   ```

2. **Transducer Library**
   - Pre-defined clinical transducers (L7-4, C5-2, etc.)
   - Automatic element positioning and delays
   - Validated beamforming parameters

3. **Post-Processing Pipeline**
   - Chained operations: sensor_data → beamform → envelope → log_compress
   - Lazy evaluation for memory efficiency
   - Automatic unit handling

4. **Explicit Four-Component Structure**
   - Enforce Grid + Medium + Source + Sensor pattern
   - Type system ensures completeness
   - Clear documentation mapping physical concepts to code

### Integration Recommendations

**Priority 1 (P1 - High)**: Builder Pattern Simulation API
- **Scope**: Ergonomic simulation construction with validation
- **Approach**:
  1. Create `SimulationBuilder` with fluent interface
  2. Add compile-time completeness checking (typestate pattern)
  3. Automatic validation and defaults
- **Effort**: 20-30 hours
- **Files**: `src/simulation/builder.rs`
- **Validation**: Example gallery reproducing k-wave-python tutorials
- **Benefit**: Lower learning curve, fewer user errors

**Priority 2 (P1 - High)**: Clinical Transducer Library
- **Scope**: Pre-defined transducers with validated parameters
- **Approach**:
  1. Curate database of clinical transducers (manufacturer specs)
  2. Implement TransducerLibrary with lookup by model number
  3. Automatic geometry + beamforming configuration
- **Effort**: 40-50 hours
- **Files**: `src/domain/source/transducers/clinical_library.rs`
- **Validation**: Compare beamplots with manufacturer datasheets
- **Benefit**: Rapid prototyping, clinical relevance

**Priority 3 (P2 - Medium)**: Post-Processing Pipeline
- **Scope**: Chained beamforming → envelope → compression operations
- **Approach**:
  1. Implement ProcessingPipeline trait with lazy evaluation
  2. Add common operations (envelope_detection, log_compression, TGC)
  3. Integrate with visualization
- **Effort**: 20-30 hours
- **Files**: `src/analysis/signal_processing/pipeline.rs`
- **Validation**: Reproduce ultrasound B-mode images
- **Benefit**: User-friendly imaging workflows

---

## 7. Sound Speed Estimation (Coherence-Based Optimization)

**Repository**: https://github.com/JiaxinZHANG97/Sound-Speed-Estimation  
**Language**: MATLAB  
**Focus**: Spatial coherence optimization for adaptive beamforming

### Key Algorithms

1. **Short-Lag Spatial Coherence (SLSC)**
   - Computes correlation between neighboring array elements
   - Maximum lag: 0-30% of transmit aperture
   - Coherent targets → high SLSC (maximize)
   - Incoherent targets → low SLSC (minimize via histogram filtering)

2. **Multi-Parameter Sound Speed Search**
   - Grid search over sound speed candidates (1400-1600 m/s typical)
   - Variable lag ranges (optimize for signal type)
   - Statistical filtering (mode detection across parameters)
   - Robust to noise via histogram-based mode extraction

3. **ROI-Based Metric Extraction**
   - User-selected regions of interest
   - Per-ROI sound speed optimization
   - Spatially-varying sound speed map generation

4. **Quality Metrics**
   - Contrast, CNR, gCNR (generalized CNR)
   - FWHM (resolution)
   - Speckle SNR
   - Multi-objective optimization across metrics

### Features Missing in Kwavers

1. **❌ Missing - Spatial Coherence Beamforming**
   - Sound Speed Repo: SLSC beamforming with adaptive weighting
   - Kwavers: DAS, MVDR, MUSIC (no coherence-based methods)
   - Gap: Cannot leverage spatial coherence for sound speed correction
   - Impact: Suboptimal image quality in heterogeneous tissues

2. **❌ Missing - Adaptive Sound Speed Estimation**
   - Sound Speed Repo: Automatic sound speed map generation
   - Kwavers: Fixed sound speed assumption
   - Gap: Manual parameter tuning required
   - Impact: Image artifacts in fatty tissues, aberration

3. **❌ Missing - Multi-Metric Image Quality Assessment**
   - Sound Speed Repo: Comprehensive quality metrics (CNR, gCNR, FWHM, speckle SNR)
   - Kwavers: Basic metrics only
   - Gap: Cannot objectively optimize imaging parameters
   - Impact: Subjective image quality assessment, non-reproducible

### Integration Recommendations

**Priority 1 (P2 - Medium)**: Short-Lag Spatial Coherence (SLSC) Beamforming
- **Scope**: Coherence-based beamforming for improved contrast
- **Approach**:
  1. Implement SLSC metric computation from channel data
  2. Add adaptive lag selection based on depth
  3. Integrate with existing beamforming infrastructure
- **Effort**: 20-30 hours
- **Files**: `src/analysis/signal_processing/beamforming/coherence.rs`
- **Validation**: Phantom experiment (cyst contrast vs DAS)
- **Benefit**: Improved lesion contrast (3-6 dB typical)

**Priority 2 (P2 - Medium)**: Adaptive Sound Speed Estimation
- **Scope**: Automatic spatially-varying sound speed map
- **Approach**:
  1. Implement grid search over sound speed candidates
  2. Use SLSC as optimization criterion (maximize or minimize per tissue)
  3. Generate 2D/3D sound speed map for adaptive beamforming
- **Effort**: 10-20 hours
- **Files**: `src/analysis/signal_processing/adaptive/sound_speed.rs`
- **Validation**: Multi-layer phantom (known c₁, c₂)
- **Benefit**: Automatic aberration correction, no manual tuning

---

## 8. HITU Simulator (HIFU Therapeutic Simulation)

**Repository**: https://github.com/jsoneson/HITU_Simulator  
**Language**: MATLAB  
**Focus**: Continuous-wave axisymmetric beams with thermal dose

### Key Algorithms

1. **Axisymmetric Beam Propagation**
   - WAKZK models: Wide-angle parabolic equation
   - Gaussian and planar source variants
   - Efficient 2D (r, z) computation for 3D cylindrical symmetry
   - Nonlinear propagation via TDNL (time-domain nonlinearity)

2. **Bioheat Transfer (BHT)**
   - Pennes bioheat equation solver
   - Perfusion and metabolic heat sources
   - Temperature-dependent tissue properties
   - Steady-state and transient modes

3. **Thermal Dose Calculation**
   - Cumulative Equivalent Minutes at 43°C (CEM43)
   - Arrhenius damage integral
   - Tissue necrosis threshold prediction
   - Safety margin assessment

4. **Synthetic Field Scanning**
   - SynthAxScan: Axial pressure profiles
   - SynthRadScan: Radial pressure profiles
   - Automated focal characterization
   - 3D field reconstruction from 2D axisymmetric

### Features Missing in Kwavers

1. **❌ Missing - Thermal Dose Calculation (CEM43)**
   - HITU: Standard CEM43 cumulative dose metric
   - Kwavers: Temperature field only, no dose integration
   - Gap: Cannot assess therapeutic efficacy or tissue damage
   - Impact: No clinical safety validation, cannot predict lesion size

2. **❌ Missing - Wide-Angle Parabolic Equation (WAKE/WAKZK)**
   - HITU: Efficient WAKE/WAKZK beam propagation
   - Kwavers: FDTD, PSTD, Angular Spectrum (no parabolic equation)
   - Gap: No intermediate-accuracy fast forward solver
   - Impact: FDTD too slow, Angular Spectrum too approximate for nonlinear

3. **✅ Partial - Axisymmetric Solvers**
   - Kwavers has: Basic 2D support (can enforce cylindrical symmetry manually)
   - Missing: Specialized radial coordinate handling, Hankel transforms
   - Gap: Not optimized for axisymmetric problems

4. **✅ Implemented - Bioheat Transfer**
   - Kwavers has: Pennes solver (`src/solver/forward/thermal_diffusion/`)
   - Status: Core implementation complete
   - Gap: Thermal dose calculation missing (item 1 above)

### Integration Recommendations

**Priority 1 (P2 - Medium)**: Thermal Dose Calculation (CEM43)
- **Scope**: Cumulative dose integration for tissue damage prediction
- **Approach**:
  1. Implement CEM43 formula: t_43 = Σ R^(43-T) dt where R=0.5 (T>43°C) or 0.25 (T<43°C)
  2. Add Arrhenius damage integral for alternate damage model
  3. Track thermal dose field during simulation
- **Effort**: 20-30 hours
- **Files**: `src/physics/thermal/dose.rs`, integrate with bioheat solver
- **Validation**: Compare with HITU_Simulator test cases
- **Benefit**: Clinical safety assessment, lesion size prediction

**Priority 2 (P2 - Medium)**: Wide-Angle Parabolic Equation (WAKE)
- **Scope**: Efficient forward-propagating nonlinear solver
- **Approach**:
  1. Implement WAKE/WAKZK in Hankel transform domain (radial)
  2. Add split-step Fourier for axial propagation
  3. Nonlinearity via operator splitting
- **Effort**: 20-30 hours
- **Files**: `src/solver/forward/parabolic_equation/wake.rs`
- **Validation**: Focused transducer (compare with FDTD for accuracy, with Angular Spectrum for speed)
- **Benefit**: 10-100× faster than FDTD for weakly nonlinear forward problems

---

## 9. Kranion (Transcranial HIFU Visualization)

**Repository**: https://github.com/jws2f/Kranion  
**Language**: Java with OpenGL 4.5  
**Focus**: Interactive 3D visualization for treatment planning

### Key Features

1. **Interactive 3D Rendering**
   - GPU-accelerated volume rendering (OpenGL 4.5)
   - Real-time slice planes (axial, coronal, sagittal)
   - Transducer geometry visualization
   - Focal zone overlay

2. **Patient Selection Metrics**
   - Skull Density Ratio (SDR) calculation from CT
   - Automatic patient suitability assessment
   - Threshold-based screening (SDR > 0.3 typical for transcranial)

3. **Phase Aberration Estimation**
   - CT-based acoustic phase delay computation
   - Skull-induced beam distortion prediction
   - Transducer efficiency assessment (predicted vs ideal intensity)

4. **DICOM/MRI Integration**
   - Direct medical image import
   - Coordinate system handling
   - Multi-modal overlay (MRI + CT)

### Features Missing in Kwavers

1. **❌ Missing - Interactive 3D Visualization**
   - Kranion: Real-time GPU volume rendering with transducer overlay
   - Kwavers: Static plots via plotly (web) or egui (GPU, limited)
   - Gap: No interactive treatment planning visualization
   - Impact: Cannot intuitively position transducers, assess coverage

2. **❌ Missing - Skull Density Ratio (SDR) Metric**
   - Kranion: Automatic SDR calculation for patient screening
   - Kwavers: No automated patient selection metrics
   - Gap: Manual CT analysis required
   - Impact: Cannot quickly assess transcranial feasibility

3. **❌ Missing - Phase Aberration Visualization**
   - Kranion: Visual overlay of phase delays and beam distortion
   - Kwavers: Computed but not visualized interactively
   - Gap: Difficult to interpret aberration effects
   - Impact: Unintuitive debugging of transcranial simulations

### Integration Recommendations

**Priority 1 (P3 - Low)**: Skull Density Ratio (SDR) Calculator
- **Scope**: Automated patient selection metric from CT
- **Approach**:
  1. Compute mean HU in skull ROI and reference brain ROI
  2. Calculate SDR = HU_skull / HU_brain
  3. Apply clinical threshold (SDR > 0.3 for transcranial HIFU)
- **Effort**: 10-20 hours
- **Files**: `src/clinical/therapy/patient_selection.rs`
- **Validation**: Process Kranion test datasets
- **Benefit**: Rapid patient screening, clinical workflow

**Priority 2 (P3 - Low)**: Interactive 3D Treatment Planning Visualization
- **Scope**: Real-time 3D rendering with transducer positioning
- **Approach**:
  1. Extend egui-based visualization with 3D scene
  2. Add transducer geometry rendering (phased array elements)
  3. Interactive focal point manipulation (click-and-drag)
- **Effort**: 60-80 hours (significant GPU programming)
- **Files**: `src/analysis/visualization/planning/`
- **Validation**: User study (ease of transducer positioning)
- **Benefit**: Intuitive clinical planning interface

---

## 10. dbua (Differentiable Beamforming with Ultrasound Arrays)

**Repository**: https://github.com/waltsims/dbua  
**Language**: Python  
**Focus**: Differentiable beamforming for learned reconstruction

### Key Algorithms

1. **Differentiable Delay-and-Sum (DAS)**
   - Backpropagation through beamforming pipeline
   - Learnable time-of-flight corrections
   - End-to-end optimization (raw data → image quality)

2. **Phase-Error Loss Functions**
   - Differentiable loss on phase coherence
   - Automatic aberration correction via gradient descent
   - No manual sound speed tuning required

3. **Time-of-Flight Gradient Computation**
   - Analytic gradients for geometric beamforming parameters
   - Efficient backpropagation (no finite differences)
   - Enables learned beamforming without simulation

### Features Missing in Kwavers

1. **❌ Missing - Differentiable Beamforming Pipeline**
   - dbua: Full gradient computation through DAS
   - Kwavers: Forward-only beamforming (no autodiff)
   - Gap: Cannot optimize beamforming parameters end-to-end
   - Impact: Manual tuning required, suboptimal image quality

2. **✅ Partial - Learned Beamforming**
   - Kwavers has: Neural beamforming module (`analysis/signal_processing/beamforming/neural/`)
   - Status: PINN-based approaches exist
   - Gap: No direct gradient backpropagation through beamforming operations

### Integration Recommendations

**Priority 1 (P2 - Medium)**: Differentiable Beamforming Operations
- **Scope**: Gradient computation through delay-and-sum
- **Approach**:
  1. Implement beamforming ops using Burn tensors (automatic gradient tracking)
  2. Add custom backward pass for interpolation (spline gradients)
  3. Create end-to-end training pipeline: RF data → beamformed image → loss
- **Effort**: 30-40 hours
- **Files**: `src/analysis/signal_processing/beamforming/differentiable.rs`
- **Validation**: Gradient check vs finite differences
- **Benefit**: Learned aberration correction, adaptive beamforming without explicit models

---

## 11. SimSonic (Advanced Ultrasound Simulation Platform)

**Website**: www.simsonic.fr (Note: SSL certificate expired, website inaccessible)  
**Language**: Unknown (likely C++/Python)  
**Focus**: Advanced tissue models, multi-modal integration

### Expected Features (Based on Literature)

SimSonic is known in the ultrasound community for:
1. High-fidelity tissue scattering models
2. Multi-modal imaging integration (B-mode, Doppler, elastography)
3. Clinical workflow integration
4. Advanced nonlinear propagation models

**Note**: Unable to perform detailed analysis due to website inaccessibility. Recommend revisiting when certificate is renewed.

---

## Cross-Cutting Integration Themes

### Theme 1: Differentiability and Inverse Problems

**Repositories**: jwave, dbua, Sound Speed Estimation  
**Common Pattern**: Gradient-based optimization for inverse problems

**Kwavers Status**: ✅ Partial (PINN infrastructure exists, forward solver differentiation missing)

**Recommended Actions**:
1. Extend Burn integration to forward solvers (FDTD, PSTD)
2. Implement differentiable beamforming operations
3. Add inverse problem templates (source localization, medium estimation, beamforming optimization)

**Effort**: 100-140 hours  
**Priority**: P0 (Critical for modern ML-based methods)

---

### Theme 2: Clinical Workflow Integration

**Repositories**: BabelBrain, Kranion, HITU Simulator  
**Common Pattern**: Medical imaging → treatment planning → safety assessment

**Kwavers Status**: ❌ Missing (standalone simulations only)

**Recommended Actions**:
1. Build medical imaging pipeline (DICOM/NIFTI → segmentation → acoustic properties)
2. Create integrated treatment planner (acoustic + thermal + dose)
3. Implement patient selection metrics (SDR, skull thickness, acoustic window)
4. Add interactive visualization for clinical users

**Effort**: 180-240 hours  
**Priority**: P0 (Critical for clinical translation)

---

### Theme 3: Multi-Scale Performance Optimization

**Repositories**: fullwave25, mSOUND, jwave  
**Common Pattern**: Problem-specific solver selection for optimal efficiency

**Kwavers Status**: ✅ Partial (multiple solvers exist, no automatic selection)

**Recommended Actions**:
1. Implement solver selection heuristics (problem size, frequency, nonlinearity → recommended solver)
2. Add multi-GPU domain decomposition for large 3D problems
3. Create mixed-domain solvers (TMDM, FSMDM) for efficiency
4. Optimize memory layouts for GPU transfers

**Effort**: 120-180 hours  
**Priority**: P1 (High for scalability)

---

### Theme 4: Advanced Acoustic Models

**Repositories**: k-Wave, fullwave25, mSOUND  
**Common Pattern**: Heterogeneous, nonlinear, dispersive propagation with complex boundaries

**Kwavers Status**: ✅ Mostly Implemented (core models exist, refinements needed)

**Recommended Actions**:
1. Add spatially-varying attenuation exponent γ(x,y,z)
2. Implement off-grid source/sensor integration (kWaveArray)
3. Add axisymmetric solver specializations
4. Enhance k-space temporal correction

**Effort**: 100-140 hours  
**Priority**: P1 (High for accuracy)

---

## Prioritized Integration Roadmap

### Sprint 1-2 (Weeks 1-4): Differentiable Simulations - P0 Critical
**Estimated Effort**: 140-180 hours

1. **Differentiable FDTD/PSTD Forward Solvers** (80-120 hours)
   - Burn integration for state evolution
   - Gradient checkpointing for memory efficiency
   - Validation against finite difference gradients

2. **Off-Grid Source/Sensor (kWaveArray)** (60-80 hours)
   - Gaussian quadrature surface integration
   - Fourier-space implementation
   - Realistic transducer element modeling

**Deliverables**: Full-waveform inversion capability, realistic near-field predictions

---

### Sprint 3-4 (Weeks 5-8): Clinical Workflow Integration - P0 Critical
**Estimated Effort**: 140-180 hours

1. **CT-Based Skull Modeling** (40-60 hours)
   - HU → acoustic property conversion
   - Heterogeneous skull with cortical/trabecular distinction

2. **Medical Imaging Pipeline** (60-80 hours)
   - DICOM/NIFTI loading with metadata parsing
   - Automatic tissue segmentation
   - Coordinate transformation infrastructure

3. **Integrated Treatment Planning** (40-60 hours)
   - Acoustic → thermal → dose chaining
   - Safety validation framework
   - Clinical parameter presets

**Deliverables**: End-to-end clinical HIFU planning workflow

---

### Sprint 5-6 (Weeks 9-12): Multi-Scale Performance - P1 High
**Estimated Effort**: 130-170 hours

1. **Spatially-Varying Attenuation Exponent** (20-30 hours)
   - γ(x,y,z) heterogeneous absorption
   - Multi-tissue dispersion

2. **Axisymmetric Solver Specialization** (40-60 hours)
   - 2D (r,z) solver for cylindrical symmetry
   - 10-100× speedup for focused transducers

3. **Multi-GPU Domain Decomposition** (50-70 hours)
   - Linear scaling across GPUs
   - Overlapped computation + communication

4. **Thermal Dose Calculation (CEM43)** (20-30 hours)
   - Cumulative dose integration
   - Lesion size prediction

**Deliverables**: Clinical-scale 3D simulations, therapeutic safety assessment

---

### Sprint 7-8 (Weeks 13-16): Advanced Methods - P1 High
**Estimated Effort**: 120-150 hours

1. **Transient Mixed-Domain Method (TMDM)** (50-70 hours)
   - Hybrid time-frequency for broadband sources
   - 5-10× speedup for attenuating media

2. **Frequency-Specific Mixed-Domain (FSMDM)** (30-50 hours)
   - Direct steady-state solution for CW
   - 10-100× faster HIFU simulations

3. **4th-Order Time Integration** (30-40 hours)
   - RK4 for FDTD
   - 4-16× fewer timesteps

4. **Enhanced k-Space Temporal Correction** (20-30 hours)
   - Exact dispersion relation
   - Spectral accuracy preservation

**Deliverables**: State-of-the-art efficiency for broadband and CW problems

---

### Sprint 9-10 (Weeks 17-20): User Experience - P1-P2
**Estimated Effort**: 100-130 hours

1. **Builder Pattern Simulation API** (20-30 hours)
   - Ergonomic construction with validation
   - Typestate pattern for completeness

2. **Clinical Transducer Library** (40-50 hours)
   - Pre-defined transducers with validated parameters
   - Manufacturer datasheets integration

3. **SLSC Beamforming + Adaptive Sound Speed** (30-50 hours)
   - Coherence-based beamforming
   - Automatic aberration correction

4. **Post-Processing Pipeline** (20-30 hours)
   - Chained operations: beamform → envelope → compress
   - Lazy evaluation

**Deliverables**: Researcher-friendly API, rapid prototyping tools

---

### Sprint 11-12 (Weeks 21-24): Advanced Features - P2-P3
**Estimated Effort**: 90-130 hours

1. **Differentiable Beamforming** (30-40 hours)
   - Gradient backpropagation through DAS
   - Learned aberration correction

2. **Wide-Angle Parabolic Equation (WAKE)** (20-30 hours)
   - Efficient forward-propagating solver
   - Intermediate accuracy option

3. **Skull Density Ratio (SDR) Calculator** (10-20 hours)
   - Patient screening metric

4. **Configurable Interface Reflections** (10-20 hours)
   - Toggle reflections on/off for validation

5. **Interactive 3D Visualization** (60-80 hours, optional)
   - Real-time treatment planning interface
   - GPU volume rendering

**Deliverables**: Research-grade advanced features, optional clinical visualization

---

## Summary Statistics

### Total Integration Effort
- **Minimum**: 750 hours (18.75 weeks FTE)
- **Maximum**: 1,050 hours (26.25 weeks FTE)
- **Average**: 900 hours (22.5 weeks FTE)

### Priority Breakdown
- **P0 Critical**: 280-360 hours (31-34% of total)
- **P1 High**: 320-430 hours (43-41% of total)
- **P2 Medium**: 150-220 hours (17-21% of total)
- **P3 Low**: 70-100 hours (9-10% of total)

### Category Breakdown
- **Algorithms**: 320-420 hours (43%)
- **Clinical Integration**: 180-240 hours (22%)
- **Performance**: 150-210 hours (19%)
- **User Experience**: 100-130 hours (13%)
- **Visualization**: 70-100 hours (9%)

---

## Implementation Guidelines

### Code Quality Standards
1. **Mathematical Specifications**: Every algorithm must include LaTeX formulas in documentation
2. **Literature References**: Cite original papers with DOI
3. **Validation Tests**: Compare against reference implementations or analytical solutions
4. **Convergence Studies**: Verify expected accuracy order (spatial, temporal)
5. **Performance Benchmarks**: Measure speedup vs existing methods

### Architecture Principles
1. **Single Source of Truth**: Reuse existing infrastructure (no duplication)
2. **Clean Architecture**: Domain → Application → Infrastructure → Interface layers
3. **Trait-Based Design**: Physics specifications as traits for testability
4. **Feature Flags**: Optional dependencies (GPU, PINN, clinical workflows)
5. **Backward Compatibility**: Maintain existing API surface during integration

### Testing Strategy
1. **Unit Tests**: Per-function correctness (100% coverage target)
2. **Integration Tests**: Multi-component workflows (90% coverage)
3. **Physics Validation**: Literature benchmarks (key papers reproduced)
4. **Performance Tests**: Regression detection (no slowdowns)
5. **Clinical Validation**: Comparison with commercial systems (when possible)

---

## Comparison with Kwavers Current State

### Strengths (Already Implemented)
1. ✅ **Clean Architecture**: Well-organized layer separation (domain/simulation/solver/clinical)
2. ✅ **PINN Infrastructure**: Burn-based physics-informed neural networks
3. ✅ **Multiple Solvers**: FDTD, PSTD, nonlinear (Westervelt, KZK, HAS)
4. ✅ **CPML Boundaries**: Convolutional perfectly matched layers
5. ✅ **Elastic Wave Support**: Shear wave propagation for elastography
6. ✅ **Bioheat Transfer**: Pennes equation solver for thermal effects
7. ✅ **Beamforming Algorithms**: DAS, MVDR, MUSIC, ESPRIT, subspace methods
8. ✅ **Multi-Physics Coupling**: Acoustic-thermal interactions
9. ✅ **GPU Support**: WGPU-based acceleration (partial)
10. ✅ **Medical Imaging I/O**: NIFTI and DICOM support (basic)

### Critical Gaps (P0 Priority)
1. ❌ **Differentiable Forward Solvers**: Cannot compute gradients through FDTD/PSTD
2. ❌ **Off-Grid Source/Sensor**: Grid-aligned only, no kWaveArray equivalent
3. ❌ **CT-Based Skull Modeling**: No HU→acoustic property conversion (TODO exists)
4. ❌ **Clinical Workflow Integration**: No end-to-end treatment planning pipeline
5. ❌ **Multi-GPU Scaling**: Single GPU only, cannot leverage clusters

### High-Priority Gaps (P1)
1. ❌ **Spatially-Varying Attenuation Exponent**: Single γ for entire domain
2. ❌ **Axisymmetric Solver**: No specialized cylindrical coordinate solver
3. ❌ **Mixed-Domain Methods**: No TMDM or FSMDM implementations
4. ❌ **4th-Order Time Integration**: 2nd-order leapfrog only
5. ❌ **Thermal Dose Calculation**: Temperature only, no CEM43 or damage integral

### Medium-Priority Gaps (P2)
1. ❌ **Fourier Series Initial Conditions**: Direct spatial arrays only
2. ❌ **Coherence-Based Beamforming**: No SLSC implementation
3. ❌ **Adaptive Sound Speed Estimation**: Fixed sound speed assumption
4. ❌ **Differentiable Beamforming**: Forward-only, no autodiff
5. ❌ **Clinical Transducer Library**: No pre-defined transducers
6. ❌ **Post-Processing Pipeline**: Manual chaining of operations

### Architectural Advantages of Kwavers
1. **Rust Safety**: Memory safety, thread safety, no undefined behavior
2. **Type System**: Compile-time guarantees vs runtime errors (MATLAB/Python)
3. **Zero-Cost Abstractions**: Trait-based design with no runtime overhead
4. **Cargo Ecosystem**: Modern build system, dependency management, testing
5. **Documentation**: Comprehensive rustdoc with mathematical specifications

---

## Risk Assessment

### Technical Risks

1. **Burn Framework Maturity** (Medium Risk)
   - Burn is evolving rapidly (v0.19 currently)
   - API changes may require refactoring
   - Mitigation: Version pinning, abstraction layers

2. **Multi-GPU Complexity** (High Risk)
   - Device synchronization challenging
   - Memory management across GPUs error-prone
   - Mitigation: Start with 2-GPU case, thorough testing, use NCCL

3. **Clinical Validation Requirements** (High Risk)
   - Medical device regulations (FDA, CE marking)
   - Extensive validation data needed
   - Mitigation: Collaborate with clinical partners, literature validation

4. **Performance Regression** (Medium Risk)
   - New abstractions may slow existing code
   - Mitigation: Continuous benchmarking, zero-cost abstraction principles

### Resource Risks

1. **Development Time** (Medium Risk)
   - 900 hours = 5.5 months FTE (optimistic)
   - Realistic with interruptions: 8-12 months
   - Mitigation: Prioritize P0 items, defer P2-P3

2. **Domain Expertise** (Medium Risk)
   - Requires acoustics, medical imaging, GPU programming expertise
   - Steep learning curve for each repository's methods
   - Mitigation: Literature review, reproduce reference examples first

3. **Testing Infrastructure** (Low Risk)
   - Validation requires reference data (k-Wave, experimental)
   - Mitigation: Use published benchmarks, open datasets

### Mitigation Strategy

1. **Phased Integration**: Implement P0 → P1 → P2 → P3 sequentially
2. **Early Validation**: Reproduce reference examples before full integration
3. **Community Engagement**: Collaborate with repository maintainers (feedback, bug reports)
4. **Continuous Integration**: Automated testing against reference solutions
5. **Documentation-First**: Write specs before implementation (prevents scope creep)

---

## Conclusion and Next Steps

This comprehensive review identifies **11 leading research repositories** with critical capabilities missing in kwavers. The prioritized integration roadmap spans **750-1,050 hours** (5-7 months FTE) across 12 sprints.

### Immediate Actions (Week 1)

1. **Validate Current Gaps** (8 hours)
   - Review kwavers backlog.md TODO items (align with this review)
   - Confirm P0 items not already in progress
   - Check for completed items since last audit

2. **Set Up Reference Environments** (12 hours)
   - Install k-Wave MATLAB toolbox
   - Install jwave Python library
   - Download BabelBrain and Kranion test datasets
   - Set up validation notebooks

3. **Create Sprint 213+ Plans** (8 hours)
   - Detailed task breakdown for differentiable FDTD/PSTD
   - Architecture design for off-grid source/sensor
   - Test plan for validation against k-Wave

4. **Stakeholder Review** (4 hours)
   - Present roadmap to project maintainers
   - Prioritize based on research needs
   - Identify collaboration opportunities (co-authors, datasets)

### Success Criteria

By the end of the 12-sprint integration:

1. **Capability Parity**: Match or exceed k-Wave feature set
2. **Performance**: 2-10× faster than MATLAB/Python equivalents (Rust advantage)
3. **Clinical Readiness**: End-to-end HIFU planning validated against BabelBrain
4. **Modern ML**: Differentiable simulations competitive with jwave
5. **Community**: 3+ external contributors, 5+ citations in literature

### Long-Term Vision

Transform kwavers into **the reference ultrasound simulation library** by:
- Combining best-in-class algorithms from 11+ repositories
- Leveraging Rust safety and performance advantages
- Maintaining open-source accessibility
- Supporting both research and clinical translation
- Enabling next-generation ML-guided ultrasound applications

---

**Report Completed**: 2026-01-23  
**Next Review**: After Sprint 213 (differentiable solvers complete)  
**Contact**: See kwavers repository for maintainer information
