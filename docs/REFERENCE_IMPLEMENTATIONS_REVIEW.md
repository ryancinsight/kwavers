# Reference Implementation Review: Best Practices for Ultrasound and Optics Simulation Libraries

**Date:** 2025-01-22  
**Purpose:** Comprehensive review of leading ultrasound/optics simulation libraries to identify architectural patterns, module organization strategies, and features that kwavers should incorporate  
**Scope:** 12 reference implementations across academic, research, and clinical domains

---

## Executive Summary

This document analyzes 12 leading ultrasound and optics simulation libraries to extract best practices for the kwavers architecture. Key findings reveal consistent patterns in:

1. **Four-layer separation model** (Grid → Medium → Source/Sensor → Solver)
2. **Physics-solver decoupling** with pluggable numerical backends
3. **Differentiable simulation frameworks** for inverse problems
4. **Multi-domain heterogeneous media** support
5. **Clinical workflow integration** with medical imaging standards

### Critical Recommendations for Kwavers

| Priority | Recommendation | Rationale | Implementation |
|----------|---------------|-----------|----------------|
| **P0** | Adopt configuration object pattern | Reduces function parameter proliferation | Create `SimulationConfig` struct |
| **P0** | Implement physics trait system | Enables solver interchangeability | Already started in ADR PINN restructuring |
| **P1** | Add differentiation support | Research applications (FWI, parameter estimation) | Integrate with burn's autodiff |
| **P1** | Standardize medical imaging I/O | Clinical interoperability | Extend DICOM/NIfTI support |
| **P2** | Create material property database | Tissue modeling | Add IT'IS Foundation integration |
| **P2** | Implement hybrid solver framework | Performance + accuracy optimization | Domain decomposition strategies |

---

## 1. J-Wave (JAX-Based Wave Simulation)

**Repository:** [https://github.com/ucl-bug/jwave](https://github.com/ucl-bug/jwave)  
**Language:** Python (JAX)  
**Key Innovation:** Differentiable physics for machine learning integration

### Architecture Analysis

#### Module Organization
```
jwave/
├── acoustics/          # Physics solvers and simulations
├── geometry/           # Domain, medium, and spatial definitions
└── utils/              # Helper functions and utilities
```

**Separation of Concerns:**
- **Physics Layer:** `acoustics` module contains time-varying simulators with `simulate_wave_propagation()` as primary entry point
- **Geometric Primitives:** `geometry` module abstracts domain discretization through `Domain`, `Medium`, and `TimeAxis` classes
- **Data Representations:** `FourierSeries` encapsulates field data in frequency domain

#### Key Design Patterns

**1. Functional Composition**
```python
# JAX decorator pattern for compilation and autodiff
@jit
def simulate_wave_propagation(medium, sources, sensors, time_axis):
    # Composable simulation blocks
    pass
```

**2. Configuration Objects**
- `Medium` and `TimeAxis` encapsulate simulation parameters
- Reduces function argument proliferation
- Enables flexible setup without rigid class hierarchies

**3. Modular Blocks**
- Discrete, reusable components easily incorporated into ML pipelines
- Clear responsibility boundaries between physics, geometry, utilities

#### Notable Features

| Feature | Description | Kwavers Application |
|---------|-------------|---------------------|
| **Differentiability** | Core design principle enabling gradient-based optimization | Integrate with burn autodiff for PINNs |
| **GPU/TPU Portability** | JAX backend provides transparent hardware acceleration | Extend wgpu support to cover all physics |
| **Initial Value Problems** | Supports photoacoustic acquisition through field initialization | Add `InitialCondition` trait |

### Recommendations for Kwavers

1. **Adopt Configuration Objects** ✅ (Already have `SimulationConfig`)
   - Consolidate scattered parameters into typed structs
   - Example: `TimeAxis`, `MediumConfig`, `SourceConfig`

2. **Functional Composition Pattern**
   - Current plugin system is similar but could be more compositional
   - Consider builder pattern for complex simulations

3. **Frequency Domain Abstractions**
   - J-Wave's `FourierSeries` suggests separating field representation from solvers
   - Kwavers has this in `math/fft` but could expose higher-level API

---

## 2. K-Wave (MATLAB k-Wave)

**Repository:** [https://github.com/ucl-bug/k-wave](https://github.com/ucl-bug/k-wave)  
**Language:** MATLAB with C++/CUDA binaries  
**Key Innovation:** k-space pseudospectral method for efficiency

### Architecture Analysis

#### Key Architectural Patterns

**1. Separation of Numerical Methods**
- **Spatial discretization:** k-space pseudospectral (Fourier collocation)
- **Temporal discretization:** Separate from spatial gradients
- **Physics layers:** Linear and nonlinear wave propagation as distinct pathways

**2. Acoustic Absorption Modeling**
- Power law absorption uses "fractional Laplacian" as specialized mathematical abstraction
- Suggests domain-specific operators rather than monolithic solvers

**3. Boundary Condition Handling**
- Split-field perfectly matched layer (PML) as dedicated mechanism
- Modular boundary treatment separate from core equations
- **Kwavers Status:** ✅ Already have `CPMLBoundary` in `domain/boundary/cpml/`

**4. Dimensional Flexibility**
- Support for 1D, 2D, 3D through parameterized spatial handling
- Not separate code paths (DRY principle)
- **Kwavers Status:** ✅ Grid supports 3D, can be constrained to 2D/1D

#### Notable Features

| Feature | Description | Kwavers Equivalent | Action |
|---------|-------------|-------------------|--------|
| **kWaveArray class** | Source/sensor distribution | `domain/source/` and `domain/sensor/` | ✅ Implemented |
| **Accelerated binaries** | C++/CUDA in `k-Wave/binaries` | `gpu/` module with wgpu | Extend GPU coverage |
| **Axisymmetric models** | Specialized for cylindrical symmetry | None | Add to backlog (P2) |
| **Elastic wave variants** | Inheritance or strategy pattern | `physics/acoustics/mechanics/elastic_wave/` | ✅ Implemented |

#### Design Philosophy

> "fewer spatial and temporal grid points are needed for accurate simulations"

This reflects **performance-conscious decomposition** of the computational problem.

**Kwavers Alignment:**
- k-space methods in `solver/forward/pstd/implementation/k_space/`
- Focus on efficiency in `analysis/performance/optimization/`

### Recommendations for Kwavers

1. **Axisymmetric Solver Support** (P2 Priority)
   - Many clinical transducers have cylindrical symmetry
   - Significant computational savings (2D instead of 3D)
   - Implementation path: `solver/forward/pstd/axisymmetric/`

2. **Material Database Integration** (P1 Priority)
   - k-Wave provides extensive tissue property data
   - Kwavers should integrate standard databases
   - Location: `domain/medium/properties/tissue_database.rs`

3. **Binary I/O Format**
   - k-Wave uses HDF5 for large datasets
   - Kwavers currently uses JSON (not scalable)
   - Add HDF5 support to `infra/io/`

---

## 3. K-Wave-Python

**Documentation:** [https://k-wave-python.readthedocs.io/en/latest/](https://k-wave-python.readthedocs.io/en/latest/)  
**Language:** Python  
**Key Innovation:** Pythonic API design while preserving physics accuracy

### Architecture Analysis

#### Core Architectural Components

The toolkit organizes around **four fundamental simulation elements:**

```python
1. kWaveGrid     # Computational domain foundation
2. kWaveMedium   # Material property definition
3. kSource       # Acoustic input configuration
4. kSensor       # Measurement/data acquisition
```

**Design Pattern:** Clean separation of concerns, independently configurable aspects

**Kwavers Alignment:**
```rust
✅ Grid          → domain::grid::Grid
✅ Medium        → domain::medium::Medium (trait)
✅ Source        → domain::source::Source
✅ Sensor        → domain::sensor::GridSensorSet
```

#### API Design Patterns

**Primary Classes:**
- Object-oriented approach encapsulating domain concepts
- Composable entities (no monolithic simulator)
- **Pythonic practices** while maintaining MATLAB conceptual alignment

**Kwavers Status:** ✅ Similar trait-based design with Rust's type system

#### Physics & Solver Implementation

**Multiple Solver Variants:**
- `kspaceFirstOrder2D`
- `kspaceFirstOrder3D`
- `kspaceFirstOrderAS` (axisymmetric)

**Kwavers Equivalent:**
- `solver/forward/pstd/implementation/`
- Dimensionality handled through generic `Grid` rather than separate solvers

#### Utility & Support Modules

**Extensive utility submodules:**
- Reconstruction
- Filters
- Conversion
- Interpolation
- Signal processing
- Ultrasound B-mode reconstruction

**Kwavers Status:**
- ✅ `analysis/signal_processing/`
- ✅ `clinical/imaging/workflows/`
- ⚠️ Missing: Standardized reconstruction pipelines

### Recommendations for Kwavers

1. **Standardized Reconstruction API** (P1 Priority)
   ```rust
   // Proposed API
   pub mod reconstruction {
       pub fn time_reversal(sensor_data: &Array3<f64>, grid: &Grid) -> Array3<f64>;
       pub fn delay_and_sum(rf_data: &Array3<f64>, params: &BeamformParams) -> Array2<f64>;
       pub fn backprojection(sinogram: &Array2<f64>, angles: &[f64]) -> Array2<f64>;
   }
   ```
   - Location: `clinical/imaging/reconstruction/`
   - Status: Partially exists, needs consolidation

2. **HDF5 Integration for Interoperability** (P1 Priority)
   - k-Wave-Python uses HDF5 for data exchange
   - Enables cross-validation with other tools
   - Implementation: Add `hdf5` crate to `infra/io/`

3. **Documentation Structure**
   - k-Wave-Python has excellent "Step-by-Step" tutorials
   - Kwavers should add `docs/tutorials/` with runnable examples
   - Use `mdbook` or similar for user-friendly docs

---

## 4. OptimUS (Boundary Element Method)

**Repository:** [https://github.com/optimuslib/optimus](https://github.com/optimuslib/optimus)  
**Language:** Python  
**Key Innovation:** Boundary element method (BEM) for unbounded domains

### Architecture Analysis

#### Physics-Solver Separation

**Mathematical Formulation:**
- Solves Helmholtz equation in multiple domains
- Homogeneous material parameters per domain
- Boundary element method (BEM) as computational backend

**Key Insight:** Separates physics (Helmholtz PDE) from numerics (BEM implementation)

**Kwavers Parallel:**
- `domain/physics/` defines equations (trait specifications)
- `solver/forward/helmholtz/` implements numerical methods
- ✅ Already aligned with OptimUS philosophy

#### Multi-Domain Handling

**Architectural Features:**
- Domain decomposition across different materials
- Boundary interface management
- Heterogeneous material property handling
- Unbounded domain support (critical for ultrasound in water tanks)

**Kwavers Status:**
- ✅ `domain/medium/heterogeneous/` supports multi-region media
- ⚠️ Missing: Explicit interface tracking between regions
- ⚠️ Missing: Unbounded domain support (infinite elements or truncation strategies)

#### Backend Abstraction

**Design Pattern:**
- Uses `bempp-legacy` as computational backend
- Core algorithms independent from low-level numerics
- **Pluggable backend architecture**

**Kwavers Equivalent:**
- Current solvers (FDTD, PSTD) are tightly coupled
- **Opportunity:** Extract `SolverBackend` trait
  ```rust
  pub trait SolverBackend {
      fn step(&mut self, dt: f64) -> Result<()>;
      fn get_field(&self, field_type: FieldType) -> &Array3<f64>;
      fn apply_source(&mut self, source: &dyn Source);
  }
  ```

#### Notable Design Features

| Feature | Description | Kwavers Application |
|---------|-------------|---------------------|
| **Application-focused** | Targets focused ultrasound in biomedical engineering | ✅ `clinical/therapy/` module |
| **Tissue database** | IT'IS Foundation material properties | Add to `domain/medium/properties/` |
| **Docker containerization** | Reproducible environments | Add Dockerfile for CI/CD |

### Recommendations for Kwavers

1. **Boundary Element Method Solver** (P2 Priority)
   - Useful for unbounded domains (ultrasound in water tanks)
   - Location: `solver/forward/bem/`
   - Status: Directory exists but unimplemented

2. **Interface Tracking for Multi-Region Media** (P1 Priority)
   ```rust
   pub struct MaterialInterface {
       region_a: MaterialId,
       region_b: MaterialId,
       boundary: Vec<Point3<f64>>,
       continuity: InterfaceContinuity, // Acoustic, Sliding, Rigid
   }
   ```
   - Location: `domain/medium/heterogeneous/interfaces.rs`
   - Supports mode conversion, reflection/transmission

3. **IT'IS Tissue Property Database** (P1 Priority)
   - Standard reference for biomedical simulations
   - Include frequency-dependent properties
   - JSON or embedded database at `domain/medium/properties/tissue_db.json`

---

## 5. Fullwave 2.5 (Heterogeneous Power Law Attenuation)

**Repository:** [https://github.com/pinton-lab/fullwave25](https://github.com/pinton-lab/fullwave25)  
**Language:** Python/CUDA  
**Key Innovation:** Heterogeneous power law attenuation with GPU acceleration

### Architecture Analysis

#### Layered Design

**Three-layer architecture:**
```
1. Python Interface Layer    → User-facing API (accessibility)
2. CUDA/C Backend            → High-performance computation
3. Domain Abstraction        → Grid, medium, sources managed separately
```

**Design Philosophy:**
> "offers a user experience similar to k-Wave and k-wave-python, while providing advanced attenuation modeling"

**Kwavers Parallel:**
- Layer 1: Public API in `src/lib.rs` re-exports
- Layer 2: GPU kernels in `gpu/shaders/`
- Layer 3: Domain types in `domain/`

✅ **Already aligned with Fullwave's architecture**

#### Simulation Workflow Separation

**Distinct phases enable modular development:**
1. Grid definition
2. Medium property specification
3. Acoustic source configuration
4. Sensor placement
5. Execution pipeline

**Kwavers Status:** ✅ Similar workflow in examples and `simulation/` module

#### Multi-GPU Support

**Key Architectural Decision:**
> "Multi-GPU domain decomposition is processed in the depth dimension"

**Grid coordinates:** (depth, lateral, elevational) for optimization

**Kwavers Opportunity:**
- Current GPU support through `wgpu` (single device)
- Multi-GPU decomposition should use rayon parallelism + device replication
- Implementation path: `gpu/distributed/`

#### Medium Builder Utility

**Separate utility module:**
- Creating computational medium from geometric operations
- Complex heterogeneous media without monolithic code

**Kwavers Status:**
- ✅ `domain/medium/heterogeneous/factory/` provides similar functionality
- Could add more geometric primitives (spheres, ellipsoids, layered media)

### Development Standards

**Contribution Structure:**
- Branch naming: `TYPE/BRANCH_NAME` (feature, bugfix, hotfix, docs, refactor, release, test, experiment)
- Pre-commit configuration
- `uv` package manager

**Kwavers Alignment:**
- ✅ Clear sprint-based development
- ✅ Comprehensive test suite
- Consider: Add pre-commit hooks for Rust (rustfmt, clippy)

### Recommendations for Kwavers

1. **Power Law Attenuation Implementation** (P0 Priority)
   - Fullwave's key innovation: spatially heterogeneous attenuation
   - Kwavers has basic absorption in `domain/medium/absorption/`
   - **Enhancement needed:** Fractional derivative operators
   - Location: `physics/acoustics/mechanics/acoustic_wave/absorption.rs`

2. **Multi-GPU Domain Decomposition** (P2 Priority)
   ```rust
   pub struct MultiGPUDomain {
       devices: Vec<wgpu::Device>,
       subdomains: Vec<Subdomain>,
       halo_regions: Vec<HaloExchange>,
   }
   ```
   - Location: `gpu/distributed/`
   - Requires halo exchange for boundary data

3. **Medium Builder DSL** (P2 Priority)
   ```rust
   // Fluent API for complex media
   let medium = MediumBuilder::new(&grid)
       .add_layer(0.0..5.0, tissue::WATER)
       .add_sphere(Point3::new(10.0, 10.0, 10.0), 2.0, tissue::LIVER)
       .add_ellipsoid(center, radii, tissue::KIDNEY)
       .build()?;
   ```
   - Location: `domain/medium/heterogeneous/builder.rs`

---

## 6. DBUA (Differentiable Beamforming for Ultrasound Autofocusing)

**Repository:** [https://github.com/waltsims/dbua](https://github.com/waltsims/dbua)  
**Language:** Python (JAX)  
**Key Innovation:** End-to-end differentiable ultrasound beamforming

### Architecture Analysis

#### Modular, Functional Decomposition

**Four core modules:**
```python
dbua.py     # Main orchestrator (experiment config, execution)
das.py      # Delay-and-sum beamforming
paths.py    # Acoustic propagation times (heterogeneous SoS)
losses.py   # Differentiable loss functions (phase error)
```

**Domain-Driven Design:** Each module encapsulates specific ultrasound signal processing concern

**Kwavers Status:**
- ✅ `analysis/signal_processing/beamforming/` exists
- ⚠️ Not fully differentiable (no autodiff integration)
- ⚠️ Path calculation tightly coupled to solver

#### Gradient-Based Optimization Pattern

**Workflow:**
```
1. Load IQ data (MATLAB files)
2. Parameterize time-delay profiles (paths module)
3. Apply delay-and-sum beamforming (das module)
4. Evaluate differentiable loss (losses module)
5. Optimize delays using autodiff
```

**Key Insight:** Treating ultrasound beamforming as learnable process

**Kwavers Opportunity:**
- Current beamforming in `analysis/signal_processing/beamforming/` is deterministic
- Add `adaptive_differentiable.rs` module using burn autodiff
- Enable data-driven beamforming optimization

#### Configuration-Driven Execution

**Global configuration parameters:**
- Facilitates reproducibility
- Parameter sweeping without code modification

**Kwavers Status:**
- ✅ `simulation/configuration/` provides similar capability
- Enhancement: Add serialization for experiment tracking

#### Notable Features

| Feature | Description | Kwavers Application |
|---------|-------------|---------------------|
| **JAX autodiff** | Transforms signal processing into differentiable blocks | Use burn autodiff for beamforming |
| **Multi-scenario testing** | 14 phantom datasets (checker, layers, inclusions) | Add to `tests/phantoms/` |
| **Video output** | Visualization for result analysis | Extend `analysis/visualization/` |

### Recommendations for Kwavers

1. **Differentiable Beamforming Module** (P1 Priority)
   ```rust
   // Using burn for autodiff
   pub mod differentiable_beamforming {
       use burn::prelude::*;
       
       pub struct DifferentiableBeamformer<B: Backend> {
           delays: Tensor<B, 2>,  // Learnable delays
           apodization: Tensor<B, 1>,  // Learnable weights
       }
       
       impl<B: Backend> DifferentiableBeamformer<B> {
           pub fn forward(&self, iq_data: Tensor<B, 3>) -> Tensor<B, 2> {
               // Differentiable DAS
           }
       }
   }
   ```
   - Location: `analysis/signal_processing/beamforming/neural/differentiable.rs`
   - Status: Directory exists but needs implementation

2. **Acoustic Path Calculator** (P1 Priority)
   - Current time-of-flight assumes homogeneous medium
   - Need ray tracing through heterogeneous SoS
   - Location: `physics/acoustics/analytical/propagation/heterogeneity/`
   - Algorithm: Eikonal equation solver or Fermat's principle

3. **Phase Error Loss Functions** (P2 Priority)
   ```rust
   pub enum BeamformingLoss {
       PhaseError,
       CoherenceFactor,
       GeneralizedCoherence,
       SpeckleSNR,
   }
   ```
   - Location: `analysis/signal_processing/beamforming/metrics.rs`
   - Enables optimization-based beamforming

---

## 7. Sound Speed Estimation (Inverse Problem Focus)

**Repository:** [https://github.com/JiaxinZHANG97/Sound-Speed-Estimation](https://github.com/JiaxinZHANG97/Sound-Speed-Estimation)  
**Language:** MATLAB + Python  
**Key Innovation:** Spatial coherence-based parameter estimation

### Architecture Analysis

#### Inverse Problem Formulation

**Problem:** Estimate acoustic velocity parameters from ultrasound measurements

**Method:** Grid-search optimization by maximizing short-lag spatial coherence (SLSC)

**Key Modules:**
```matlab
% Signal Processing Pipeline
iq2rf_jz.m                          % IQ-to-RF conversion
beamformer_SLSC_PW_US_linear.m      % SLSC beamforming
SoS_estimation_coherence.m          % Maximum/minimum SLSC
```

**Inverse Problem Solving:**
- mLOC (maximum lag-one coherence) metric
- Region-of-interest based optimization
- Histogram filtering for robust estimation

#### Integration with Simulation Framework

**Design Pattern:** Multilingual separation of concerns
- **Python:** I/O operations (CUBDL dataset loading)
- **MATLAB:** Computationally intensive coherence calculations
- **Separation rationale:** Leverage strengths of each language

**Kwavers Opportunity:**
- Currently no inverse problem solvers for parameter estimation
- Add optimization-based methods to `solver/inverse/reconstruction/`

### Recommendations for Kwavers

1. **Coherence-Based Metrics** (P2 Priority)
   ```rust
   pub mod coherence {
       pub fn short_lag_spatial_coherence(
           iq_data: &Array3<Complex<f64>>,
           lag: usize
       ) -> Array2<f64>;
       
       pub fn lag_one_coherence(
           iq_data: &Array3<Complex<f64>>
       ) -> Array2<f64>;
   }
   ```
   - Location: `analysis/signal_processing/beamforming/coherence.rs`
   - Used for sound speed estimation, aberration correction

2. **Parameter Estimation Framework** (P2 Priority)
   ```rust
   pub trait ParameterEstimator {
       type Parameters;
       type Measurements;
       
       fn estimate(&self, measurements: &Self::Measurements) -> Result<Self::Parameters>;
       fn cost_function(&self, params: &Self::Parameters, data: &Self::Measurements) -> f64;
   }
   
   pub struct SoundSpeedEstimator {
       grid: Grid,
       roi: Region,
       metric: CoherenceMetric,
   }
   ```
   - Location: `solver/inverse/estimation/`
   - Generic framework for various inverse problems

3. **ROI-Based Processing** (P1 Priority)
   - Many inverse problems require region-of-interest selection
   - Add `Region` type to `domain/geometry/`
   - Enable selective processing for efficiency

---

## 8. mSOUND (Mixed-Domain Method)

**Repository:** [https://github.com/m-SOUND/mSOUND](https://github.com/m-SOUND/mSOUND)  
**Language:** MATLAB  
**Key Innovation:** Transient and frequency-specific mixed-domain methods

### Architecture Analysis

#### Dual Solver Architecture

**Two computational methods:**

1. **TMDM (Transient Mixed-Domain Method)**
   - Generates time-domain results
   - Suited for pulsed-wave simulations
   - Arbitrary nonlinearity modeling

2. **FSMDM (Frequency-Specific Mixed-Domain Method)**
   - Produces steady-state results at specific frequencies
   - Center frequency and second harmonics
   - Linear/weakly nonlinear propagation

**Key Insight:** Different physics regimes require different numerical approaches

**Kwavers Status:**
- ✅ Time-domain: FDTD in `solver/forward/fdtd/`
- ✅ Frequency-domain: PSTD in `solver/forward/pstd/`
- ✅ Nonlinear: Kuznetsov in `solver/forward/nonlinear/kuznetsov/`
- ⚠️ Missing: Explicit frequency-specific steady-state solver

#### Domain Separation

**Physics:** Generalized Westervelt equation

**Heterogeneous media support:**
- Sound speed (spatially varying)
- Density (spatially varying)
- Attenuation coefficient (spatially varying)
- Power law exponent (spatially varying)
- Nonlinear coefficient (B/A, spatially varying)

**Kwavers Status:**
- ✅ `domain/medium/heterogeneous/` supports all properties
- Could enhance with more material models

#### Mixed-Domain Methodology

**Suggestion:** Integration of multiple computational domains (frequency + spatial)

**Kwavers Opportunity:**
- Current solvers are pure time-domain or pure frequency-domain
- Hybrid time-frequency methods could improve efficiency
- Example: Nonlinear term in time domain, linear propagation in frequency domain

### Recommendations for Kwavers

1. **Frequency-Specific Steady-State Solver** (P2 Priority)
   ```rust
   pub struct FrequencySpecificSolver {
       fundamental: f64,
       harmonics: Vec<usize>,  // [1, 2, 3] for fundamental + 2nd + 3rd
       grid: Grid,
       medium: Box<dyn Medium>,
   }
   
   impl FrequencySpecificSolver {
       pub fn solve_steady_state(&self) -> HashMap<usize, Array3<Complex<f64>>> {
           // Helmholtz equation per frequency
       }
   }
   ```
   - Location: `solver/forward/helmholtz/frequency_specific.rs`
   - Useful for CW ultrasound, harmonic imaging

2. **Generalized Westervelt Equation** (P1 Priority)
   - More general than current Kuznetsov equation
   - Includes thermoviscous effects
   - Location: `physics/acoustics/mechanics/acoustic_wave/nonlinear/westervelt.rs`

3. **Hybrid Time-Frequency Solver** (P2 Priority)
   - Nonlinear terms computed in time domain
   - Linear propagation in frequency domain
   - Trade-off: Complexity vs. efficiency
   - Location: `solver/forward/hybrid/time_frequency.rs`

---

## 9. HITU Simulator (High-Intensity Therapeutic Ultrasound)

**Repository:** [https://github.com/jsoneson/HITU_Simulator](https://github.com/jsoneson/HITU_Simulator)  
**Language:** MATLAB  
**Key Innovation:** Coupled acoustic-thermal simulation for therapy

### Architecture Analysis

#### Module Organization

**Core simulation modules:**
```matlab
% Acoustic propagation
WAKZK.m              % General ultrasound field
WAKZK_Gaussian.m     % Gaussian beam
WAKZK_planar.m       % Planar wave

% Nonlinear acoustics
TDNL.m               % Time-domain nonlinear

% Thermal modeling
BHT.m                % Bioheat transfer
```

**Supporting infrastructure:**
```matlab
BuildPade11operators.m    % Numerical operators (Padé approximations)
BuildPade12operators.m
BuildBHTperipherals.m     % Peripheral thermal configuration
SourceFilterH.m           % Source field filtering
LinearHeating.m           % Linear thermal response
```

**Utility functions:**
```matlab
SynthAxScan.m        % Synthetic axial scanning
SynthRadScan.m       % Synthetic radial scanning
matrixize.m          % Data structure conversions
vektorize.m
```

#### Key Architectural Patterns

**Modular Design:**
- Clear separation: acoustic propagation → heating → data transformation
- Flexible simulation workflows
- Each module has single responsibility

**Axisymmetric Focus:**
- Specializes in axisymmetric beams (reduces 3D → 2D)
- Computational efficiency for typical HITU transducers
- Cylindrical coordinate system

**Continuous Wave Support:**
- Emphasizes CW ultrasound (not pulsed)
- Simplifies thermal dose calculations
- Appropriate for therapeutic applications

#### Acoustic-Thermal Coupling

**Two-way coupling:**
1. **Acoustic → Thermal:** Pressure field generates heat (absorption)
2. **Thermal → Acoustic:** Temperature affects sound speed, attenuation

**Kwavers Status:**
- ✅ Basic thermal diffusion in `solver/forward/thermal_diffusion/`
- ⚠️ Weak coupling: Acoustic field doesn't feed back to thermal
- ⚠️ Missing: Nonlinear bioheat transfer (Pennes equation)

### Recommendations for Kwavers

1. **Bioheat Transfer Module** (P1 Priority)
   ```rust
   pub struct BioheatSolver {
       thermal_conductivity: Array3<f64>,
       perfusion_rate: Array3<f64>,      // Blood perfusion
       metabolic_heat: Array3<f64>,       // Baseline metabolism
       blood_temperature: f64,            // Arterial blood temp
   }
   
   impl BioheatSolver {
       pub fn step(&mut self, dt: f64, acoustic_intensity: &Array3<f64>) {
           // Pennes bioheat equation
           // ρc ∂T/∂t = ∇·(k∇T) - ρ_b c_b ω_b (T - T_a) + Q_m + Q_a
       }
   }
   ```
   - Location: `physics/thermal/bioheat.rs`
   - Includes perfusion, metabolism, acoustic heating

2. **Two-Way Acoustic-Thermal Coupling** (P1 Priority)
   ```rust
   pub struct CoupledAcousticThermal {
       acoustic_solver: Box<dyn AcousticSolver>,
       thermal_solver: BioheatSolver,
       coupling_strength: f64,
   }
   
   impl CoupledAcousticThermal {
       pub fn step(&mut self, dt: f64) {
           // 1. Acoustic step → get intensity field
           let intensity = self.acoustic_solver.step(dt);
           
           // 2. Thermal step with acoustic heating
           self.thermal_solver.step(dt, &intensity);
           
           // 3. Update acoustic properties from temperature
           let temp = self.thermal_solver.temperature();
           self.update_acoustic_properties(temp);
       }
   }
   ```
   - Location: `solver/multiphysics/acoustic_thermal.rs`
   - Directory exists but unimplemented

3. **Axisymmetric Solver Specialization** (P2 Priority)
   - Reduces computational cost for HIFU applications
   - Cylindrical coordinates (r, z, θ)
   - Location: `solver/forward/pstd/axisymmetric/`

4. **Therapy-Specific Utilities** (P2 Priority)
   ```rust
   pub mod therapy_metrics {
       pub fn thermal_dose(temperature: &Array3<f64>, dt: f64) -> Array3<f64>;
       pub fn cumulative_equivalent_minutes(temp: &Array3<f64>, time: f64) -> f64;
       pub fn mechanical_index(pressure: f64, frequency: f64) -> f64;
       pub fn thermal_index(power: f64, area: f64) -> f64;
   }
   ```
   - Location: `clinical/therapy/metrics/`
   - ✅ Partially implemented, needs expansion

---

## 10. BabelBrain (Transcranial Focused Ultrasound)

**Repository:** [https://github.com/ProteusMRIgHIFU/BabelBrain](https://github.com/ProteusMRIgHIFU/BabelBrain)  
**Language:** Python  
**Key Innovation:** Clinical workflow integration with medical imaging

### Architecture Analysis

#### Module Organization

```python
BabelBrain/                # Core application (GUI, orchestration)
TranscranialModeling/      # Acoustic simulation logic
ThermalModeling/           # Thermal effects calculations
PlanningModels/            # Treatment planning utilities
OfflineBatchExamples/      # Jupyter notebooks for parametric studies
Profiles/                  # Configuration files (treatment protocols)
```

**Clear separation:** Physics domains, workflows, configuration

#### Architectural Patterns

**1. Separation of Physics Domains from Execution**

**Physics Solvers:** Delegates to external "BabelViscoFDTD"
- Finite-difference time-difference solver
- Isotropic viscoelastic equation
- Multi-GPU support (Metal, OpenCL, CUDA)

**Key Insight:** Core application remains GPU-backend agnostic

**Image Processing Pipeline:** Step 1 handles:
- Domain generation
- Tissue segmentation
- Mesh preparation
- **Independent from acoustic simulation**

**Thermal Modeling:** Step 3 computes heating
- Uses exposure parameters from Step 2
- Separated from acoustic field calculations

**Kwavers Parallel:**
- ✅ Plugin-based solver system allows backend swapping
- ✅ Multi-physics in `solver/multiphysics/`
- Could improve: More explicit workflow orchestration

**2. Multi-Backend GPU Strategy**

**Design Decision:** Leverage external solver (BabelViscoFDTD) rather than embedding GPU code

**Benefits:**
- Platform-agnostic deployment (Metal, OpenCL, CUDA)
- Application logic separate from GPU implementation
- Backend upgrades don't require app changes

**Kwavers Status:**
- ✅ Uses wgpu (cross-platform abstraction)
- Could add: External solver integration API

**3. Workflow Orchestration**

**Three-step pipeline:**
1. **Step 1:** Mesh/domain construction (from MRI/CT)
2. **Step 2:** Acoustic field computation (pressure, intensity)
3. **Step 3:** Thermal analysis and output (temperature, dose)

**Configuration-driven:**
- YAML thermal profiles
- Flexible parametric exploration
- No code modification needed

**Kwavers Opportunity:**
- Current simulations require programmatic setup
- Add workflow DSL or configuration-driven execution
- Location: `clinical/workflows/`

#### Notable Design Features

| Feature | Description | Kwavers Application |
|---------|-------------|---------------------|
| **Smart caching** | Hash-signature reuse (20-80% speedup) | Add to `analysis/performance/` |
| **Transducer abstraction** | 15+ device geometries | Extend `domain/source/transducers/` |
| **Neuronavigation** | Brainsight/3DSlicer integration | Add to `infra/io/` (P3) |
| **Medical imaging** | NIfTI, DICOM support | ✅ Already have in `nifti`, `dicom` deps |

### Recommendations for Kwavers

1. **Clinical Workflow Orchestration** (P1 Priority)
   ```rust
   pub struct ClinicalWorkflow {
       steps: Vec<Box<dyn WorkflowStep>>,
       cache: WorkflowCache,
   }
   
   pub trait WorkflowStep {
       fn execute(&mut self, inputs: &WorkflowData) -> Result<WorkflowData>;
       fn cache_key(&self) -> String;
       fn is_cacheable(&self) -> bool;
   }
   
   // Example steps
   pub struct ImageSegmentation;
   pub struct AcousticSimulation;
   pub struct ThermalAnalysis;
   pub struct DoseCalculation;
   ```
   - Location: `clinical/workflows/orchestrator.rs`
   - Enables reproducible, cacheable pipelines

2. **Smart Caching with Hash Signatures** (P1 Priority)
   ```rust
   pub struct SimulationCache {
       cache_dir: PathBuf,
       hasher: blake3::Hasher,
   }
   
   impl SimulationCache {
       pub fn get_or_compute<F>(&self, key: &str, compute: F) -> Result<CacheEntry>
       where F: FnOnce() -> Result<CacheEntry>;
   }
   ```
   - Location: `analysis/performance/cache/simulation_cache.rs`
   - Hash simulation parameters, reuse results

3. **Transducer Device Library** (P2 Priority)
   - BabelBrain supports 15+ commercial transducers
   - Add device specifications to `domain/source/transducers/devices/`
   - JSON database with geometric/acoustic parameters
   ```rust
   pub struct TransducerDevice {
       name: String,
       manufacturer: String,
       elements: usize,
       geometry: TransducerGeometry,
       frequency_range: (f64, f64),
       max_power: f64,
   }
   ```

4. **Medical Imaging Integration** (P1 Priority)
   - ✅ DICOM support exists (`dicom` crate)
   - ✅ NIfTI support exists (`nifti` crate)
   - Enhancement needed: Higher-level API for segmentation
   ```rust
   pub mod medical_imaging {
       pub fn load_mri(path: &Path) -> Result<Volume3D>;
       pub fn segment_tissue(volume: &Volume3D, tissue_type: TissueType) -> Result<Mask3D>;
       pub fn register_transducer(image: &Volume3D, transducer: &Transducer) -> Result<Transform>;
   }
   ```
   - Location: `clinical/imaging/medical/`

---

## 11. Kranion (Transcranial Focused Ultrasound Visualization)

**Repository:** [https://github.com/jws2f/Kranion](https://github.com/jws2f/Kranion)  
**Language:** Java (90.8%) + GLSL (8.6%)  
**Key Innovation:** Interactive 3D visualization and treatment planning

### Architecture Analysis

#### Core Organization

```
src/main/java/          # Primary Java application
├── Plugins/
│   ├── ACPCPlanPlugin          # Stereotactic planning
│   ├── KranionGroovyConsolePlugin  # Scripting interface
│   └── TractographyPlugin      # Neural pathway visualization
lib/                    # External dependencies
bin/                    # Executable scripts and binaries
```

**Plugin Architecture:**
- Core application provides framework
- Domain-specific functionality as plugins
- Extensions without core modifications

#### Key Features

**1. Visualization & Planning**
- Geometric modeling of treatment procedures
- Interactive 3D rendering (OpenGL 4.5)
- Real-time manipulation

**2. Patient Assessment**
- Skull density ratio computation
- Candidate selection for transcranial procedures
- Clinical metrics (skull thickness, acoustic windows)

**3. Clinical Metrics**
- Skull measurement analysis
- Transducer efficiency modeling
- Phase aberration estimation

#### Technology Stack

- **Language:** Java 1.8+ (64-bit)
- **GPU:** CUDA/OpenGL 4.5
- **Build:** Gradle
- **Medical Imaging:** DICOM import (MR/CT)

**Hardware Requirements:**
> "NVidia GPU with 2 GB or more for interactive performance"

**Design Decision:** GPU-accelerated rendering is core architectural choice

### Recommendations for Kwavers

1. **Interactive Visualization Framework** (P2 Priority)
   - Current `analysis/visualization/` is basic
   - Add 3D volume rendering with `egui` + `wgpu`
   - Real-time field updates during parameter changes
   ```rust
   #[cfg(feature = "gpu-visualization")]
   pub struct InteractiveVisualizer {
       renderer: VolumeRenderer,
       controls: SimulationControls,
       update_rate: f64,  // Hz
   }
   ```
   - Location: `analysis/visualization/interactive/`

2. **Plugin Architecture for Clinical Tools** (P2 Priority)
   - Kwavers has plugin system for physics
   - Extend to application-level plugins
   - Example: Treatment planning, dose calculation, safety monitoring
   ```rust
   pub trait ClinicalPlugin {
       fn name(&self) -> &str;
       fn ui(&mut self, ctx: &egui::Context, state: &SimulationState);
       fn process(&self, data: &SimulationData) -> Result<PluginOutput>;
   }
   ```
   - Location: `clinical/plugins/`

3. **Scripting Interface** (P3 Priority)
   - Kranion has Groovy console for automation
   - Kwavers could add Lua or Rhai scripting
   - Enables batch processing, parameter sweeps
   - Location: `infra/scripting/`

---

## 12. Comparative Analysis: Common Patterns Across All Libraries

### Pattern 1: Four-Layer Configuration Model

**Consensus architecture across all reviewed libraries:**

```
Layer 1: Grid/Domain          → Spatial discretization
Layer 2: Medium               → Material properties
Layer 3: Source/Sensor        → Excitation and measurement
Layer 4: Solver               → Numerical method
```

**Kwavers Alignment:**
```rust
✅ Layer 1: domain::grid::Grid
✅ Layer 2: domain::medium::Medium (trait)
✅ Layer 3: domain::source::Source + domain::sensor::GridSensorSet
✅ Layer 4: solver::forward::* + solver::inverse::*
```

**Status:** ✅ **Fully aligned with industry best practices**

### Pattern 2: Configuration Object Pattern

**All libraries use configuration structs to encapsulate parameters:**

| Library | Configuration Objects |
|---------|----------------------|
| j-Wave | `Medium`, `TimeAxis`, `Domain` |
| k-Wave | `kgrid`, `medium`, `source`, `sensor` |
| k-Wave-Python | `kWaveGrid`, `kWaveMedium`, `kSource`, `kSensor` |
| Fullwave | Grid config, medium config, source config |
| BabelBrain | YAML profiles for thermal, transducer, simulation |

**Kwavers Status:**
- ✅ `simulation::configuration::Configuration`
- ✅ `domain::boundary::PMLConfig`
- Could improve: More granular configuration types

**Recommendation:**
```rust
// Proposed configuration hierarchy
pub struct SimulationConfig {
    pub grid: GridConfig,
    pub medium: MediumConfig,
    pub sources: Vec<SourceConfig>,
    pub sensors: Vec<SensorConfig>,
    pub solver: SolverConfig,
    pub output: OutputConfig,
}
```

### Pattern 3: Pluggable Solver Backends

**Libraries with backend abstraction:**

| Library | Backend Strategy | Kwavers Equivalent |
|---------|------------------|-------------------|
| OptimUS | `bempp-legacy` backend | Plugin system |
| j-Wave | JAX (CPU/GPU/TPU) | `wgpu` feature flag |
| k-Wave | C++/CUDA binaries | GPU shaders |
| BabelBrain | External BabelViscoFDTD | Could integrate external solvers |
| Fullwave | Python/CUDA hybrid | Rust/WGPU hybrid |

**Kwavers Opportunity:**
- Current plugin system is physics-focused
- Add backend abstraction for numerical methods
```rust
pub trait SolverBackend: Send + Sync {
    fn name(&self) -> &str;
    fn step(&mut self, dt: f64) -> Result<()>;
    fn get_field(&self, field_type: FieldType) -> ArrayView3<f64>;
}

pub struct CPUBackend { /* ndarray */ }
pub struct GPUBackend { /* wgpu */ }
pub struct ExternalBackend { /* shell out to C++/CUDA */ }
```

### Pattern 4: Differentiable Simulation Frameworks

**Libraries with autodiff support:**

| Library | Autodiff Framework | Applications |
|---------|-------------------|--------------|
| j-Wave | JAX | Full-wave inversion, parameter estimation |
| DBUA | JAX | Adaptive beamforming, aberration correction |
| Sound Speed Estimation | Manual gradients | Coherence-based optimization |

**Kwavers Status:**
- ✅ `burn` integration for PINNs
- ⚠️ Not integrated with forward solvers
- ⚠️ Beamforming not differentiable

**Recommendation (P1 Priority):**
1. Make beamforming differentiable with burn
2. Add gradient-based parameter estimation to `solver/inverse/estimation/`
3. Implement full-wave inversion (FWI) in `solver/inverse/reconstruction/seismic/fwi/`

### Pattern 5: Medical Imaging Standards Integration

**DICOM/NIfTI support:**

| Library | Medical Imaging Support | Clinical Workflow |
|---------|------------------------|-------------------|
| BabelBrain | NIfTI, DICOM, transforms | Full treatment planning |
| Kranion | DICOM (MR/CT) | Patient assessment, visualization |
| k-Wave | Limited (indirect) | Research-focused |

**Kwavers Status:**
- ✅ `dicom` crate dependency
- ✅ `nifti` crate dependency
- ⚠️ Low-level API only, no high-level segmentation/registration

**Recommendation (P1 Priority):**
```rust
// High-level medical imaging API
pub mod medical {
    pub struct MedicalVolume {
        data: Array3<f64>,
        voxel_size: (f64, f64, f64),
        patient_info: PatientMetadata,
    }
    
    pub fn load_dicom_series(dir: &Path) -> Result<MedicalVolume>;
    pub fn load_nifti(path: &Path) -> Result<MedicalVolume>;
    pub fn segment_tissue(volume: &MedicalVolume, tissue: TissueType) -> Result<Mask3D>;
    pub fn to_acoustic_medium(volume: &MedicalVolume) -> Result<HeterogeneousMedium>;
}
```
- Location: `clinical/imaging/medical/`

### Pattern 6: Heterogeneous Media Support

**All libraries support spatially-varying material properties:**

| Library | Heterogeneous Properties | Implementation |
|---------|-------------------------|----------------|
| k-Wave | Sound speed, density, absorption, nonlinearity | 3D arrays |
| j-Wave | Medium parameters | JAX arrays |
| Fullwave | Power law attenuation | CUDA textures |
| mSOUND | All Westervelt parameters | 3D arrays |
| OptimUS | Multiple domains with interfaces | BEM regions |

**Kwavers Status:**
- ✅ `domain::medium::heterogeneous::HeterogeneousMedium`
- ✅ Supports all properties (sound speed, density, absorption, nonlinearity)
- Could improve: Interface tracking between regions (see OptimUS)

**Recommendation (P1 Priority):**
```rust
// Explicit interface modeling
pub struct MaterialInterface {
    region_a: MaterialId,
    region_b: MaterialId,
    boundary_nodes: Vec<usize>,
    continuity: InterfaceCondition,
}

pub enum InterfaceCondition {
    AcousticContinuity,     // Pressure and normal velocity continuous
    SlidingContact,         // Normal velocity continuous, tangential slip
    RigidBoundary,          // Zero normal velocity
    TransmissionCoeff(f64), // Partial transmission
}
```
- Location: `domain::medium::heterogeneous::interfaces.rs`

### Pattern 7: Multi-Physics Coupling

**Libraries with coupled physics:**

| Library | Coupling | Implementation |
|---------|----------|----------------|
| HITU Simulator | Acoustic ↔ Thermal | Two-way coupling |
| BabelBrain | Acoustic → Thermal | One-way (heating only) |
| mSOUND | Acoustic (nonlinear) | Single physics, advanced |

**Kwavers Status:**
- ✅ `solver::multiphysics::` exists
- ⚠️ Basic implementation only
- ⚠️ No two-way coupling (temperature doesn't affect acoustics)

**Recommendation (P1 Priority):**
- Implement Pennes bioheat equation in `physics::thermal::bioheat.rs`
- Add temperature-dependent acoustic properties
- Create `solver::multiphysics::acoustic_thermal::CoupledSolver`

---

## Summary of Recommendations for Kwavers

### P0 Priority (Critical - Implement Immediately)

| # | Recommendation | Location | Effort | Impact |
|---|---------------|----------|--------|--------|
| 1 | Power law attenuation (Fullwave) | `physics/acoustics/mechanics/absorption.rs` | Medium | High (research accuracy) |
| 2 | Configuration object consolidation | `simulation/configuration/` | Small | High (API ergonomics) |
| 3 | Physics trait system refinement | `domain/physics/` | Medium | High (architecture) |

### P1 Priority (High Value - Next Sprint)

| # | Recommendation | Location | Effort | Impact |
|---|---------------|----------|--------|--------|
| 4 | Differentiable beamforming | `analysis/signal_processing/beamforming/neural/` | Large | High (research) |
| 5 | Medical imaging high-level API | `clinical/imaging/medical/` | Medium | High (clinical) |
| 6 | Material interface tracking | `domain/medium/heterogeneous/interfaces.rs` | Medium | Medium |
| 7 | Bioheat transfer (Pennes) | `physics/thermal/bioheat.rs` | Medium | High (therapy) |
| 8 | Two-way acoustic-thermal coupling | `solver/multiphysics/acoustic_thermal.rs` | Large | High (therapy) |
| 9 | HDF5 I/O for interoperability | `infra/io/hdf5.rs` | Small | Medium |
| 10 | Coherence-based metrics | `analysis/signal_processing/coherence.rs` | Small | Medium |
| 11 | Workflow orchestration | `clinical/workflows/orchestrator.rs` | Medium | Medium |
| 12 | Smart caching with hash signatures | `analysis/performance/cache/simulation_cache.rs` | Small | Medium |

### P2 Priority (Future Enhancement)

| # | Recommendation | Location | Effort | Impact |
|---|---------------|----------|--------|--------|
| 13 | Axisymmetric solver | `solver/forward/pstd/axisymmetric/` | Large | Medium |
| 14 | BEM solver for unbounded domains | `solver/forward/bem/` | Large | Medium |
| 15 | Frequency-specific steady-state | `solver/forward/helmholtz/frequency_specific.rs` | Medium | Low |
| 16 | Multi-GPU domain decomposition | `gpu/distributed/` | Large | Low (niche) |
| 17 | Interactive visualization | `analysis/visualization/interactive/` | Large | Medium |
| 18 | Medium builder DSL | `domain/medium/heterogeneous/builder.rs` | Small | Low |
| 19 | IT'IS tissue database | `domain/medium/properties/tissue_db.json` | Small | Low |
| 20 | Transducer device library | `domain/source/transducers/devices/` | Medium | Low |

### P3 Priority (Nice to Have)

- Scripting interface (Lua/Rhai)
- Neuronavigation system integration
- VR/AR visualization support
- Docker containerization for reproducibility

---

## Key Architectural Insights

### 1. Separation of Concerns is Universal

**All successful libraries separate:**
- **Domain specification** (Grid, Medium, Source, Sensor)
- **Physics equations** (Wave equations, constitutive relations)
- **Numerical methods** (FDTD, PSTD, BEM, FEM)
- **Post-processing** (Beamforming, reconstruction, visualization)

**Kwavers Status:** ✅ **Already follows this pattern**

### 2. Configuration Over Code

**Trend:** Move from programmatic API to configuration-driven workflows

**Evidence:**
- BabelBrain: YAML profiles
- k-Wave: Struct-based configuration
- j-Wave: Composable configuration objects

**Kwavers Opportunity:** Add YAML/TOML-based simulation configuration

### 3. Differentiability is the Future

**Research libraries prioritize gradient-based methods:**
- j-Wave (JAX): Full-wave inversion
- DBUA (JAX): Adaptive beamforming
- PINNs: Parameter estimation

**Kwavers Status:**
- ✅ Burn integration for PINNs
- ⚠️ Not integrated with forward solvers or beamforming

**Action:** Make key operations differentiable (P1 priority)

### 4. Clinical Applications Require Workflow Tools

**BabelBrain and Kranion show clear trend:**
- Three-step workflows (segmentation → simulation → analysis)
- Smart caching for efficiency
- Medical imaging integration (DICOM, NIfTI)
- Treatment planning utilities

**Kwavers Status:**
- ✅ Basic clinical modules exist
- ⚠️ No workflow orchestration
- ⚠️ Low-level medical imaging only

**Action:** Build high-level clinical workflow API (P1 priority)

### 5. GPU Acceleration is Expected

**All modern libraries support GPU:**
- j-Wave: JAX (CPU/GPU/TPU)
- Fullwave: CUDA
- k-Wave: C++/CUDA binaries
- BabelBrain: Multi-backend (Metal/OpenCL/CUDA)

**Kwavers Status:**
- ✅ WGPU support (cross-platform)
- ⚠️ Limited GPU kernel coverage

**Action:** Expand GPU implementation for more solvers

### 6. Multi-Physics Coupling is Critical for Therapy

**HIFU/therapy applications require:**
- Acoustic-thermal coupling (HITU Simulator)
- Nonlinear wave propagation (mSOUND, Fullwave)
- Bioheat transfer (BabelBrain)

**Kwavers Status:**
- ✅ Basic framework exists
- ⚠️ Weak coupling only

**Action:** Implement strong two-way coupling (P1 priority)

---

## Conclusion

Kwavers is **architecturally well-positioned** relative to leading ultrasound simulation libraries. The recent ADR on PINN restructuring aligns perfectly with best practices observed in j-Wave, OptimUS, and DBUA.

### Strengths

1. ✅ **Clean layer separation** (domain, physics, solver, analysis)
2. ✅ **Trait-based physics abstractions** (matches j-Wave, OptimUS philosophy)
3. ✅ **Plugin system for extensibility** (similar to BabelBrain, Kranion)
4. ✅ **Heterogeneous media support** (on par with k-Wave, Fullwave)
5. ✅ **GPU abstraction via wgpu** (matches modern multi-backend trend)

### Gaps to Address

1. ⚠️ **Differentiability integration** → Add burn autodiff to forward solvers/beamforming
2. ⚠️ **Clinical workflow tools** → Build orchestration layer like BabelBrain
3. ⚠️ **Medical imaging API** → High-level segmentation/registration utilities
4. ⚠️ **Power law attenuation** → Heterogeneous frequency-dependent absorption (Fullwave)
5. ⚠️ **Multi-physics coupling** → Strong two-way acoustic-thermal (HITU Simulator)

### Strategic Direction

**Next 3 Sprints:**
1. **Sprint 213:** Differentiable beamforming + power law attenuation (P0/P1)
2. **Sprint 214:** Medical imaging API + interface tracking (P1)
3. **Sprint 215:** Bioheat transfer + acoustic-thermal coupling (P1)

**Long-term Vision:**
- Position kwavers as **Rust-native alternative to k-Wave**
- Target **research + clinical** dual audience
- Differentiation: **Type safety, performance, differentiability**

---

## References

### Primary Sources

1. **j-Wave:** [GitHub](https://github.com/ucl-bug/jwave) | [Documentation](https://ucl-bug.github.io/jwave/)
2. **k-Wave:** [GitHub](https://github.com/ucl-bug/k-wave) | [Website](http://www.k-wave.org/)
3. **k-Wave-Python:** [Documentation](https://k-wave-python.readthedocs.io/)
4. **OptimUS:** [GitHub](https://github.com/optimuslib/optimus)
5. **Fullwave 2.5:** [GitHub](https://github.com/pinton-lab/fullwave25)
6. **Sound Speed Estimation:** [GitHub](https://github.com/JiaxinZHANG97/Sound-Speed-Estimation)
7. **DBUA:** [GitHub](https://github.com/waltsims/dbua)
8. **Kranion:** [GitHub](https://github.com/jws2f/Kranion)
9. **mSOUND:** [GitHub](https://github.com/m-SOUND/mSOUND)
10. **HITU Simulator:** [GitHub](https://github.com/jsoneson/HITU_Simulator)
11. **BabelBrain:** [GitHub](https://github.com/ProteusMRIgHIFU/BabelBrain)

### Key Publications

1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields." *Journal of Biomedical Optics*, 15(2), 021314.

2. Stanziola, A., et al. (2023). "j-Wave: An open-source differentiable wave simulator." *SoftwareX*, 22, 101338.

3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

4. Pinton, G. F., et al. (2009). "A heterogeneous nonlinear attenuating full-wave model of ultrasound." *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 56(3), 474-488.

### Industry Standards

- **IT'IS Foundation:** Tissue property database ([https://itis.swiss/virtual-population/tissue-properties/](https://itis.swiss/virtual-population/tissue-properties/))
- **DICOM Standard:** Medical imaging format ([https://www.dicomstandard.org/](https://www.dicomstandard.org/))
- **NIfTI Format:** Neuroimaging standard ([https://nifti.nimh.nih.gov/](https://nifti.nimh.nih.gov/))

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-22  
**Review Status:** Complete  
**Next Review:** After Sprint 213 implementation
