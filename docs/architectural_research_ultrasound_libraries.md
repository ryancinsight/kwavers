# Architectural Research: Ultrasound and Optics Simulation Libraries

**Research Date:** 2026-01-23  
**Purpose:** Extract architectural patterns, module organization, and key features from leading ultrasound simulation libraries to inform kwavers architecture improvements  
**Focus:** Resolve beamforming duplication, layering violations, and clinical-solver coupling

---

## Executive Summary

This research analyzed 11 ultrasound and optics simulation libraries to extract architectural patterns that could address kwavers' current architectural issues:

1. **Beamforming Duplication** - Code exists in both `domain/sensor/beamforming/` and `analysis/signal_processing/beamforming/`
2. **Layering Violations** - Clinical workflows tightly coupled with physics solvers
3. **Clinical-Solver Coupling** - Direct dependencies between high-level workflows and low-level numerics

### Key Findings

- **Separation of Concerns:** Most successful libraries separate physics/solvers, signal processing, and application workflows into distinct modules
- **Composable Design:** JAX-based j-Wave and Python k-Wave emphasize modular "building blocks" that can be combined
- **Backend Abstraction:** Multiple libraries (BabelBrain, Fullwave) use thin API layers over optimized backends (CUDA/C++/FDTD)
- **Physics-First Philosophy:** Sound speed estimation and DBUA maintain physics primacy with optional ML enhancement
- **Clinical Integration Patterns:** BabelBrain demonstrates clean separation through step-based workflows (domain prep → acoustic calc → thermal analysis)

---

## Repository Analysis

### 1. j-Wave: JAX-Based Wave Simulation

**Repository:** https://github.com/ucl-bug/jwave  
**Language:** Python (JAX)  
**Focus:** Differentiable acoustic simulation for ML pipelines

#### Module Organization

```
jwave/
├── jwave/              # Core package with modular components
│   ├── geometry/       # Domain, Medium, TimeAxis
│   ├── acoustics/      # Physics solvers (time_varying)
│   ├── operators/      # FourierSeries field representations
│   └── utils/          # I/O, image loading
├── tests/              # Comprehensive test suite
├── docs/               # Documentation with notebooks
└── scripts/            # Utility scripts
```

#### Architectural Patterns

**1. Functional Composition Over Inheritance**
- Core function: `simulate_wave_propagation(medium, p0, domain, ...)`
- Separate concerns: geometry (Domain), physics (Medium), discretization (TimeAxis)
- No monolithic solver classes

**2. Differentiability as First-Class Concern**
- All operations JAX-compatible for automatic differentiation
- Enables gradient-based optimization throughout pipeline
- Physics models exposed as differentiable functions

**3. Clean Layer Separation**

| Layer | Purpose | Example |
|-------|---------|---------|
| **Geometry** | Spatial/temporal discretization | `Domain`, `TimeAxis` |
| **Physics** | Constitutive relations | `Medium` (sound speed, density) |
| **Acoustics** | Wave equation solvers | `time_varying.simulate_wave_propagation` |
| **Representations** | Field encoding | `FourierSeries` |
| **Utilities** | I/O, data processing | Image loaders, data savers |

**4. GPU Acceleration Pattern**
- Leverage JAX's `@jit` decorator for XLA compilation
- Automatic GPU/TPU dispatch without code changes
- Example:
  ```python
  @jit
  def solver(medium, p0):
      return simulate_wave_propagation(...)
  ```

#### Key Insights for Kwavers

✅ **Adopt composable function-based API** instead of trait hierarchies for solvers  
✅ **Separate geometry, physics, and solver concerns** into independent modules  
✅ **Make differentiability optional** but design-compatible (useful for PINN integration)  
✅ **Thin GPU abstraction** - let backend (wgpu/burn) handle device dispatch

---

### 2. k-Wave: MATLAB Ultrasound Simulation

**Repository:** https://github.com/ucl-bug/k-wave  
**Language:** MATLAB with C++/CUDA backends  
**Focus:** k-space pseudospectral methods

#### Module Organization

```
k-Wave/
├── k-Wave/                 # Core toolbox (flat structure)
│   ├── kWaveGrid.m         # Computational domain
│   ├── kWaveMedium.m       # Material properties
│   ├── kSource.m           # Acoustic sources
│   ├── kSensor.m           # Data collection
│   └── kspaceFirstOrder*.m # Solver implementations
├── binaries/               # Optional C++/CUDA accelerators
└── examples/               # Application demonstrations
```

#### Architectural Patterns

**1. Component Composition Pattern**
- Four distinct component classes: `kWaveGrid`, `kWaveMedium`, `kSource`, `kSensor`
- Orchestrated by solver functions: `kspaceFirstOrder2D`, `kspaceFirstOrder3D`
- Each component independently configurable

**2. Backend Substitution**
- MATLAB implementation serves as reference
- C++/CUDA binaries drop into `binaries/` folder
- Runtime substitution without API changes
- **Kwavers equivalent:** Rust reference + optional GPU kernels

**3. Separation of Solvers and Signal Processing**
- `kWaveArray` class abstracts source/sensor distributions
- Beamforming handled separately from wave propagation
- **Issue in kwavers:** Beamforming mixed with sensor domain logic

**4. Physics Model Abstraction**
- Heterogeneous material parameters (speed, density, absorption)
- Power-law absorption via fractional Laplacian
- Elastic wave propagation as separate module

#### Key Insights for Kwavers

✅ **Adopt component composition:** `Grid + Medium + Source + Sensor → Solver`  
✅ **Backend substitution pattern:** Reference Rust + optional GPU via wgpu  
✅ **Beamforming as separate concern** from solvers (analysis layer only)  
⚠️ **Avoid flat structure** - MATLAB's flat namespace doesn't scale to Rust modules

---

### 3. k-Wave-Python: Python Interface

**Repository:** https://k-wave-python.readthedocs.io  
**Language:** Python  
**Focus:** Python bindings to k-Wave v1.4.0

#### Module Organization

```
kwave/
├── kwave/
│   ├── kgrid.py            # Grid definitions
│   ├── kmedium.py          # Material properties
│   ├── ksource.py          # Source specification
│   ├── ksensor.py          # Sensor configuration
│   ├── simulation.py       # Orchestration layer
│   └── reconstruction/     # Post-processing
│       ├── beamform.py     # Beamforming algorithms
│       └── converters.py   # Data format conversion
└── binaries/               # Pre-compiled GPU binaries
```

#### Architectural Patterns

**1. Orchestration Layer**
- `kWaveSimulation` class manages workflow
- Decouples configuration (grid, medium, source, sensor) from execution
- **Kwavers parallel:** `simulation::configuration::Configuration` + `PluginExecutor`

**2. Beamforming in Reconstruction Module**
- **Critical:** Beamforming lives in `reconstruction/` not `sensor/`
- Signal processing pipeline: acquisition → reconstruction → beamforming
- **Kwavers issue:** Beamforming duplicated in `domain/sensor/` and `analysis/`

**3. Binary Backend Integration**
- Pre-compiled binaries for GPU architectures (sm 5.0 - 9.0a)
- Python layer orchestrates, C++/CUDA executes
- Clean interface: Python config → binary solver → Python results

#### Key Insights for Kwavers

✅ **Move ALL beamforming to analysis layer** (`analysis/signal_processing/beamforming/`)  
✅ **Reconstruction as separate module** from acquisition  
✅ **Orchestration pattern:** Configuration object + executor (already implemented with `PluginExecutor`)  
✅ **GPU binaries via wgpu shaders** - compile at build time, dispatch at runtime

---

### 4. OptimUS: Boundary Element Acoustic Solver

**Repository:** https://github.com/optimuslib/optimus  
**Language:** Python  
**Focus:** Helmholtz equation via BEM

#### Module Organization

```
optimus/
├── optimus/            # Core library
│   ├── physics/        # Helmholtz formulation
│   ├── bem/            # Boundary element methods
│   └── applications/   # Domain-specific (focused ultrasound)
├── docs/               # Documentation
└── notebooks/          # Examples
```

#### Architectural Patterns

**1. Physics-Backend Separation**
- Physics formulation: Helmholtz equation
- Computational backend: bempp-legacy library
- Application layer: Focused ultrasound biomedical engineering

**2. Domain Database Integration**
- IT'IS Foundation Tissue Properties Database V4.1
- External data source for material parameters
- **Kwavers opportunity:** Standardized tissue property database

**3. Containerization for Reproducibility**
- Docker containers ensure consistent environment
- Useful for complex dependency chains (bempp)

#### Key Insights for Kwavers

✅ **Leverage external backends** - Don't reinvent BEM/FEM solvers  
✅ **Tissue property database** - Create standardized `medium::TissueDatabase`  
✅ **Docker for CI/CD** - Ensure reproducible builds and tests  
⚠️ **BEM not primary solver** for kwavers (FDTD/PSTD focus)

---

### 5. Fullwave 2.5: High-Order FDTD Simulation

**Repository:** https://github.com/pinton-lab/fullwave25  
**Language:** Python wrapper + CUDA/C core  
**Focus:** 8th-order spatial, 4th-order temporal FDTD

#### Module Organization

```
fullwave/               # Python package
├── core/               # CUDA/C FDTD engine
├── medium_builder/     # Geometric medium construction
├── transducers/        # Transducer models (linear, convex)
├── beamforming/        # Post-processing (DAS, SAF, PWC)
└── examples/           # Application scenarios
    ├── simple_plane_wave/
    ├── linear_transducer/
    └── convex_transducer/
```

#### Architectural Patterns

**1. Python API + CUDA/C Backend**
- **Python:** User-facing configuration, orchestration
- **CUDA/C:** High-performance FDTD kernel (8th-order staggered grid)
- Clear interface: Python sets up problem → CUDA executes → Python post-processes

**2. Medium Builder Utility**
- Separate tool for constructing computational domains
- Geometric operations: planes, spheres, layers
- **Kwavers equivalent:** `domain/medium/builder` module

**3. Multi-GPU Domain Decomposition**
- Decomposition in depth dimension
- Linear scaling with GPU count
- **Kwavers:** Future work - multi-GPU via wgpu device arrays

**4. Heterogeneous Attenuation Modeling**
- Both α₀ (attenuation coefficient) and γ (exponent) vary spatially
- More accurate than uniform-exponent models
- **Kwavers:** Already supported in `physics/acoustics/absorption`

**5. Beamforming Examples, Not Core**
- Beamforming demonstrated in examples (DAS, SAF, plane-wave compounding)
- Not integrated into FDTD solver itself
- **Critical for kwavers:** Beamforming is post-processing, not solver logic

#### Key Insights for Kwavers

✅ **Thin Rust API + wgpu compute shaders** (similar to Python + CUDA pattern)  
✅ **Medium builder as separate utility** (already exists in kwavers)  
✅ **Beamforming strictly in examples/post-processing** - remove from solver layer  
✅ **Multi-GPU roadmap** - design for future domain decomposition  
✅ **Heterogeneous attenuation** - validate kwavers implementation against Fullwave

---

### 6. Sound Speed Estimation: Coherence-Based Optimization

**Repository:** https://github.com/JiaxinZHANG97/Sound-Speed-Estimation  
**Language:** MATLAB  
**Focus:** Short-lag spatial coherence (SLSC) for sound speed optimization

#### Module Organization

```
Sound-Speed-Estimation/
├── cubdl/                  # CUBDL dataset interface
├── MATLAB_code/
│   ├── iq2rf_jz.m          # IQ to RF conversion
│   ├── slsc_beamformer.m   # SLSC beamforming
│   └── scoring/            # Image quality metrics (CNR, gCNR)
├── datasets/               # Benchmark data
└── example_results/        # Validation outputs
```

#### Architectural Patterns

**1. Physics-First with Optional ML**
- Primary algorithm: Coherence-based optimization (mathematical)
- ML integration: CUBDL benchmark datasets (external)
- **Not ML-first:** Optimization > learned representations

**2. Sequential Pipeline Architecture**
1. **Loading:** IQ data from CUBDL
2. **Conversion:** IQ → RF
3. **Coherence calculation:** Spatial coherence matrices
4. **Optimization:** Max SLSC across sound speed sweep
5. **Quality assessment:** CNR, gCNR, FWHM metrics

**3. Beamformer Composition**
- Two parallel beamformers: DAS (baseline) + SLSC (coherence)
- `zero_out_flag` for runtime adaptability (ROI vs. full image)
- **Design pattern:** Strategy pattern for beamforming algorithms

**4. Multi-Axis Validation**
- **Spatial:** Point targets (FWHM), lesions (contrast/CNR), speckle (SNR)
- **Parametric:** Sound speed sweep analysis
- **Quantitative:** Axial/lateral resolution metrics
- **Visual:** Coherence maps + B-mode images

#### Key Insights for Kwavers

✅ **Coherence-based beamforming** - Add SLSC to `analysis/signal_processing/beamforming/`  
✅ **Strategy pattern for beamformers** - Already partially implemented with traits  
✅ **Multi-metric validation** - Enhance `analysis/validation/` with CNR, gCNR, FWHM  
✅ **Physics-guided ML** - ML enhances physics, doesn't replace it (PINN philosophy)  
⚠️ **Sequential pipeline** - Not optimal for real-time; consider parallel processing

---

### 7. DBUA: Differentiable Beamforming for Autofocusing

**Repository:** https://github.com/waltsims/dbua  
**Language:** Python (JAX)  
**Focus:** Learned delay profiles via differentiable beamforming

#### Module Organization

```
dbua/
├── dbua.py             # Main orchestrator (global config)
├── das.py              # Delay-and-sum beamforming
├── paths.py            # Acoustic propagation (time-of-flight)
├── losses.py           # Phase-error loss functions
└── data/               # MATLAB .mat ultrasound IQ data
```

#### Architectural Patterns

**1. Differentiable Signal Processing**
- Beamforming implemented as differentiable JAX operations
- Enables gradient-based optimization of delay profiles
- Backpropagation through acoustic propagation + beamforming

**2. Hybrid Physics-Learned Architecture**

| Component | Type | Implementation |
|-----------|------|----------------|
| **Propagation** | Physics-based | Speed-of-sound maps, time-of-flight |
| **Delays** | Learned | Optimized via phase-error loss |
| **Beamforming** | Hybrid | DAS with learned delay corrections |

**3. Separation of Concerns**
- **das.py:** Pure beamforming algorithm
- **paths.py:** Physics (acoustic propagation)
- **losses.py:** Optimization objectives
- **dbua.py:** Workflow orchestration

**4. JAX for GPU Acceleration**
- NVIDIA RTX A6000 (48 GB VRAM), CUDA 12.1
- Automatic differentiation + GPU dispatch
- Similar to j-Wave approach

#### Key Insights for Kwavers

✅ **Differentiable beamforming** - Useful for PINN-based reconstruction  
✅ **Separation: algorithm / physics / loss / orchestration** - Clean module boundaries  
✅ **Hybrid approach** - Physics constrains learning (aberration correction)  
✅ **JAX pattern translates to Rust + burn** - AD framework for learned components  
⚠️ **Small codebase (4 files)** - Good for research, needs engineering for production

---

### 8. Kranion: Transcranial Ultrasound Planning

**Repository:** https://github.com/jws2f/Kranion  
**Language:** Java (90.8%) + GLSL shaders (8.6%)  
**Focus:** Treatment planning for transcranial focused ultrasound

#### Module Organization

```
Kranion/
├── src/
│   ├── ACPCPlanPlugin/             # ACPC plane alignment
│   ├── KranionGroovyConsolePlugin/ # Scripting interface
│   └── TractographyPlugin/         # Tractography visualization
├── shaders/                        # GLSL GPU kernels
└── build.gradle                    # Build configuration
```

#### Architectural Patterns

**1. Plugin Architecture**
- Three primary plugins: ACPC, Console, Tractography
- Extensible design - new plugins for additional functionality
- **Kwavers:** Already has plugin system (`solver::plugin::PluginManager`)

**2. Clinical Workflow Features**
- DICOM import (MR/CT studies)
- Skull density ratio calculation
- Phase aberration estimation
- Transducer efficiency modeling
- Treatment plan visualization

**3. GPU-Accelerated Visualization**
- GLSL shaders (8.6% of codebase)
- OpenGL 4.5 requirement
- NVIDIA GPU with 2+ GB VRAM
- **Beamforming likely implemented in shaders** (not confirmed from docs)

**4. Gradle Build System**
- Java ecosystem tooling
- Plugin management via build scripts

#### Key Insights for Kwavers

✅ **Plugin extensibility** - Validate kwavers' plugin system is sufficient  
✅ **DICOM integration** - Add medical imaging I/O (`infra/io/dicom/`)  
✅ **GPU visualization** - Leverage wgpu for both compute and rendering  
⚠️ **Java/GLSL stack** - Different paradigm from Rust; limited direct code reuse  
❓ **Beamforming architecture unclear** - Would need source code access to analyze

---

### 9. mSOUND: Mixed-Domain Acoustic Simulation

**Repository:** https://github.com/m-SOUND/mSOUND  
**Language:** MATLAB  
**Focus:** Generalized Westervelt equation (TMDM + FSMDM)

#### Module Organization

**Note:** Repository overview shows Jekyll documentation site, not source code. Architecture inferred from descriptions.

#### Algorithmic Patterns

**1. Dual Solver Strategy**
- **TMDM (Transient Mixed-Domain Method):** Time-domain pulsed-wave modeling
- **FSMDM (Frequency-Specific Mixed-Domain Method):** Steady-state harmonics
- **Use case:** TMDM for transient analysis, FSMDM for efficiency at specific frequencies

**2. Governing Physics**
- Generalized Westervelt equation (nonlinear acoustics)
- Heterogeneous tissue properties: sound speed, density, attenuation, nonlinearity (B/A)
- Spatial variation in all parameters

**3. Mixed-Domain Approach**
- Hybrid spatial/frequency domain methods
- Reduces computational cost vs. pure time-domain
- **Kwavers:** PSTD already uses mixed domain (spatial frequency, temporal time)

#### Key Insights for Kwavers

✅ **Dual solver strategy** - Transient (FDTD/PSTD) + harmonic (frequency-domain)  
✅ **Westervelt equation** - Already implemented in `physics/acoustics/nonlinear`  
✅ **Heterogeneous nonlinearity** - Validate kwavers supports spatially-varying B/A  
⚠️ **Limited architectural detail** - Would need source code for deeper analysis  
⚠️ **GNU GPL v3.0** - License compatibility check if porting code

---

### 10. HITU Simulator: HIFU Therapy Simulation

**Repository:** https://github.com/jsoneson/HITU_Simulator  
**Language:** MATLAB  
**Focus:** Axisymmetric beams + heating + thermal dose

#### Module Organization

```
HITU_Simulator/
├── BHT.m                   # Bioheat transfer
├── TDNL.m                  # Time-domain nonlinear propagation
├── WAKZK.m                 # Westervelt/KZK models
├── BuildPade*.m            # Padé approximation operators
├── LinearHeating.m         # Linear heating model
├── SynthAxScan.m           # Axial field synthesis
├── SynthRadScan.m          # Radial field synthesis
└── matrixize.m, vektorize.m # Utilities
```

#### Architectural Patterns

**1. Modular Solvers**
- **Propagation:** TDNL, WAKZK (time-domain nonlinear)
- **Heating:** BHT (bioheat transfer), LinearHeating
- **Operators:** Padé approximations for wave equation
- Independent modules composed for therapy simulation

**2. Axisymmetric Optimization**
- Cylindrical symmetry reduces 3D → 2D (r, z)
- Significant computational savings for focused ultrasound
- **Kwavers:** Could add axisymmetric solver variant

**3. Multi-Physics Coupling**
- Acoustic propagation → Heating → Thermal dose
- Sequential coupling: pressure field drives thermal model
- **Kwavers:** Already has `physics/thermal/`, validate coupling

**4. Synthesis Utilities**
- `SynthAxScan`, `SynthRadScan` - field reconstruction from simulations
- Separate from solver core

#### Key Insights for Kwavers

✅ **Modular solver composition** - Acoustic + thermal + dose as separate plugins  
✅ **Axisymmetric solver** - Add as optimization for focused ultrasound  
✅ **Padé operators** - Evaluate for higher-order time integration  
✅ **Field synthesis utilities** - Move to `analysis/visualization/fields/`  
⚠️ **No GPU info** - Likely CPU-only MATLAB

---

### 11. BabelBrain: MRI-Guided HIFU Planning

**Repository:** https://github.com/ProteusMRIgHIFU/BabelBrain  
**Language:** Python  
**Focus:** Transcranial focused ultrasound treatment planning

#### Module Organization

```
BabelBrain/
├── BabelBrain/             # Main GUI application
├── TranscranialModeling/   # Skull modeling & segmentation
├── ThermalModeling/        # Bioheat transfer equation (BHTE)
├── PlanningModels/         # Trajectory planning
└── OfflineBatchExamples/   # Jupyter notebooks for validation
```

#### Architectural Patterns

**1. Three-Step Clinical Workflow**

| Step | Module | Purpose |
|------|--------|---------|
| **Step 1** | TranscranialModeling | Domain preparation (imaging → mesh) |
| **Step 2** | Acoustic calculation | FDTD propagation + source coupling |
| **Step 3** | ThermalModeling | BHTE thermal prediction |

- **Clean separation:** Each step isolated, data passed via Nifti files
- **Kwavers implication:** Clinical workflows should orchestrate, not implement physics

**2. Backend Delegation**
- **Physics solver:** BabelViscoFDTD (external library)
- **Registration:** Elastix (coregistration, resampling)
- **Geometry:** trimesh, optional Blender
- **GUI:** Main BabelBrain application
- **Each concern delegated to specialized tool**

**3. Multi-Backend GPU Support**
- Metal (Apple Silicon)
- OpenCL (AMD/NVIDIA)
- CUDA (NVIDIA)
- Abstraction layer chooses backend at runtime
- **Kwavers:** wgpu already provides similar abstraction

**4. Neuronavigation Integration**
- Designed for Brainsight and 3DSlicer integration
- Invoked from clinical planning software
- Exports results in Nifti for inspection
- **Interoperability over monolithic design**

**5. Testing Strategy**
- Pytest unit tests
- Regression analysis for numerical validation
- Offline batch examples (Jupyter) for parametric studies
- Cross-validation with other tools (Aubry et al., 2022)
- Experimental validation (Pichardo et al., 2017)

**6. Extensibility via Plugins**
- PlanTUS integration (experimental)
- Transducer registry pattern (easily add new devices)
- Advanced options dialogs (tunable parameters without code changes)
- Programmatic API for batch processing

**7. Separation of Imaging and Therapy**
- **Imaging processing:** SimpleITK, Elastix (Step 1)
- **Acoustic simulation:** BabelViscoFDTD (Step 2)
- **Thermal modeling:** BHTE solver (Step 3)
- **Clinical interface:** GUI orchestration layer
- **No cross-contamination** - each module self-contained

#### Key Insights for Kwavers

✅ **Step-based workflow pattern** - Adopt for `clinical/` modules  
✅ **Backend delegation** - Don't reimplement registration/meshing (use external crates)  
✅ **Multi-backend GPU** - wgpu already handles this  
✅ **Nifti I/O for interop** - Add `infra/io/nifti/` (already using nifti crate)  
✅ **Transducer registry** - Standardize in `domain/source/transducers/registry/`  
✅ **Jupyter for validation** - Add `examples/notebooks/` for research workflows  
✅ **Regression testing** - Enhance `analysis/validation/` with numerical baselines  
✅ **Clean separation: imaging ≠ therapy ≠ physics** - Critical architectural principle

---

## Cross-Library Comparative Analysis

### Module Organization Patterns

| Library | Physics | Solvers | Beamforming | Clinical | GPU |
|---------|---------|---------|-------------|----------|-----|
| **j-Wave** | `geometry/` | `acoustics/` | ❌ None | ❌ None | JAX (`@jit`) |
| **k-Wave** | `kWaveMedium` | `kspaceFirstOrder*` | `kWaveArray` | ❌ None | `binaries/` |
| **k-Wave-Python** | `kmedium` | `simulation` | `reconstruction/beamform` | ❌ None | Pre-compiled |
| **OptimUS** | `physics/` | `bem/` | ❌ None | `applications/` | ❌ None |
| **Fullwave** | Medium builder | `core/` (CUDA) | `examples/` | `transducers/` | Native CUDA |
| **Sound-Speed** | IQ/RF | SLSC | `slsc_beamformer` | Scoring | ❌ None |
| **DBUA** | `paths.py` | `das.py` | `das.py` | ❌ None | JAX |
| **Kranion** | ❌ Unclear | ❌ Unclear | Shaders? | Plugins | GLSL |
| **mSOUND** | Westervelt | TMDM/FSMDM | ❌ None | ❌ None | ❌ None |
| **HITU** | WAKZK | TDNL/BHT | Synthesis | ❌ None | ❌ None |
| **BabelBrain** | External FDTD | BabelViscoFDTD | ❌ None | 3-step workflow | Metal/OpenCL/CUDA |

**Key Observations:**
1. **Beamforming placement varies:**
   - k-Wave-Python: `reconstruction/` (post-processing)
   - Sound-Speed: Dedicated module
   - Fullwave: `examples/` only
   - **None integrate beamforming into solvers**

2. **Clinical workflows are separate:**
   - BabelBrain: Dedicated workflow steps
   - OptimUS: `applications/` layer
   - **Never mixed with physics/solvers**

3. **GPU strategies:**
   - JAX: Automatic differentiation + XLA
   - Binaries: Pre-compiled C++/CUDA
   - Native: Direct CUDA/GLSL
   - Abstraction: wgpu-like multi-backend

---

### Beamforming Architecture Patterns

#### Pattern 1: Post-Processing Only (Recommended)
**Examples:** k-Wave-Python, Fullwave  
**Location:** `reconstruction/` or `examples/`  
**Coupling:** Zero coupling to solvers  
**Data flow:** Solver → RF data → Beamformer → Image

```
Solver (pressure field) → Sensor (RF acquisition) → Beamformer (image reconstruction)
```

#### Pattern 2: Dedicated Module (Alternative)
**Examples:** Sound-Speed, DBUA  
**Location:** Standalone module  
**Coupling:** Minimal - only data interfaces  
**Data flow:** IQ/RF data → Beamformer → Metrics

```
Data source (IQ/RF) → Beamformer (algorithm) → Quality metrics
```

#### Pattern 3: Embedded in Sensors (Anti-Pattern for Solvers)
**Examples:** k-Wave (`kWaveArray`)  
**Location:** Sensor abstraction  
**Coupling:** Tight - sensor geometry drives beamforming  
**Issue:** Mixes acquisition (domain) with reconstruction (analysis)

**Kwavers Current State:** Mix of Pattern 2 (good) and Pattern 3 (problematic)
- `analysis/signal_processing/beamforming/` ✅ Correct location
- `domain/sensor/beamforming/` ❌ Layering violation

**Recommendation:** Full migration to Pattern 1
- **Keep:** `analysis/signal_processing/beamforming/` (all algorithms)
- **Remove:** `domain/sensor/beamforming/` (delete or stub references)
- **Sensor role:** Acquire data, export geometry for beamforming calculations

---

### Physics-Clinical Separation Patterns

#### BabelBrain's Three-Step Workflow (Best Practice)

```
┌─────────────────┐
│ Step 1: Domain  │  Clinical layer: imaging → mesh
│   Preparation   │  Tools: SimpleITK, Elastix, trimesh
└────────┬────────┘
         ↓ Nifti files
┌─────────────────┐
│ Step 2: Acoustic│  Physics layer: FDTD solver
│   Calculation   │  Backend: BabelViscoFDTD
└────────┬────────┘
         ↓ Pressure fields
┌─────────────────┐
│ Step 3: Thermal │  Thermal layer: BHTE
│    Analysis     │  Backend: BHTE solver
└─────────────────┘
```

**Key Principles:**
1. **Workflow orchestrates, doesn't implement** - Clinical layer calls physics, doesn't contain physics
2. **Data-driven handoff** - Each step exports data for next step (Nifti, HDF5, etc.)
3. **Backend isolation** - Physics solvers are libraries, not part of workflow code
4. **Reversible pipeline** - Can re-run steps independently (cached intermediate results)

#### Kwavers Current Issues

```
❌ Bad: clinical/imaging/workflows/neural/ai_beamforming_processor.rs
   - Contains beamforming implementation
   - Should only call analysis/signal_processing/beamforming/

❌ Bad: clinical/therapy/ directly calling solver::forward::fdtd
   - Tight coupling to specific solver
   - Should use solver::plugin::PluginExecutor abstraction
```

**Recommended Refactoring:**

```
✅ Good: clinical/imaging/workflows/neural/workflow.rs
   - Orchestrates: data prep → beamforming call → diagnosis
   - Beamforming: calls analysis/signal_processing/beamforming/
   - Diagnosis: uses domain/sensor/beamforming/neural/diagnosis (metrics only)

✅ Good: clinical/therapy/hifu_planning.rs
   - Orchestrates: target selection → trajectory planning → dose prediction
   - Physics: calls solver::plugin::PluginExecutor (not direct solver)
   - Thermal: calls physics::thermal:: (pure physics, no workflow logic)
```

---

### GPU Acceleration Patterns

| Pattern | Libraries | Advantages | Disadvantages | Kwavers Fit |
|---------|-----------|------------|---------------|-------------|
| **JAX Auto-Diff** | j-Wave, DBUA | Automatic GPU, differentiable | Python-only, large runtime | ❌ Not Rust |
| **Pre-Compiled Binaries** | k-Wave, k-Wave-Python | Maximum performance | Build complexity, platform-specific | ⚠️ Possible with wgpu shaders |
| **Native CUDA/C++** | Fullwave | Full control, optimal perf | No portability, CUDA dependency | ❌ Want cross-platform |
| **Multi-Backend Abstraction** | BabelBrain (Metal/OpenCL/CUDA) | Portable, vendor-agnostic | Abstraction overhead | ✅ **wgpu perfect match** |
| **Shader-Based** | Kranion (GLSL) | GPU compute + rendering | OpenGL ecosystem, aging | ✅ wgpu compute shaders |

**Kwavers Strategy:** wgpu compute shaders (WGSL)
- **Advantages:**
  - Cross-platform: Vulkan, Metal, DX12, OpenGL fallback
  - Rust-native: No FFI overhead
  - Unified: Same backend for compute and visualization
  - Future-proof: Modern GPU API (vs. aging OpenGL)

- **Implementation:**
  - FDTD kernels: `src/gpu/kernels/fdtd.wgsl`
  - PSTD kernels: `src/gpu/kernels/pstd.wgsl`
  - Beamforming (if needed): `src/gpu/kernels/beamforming.wgsl`
  - Runtime dispatch: `src/gpu/executor.rs` chooses device

---

### Testing and Validation Strategies

| Library | Unit Tests | Integration Tests | Regression Tests | Validation Data | CI/CD |
|---------|------------|-------------------|------------------|-----------------|-------|
| **j-Wave** | ✅ Yes | ✅ Yes | ❌ No | Notebooks | ✅ GitHub Actions |
| **k-Wave** | ❓ Unknown | ❓ Unknown | ❓ Unknown | ❓ Unknown | ❓ Unknown |
| **k-Wave-Python** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ✅ GitHub Actions |
| **Fullwave** | ✅ `tests/` | ✅ Examples | ❌ No | Phantoms | ❓ Unknown |
| **Sound-Speed** | ❌ No | ❌ No | ✅ CUBDL benchmarks | ✅ CUBDL | ❌ No |
| **BabelBrain** | ✅ pytest | ✅ Jupyter | ✅ Numerical baselines | ✅ Experimental | ✅ GitHub Actions |

**Best Practices (BabelBrain Model):**
1. **Unit tests:** Core algorithms (pytest)
2. **Regression tests:** Numerical baselines (compare against v1.0 results)
3. **Integration tests:** End-to-end workflows (Jupyter notebooks)
4. **Cross-validation:** Compare with other tools (Aubry et al., 2022)
5. **Experimental validation:** Against physical measurements (Pichardo et al., 2017)

**Kwavers Gaps:**
- ✅ Has unit tests (`tests/`)
- ✅ Has benchmarks (`benches/`)
- ⚠️ **Missing:** Regression test suite with numerical baselines
- ⚠️ **Missing:** Cross-validation against k-Wave, Fullwave, etc.
- ⚠️ **Missing:** Experimental validation datasets

---

## Architectural Recommendations for Kwavers

### 1. Beamforming Consolidation (Priority: Critical)

**Problem:** Duplication between `domain/sensor/beamforming/` and `analysis/signal_processing/beamforming/`

**Solution:** Full migration to analysis layer

#### Phase 1: Deprecation (Sprint 213)
```rust
// src/domain/sensor/beamforming/mod.rs
#[deprecated(
    since = "0.8.0",
    note = "Use analysis::signal_processing::beamforming instead. This module will be removed in 0.9.0."
)]
pub mod beamforming_3d;

// Stub implementation that delegates to analysis layer
impl SensorBeamformer {
    pub fn beamform(&self, data: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        // Delegate to analysis layer
        crate::analysis::signal_processing::beamforming::beamform(data, &self.config)
    }
}
```

#### Phase 2: Migration (Sprint 214)
1. Move all algorithmic code to `analysis/signal_processing/beamforming/`
2. Update clinical workflows to import from analysis layer
3. Sensor module provides only geometry and acquisition

#### Phase 3: Removal (Sprint 215)
1. Delete `domain/sensor/beamforming/` (except minimal geometry traits)
2. Update documentation and examples
3. Verify all tests pass

**Pattern:** k-Wave-Python's `reconstruction/beamform` model

---

### 2. Clinical-Solver Decoupling (Priority: High)

**Problem:** `clinical/imaging/workflows/` directly imports solver implementations

**Solution:** BabelBrain's step-based workflow pattern

#### Refactoring Template

**Before:**
```rust
// clinical/imaging/workflows/advanced_imaging.rs (BAD)
use crate::solver::forward::fdtd::FdtdSolver;

impl AdvancedImagingWorkflow {
    fn run_simulation(&self) -> KwaversResult<PressureField> {
        let mut solver = FdtdSolver::new(config)?;  // Direct coupling!
        solver.step()?;
        // ...
    }
}
```

**After:**
```rust
// clinical/imaging/workflows/advanced_imaging.rs (GOOD)
use crate::solver::plugin::{PluginExecutor, PluginManager};
use crate::clinical::imaging::config::ImagingConfig;

impl AdvancedImagingWorkflow {
    fn run_simulation(&self) -> KwaversResult<PressureField> {
        // Step 1: Domain preparation (clinical layer responsibility)
        let domain = self.prepare_domain()?;
        
        // Step 2: Acoustic calculation (delegate to solver abstraction)
        let executor = PluginExecutor::new(self.solver_config)?;
        let pressure_field = executor.execute(&domain)?;
        
        // Step 3: Beamforming (delegate to analysis layer)
        let beamformed = crate::analysis::signal_processing::beamforming::beamform(
            &pressure_field,
            &self.beamform_config,
        )?;
        
        Ok(beamformed)
    }
}
```

**Benefits:**
- Clinical workflow testable without solver implementation
- Can swap solvers (FDTD → PSTD → BEM) via configuration
- Follows BabelBrain's orchestration pattern

---

### 3. Module Boundary Enforcement (Priority: Medium)

**Problem:** No automated enforcement of layering rules

**Solution:** Implement architecture lints in `architecture.rs`

#### Allowed Dependencies (Downward Only)

```
Clinical ─┬─> Solver (via PluginExecutor only)
          ├─> Analysis
          ├─> Domain
          └─> Physics

Analysis ──┬─> Domain
           ├─> Physics
           └─> Math

Solver ────┬─> Physics
           ├─> Domain
           └─> Math

Domain ────┬─> Physics
           └─> Math

Physics ───> Math

Infra ─────> All (cross-cutting)
```

#### Forbidden Dependencies (Upward/Lateral)

```
❌ Domain ──X─> Solver
❌ Domain ──X─> Analysis
❌ Domain ──X─> Clinical
❌ Physics ─X─> Domain
❌ Math ────X─> Physics
❌ Solver ──X─> Clinical
```

#### Implementation

```rust
// src/architecture.rs
pub struct DependencyChecker;

impl DependencyChecker {
    pub fn check_module_dependencies(module: &str) -> Result<(), Vec<String>> {
        let violations = Vec::new();
        
        match module {
            "domain" => {
                // Domain cannot depend on solver, analysis, or clinical
                check_no_imports(module, &["solver", "analysis", "clinical"])?;
            }
            "physics" => {
                // Physics cannot depend on domain
                check_no_imports(module, &["domain", "solver", "analysis", "clinical"])?;
            }
            "solver" => {
                // Solver cannot depend on clinical or analysis
                check_no_imports(module, &["clinical", "analysis"])?;
            }
            _ => {}
        }
        
        Ok(())
    }
}

// Add to CI/CD
// cargo test --test architecture_tests
```

**Pattern:** Hexagonal architecture dependency rules

---

### 4. Tissue Property Database (Priority: Low)

**Problem:** No standardized tissue properties (scattered constants)

**Solution:** OptimUS's database integration pattern

#### Implementation

```rust
// src/domain/medium/tissue_database.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueProperties {
    pub name: String,
    pub sound_speed: f64,       // m/s
    pub density: f64,            // kg/m³
    pub attenuation: f64,        // dB/cm/MHz
    pub nonlinearity: f64,       // B/A ratio
    pub thermal_conductivity: f64,
    pub specific_heat: f64,
    pub perfusion: f64,
}

pub struct TissueDatabase {
    tissues: HashMap<String, TissueProperties>,
}

impl TissueDatabase {
    pub fn load_itis_foundation() -> KwaversResult<Self> {
        // Load IT'IS Foundation database
        // https://itis.swiss/virtual-population/tissue-properties/
    }
    
    pub fn get(&self, tissue: &str) -> Option<&TissueProperties> {
        self.tissues.get(tissue)
    }
}

// Usage
let db = TissueDatabase::load_itis_foundation()?;
let brain = db.get("brain_grey_matter").unwrap();
let medium = Medium::from_tissue(brain, &grid)?;
```

**Data Source:** IT'IS Foundation Tissue Properties Database V4.1 (same as OptimUS)

---

### 5. Multi-Backend GPU Strategy (Priority: Medium)

**Problem:** Current wgpu implementation not consistently applied

**Solution:** BabelBrain's multi-backend pattern (already supported by wgpu)

#### Architecture

```rust
// src/gpu/backend.rs
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    OpenGL,
}

impl GpuBackend {
    pub fn auto_select() -> Self {
        // wgpu::Instance::new() handles this automatically
        // Just expose for user configuration
    }
}

// src/gpu/executor.rs
pub struct GpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

impl GpuExecutor {
    pub fn new(backend: Option<GpuBackend>) -> KwaversResult<Self> {
        // Create wgpu instance with specified backend
        // Compile all shaders (FDTD, PSTD, beamforming if needed)
    }
    
    pub fn execute_kernel(&self, kernel: &str, data: &GpuBuffer) -> KwaversResult<GpuBuffer> {
        let pipeline = self.pipelines.get(kernel).ok_or(...)?;
        // Dispatch compute shader
    }
}
```

**No change needed:** wgpu already handles multi-backend. Just expose configuration.

---

### 6. Regression Testing Framework (Priority: High)

**Problem:** No numerical baselines for regression testing

**Solution:** BabelBrain's validation strategy

#### Implementation

```rust
// tests/regression/mod.rs
use kwavers::testing::RegressionTest;

#[test]
fn regression_fdtd_homogeneous_medium() {
    let test = RegressionTest::load("fdtd_homogeneous_v1.0.json")?;
    
    // Run current implementation
    let result = run_fdtd_simulation(&test.config)?;
    
    // Compare against baseline
    test.assert_pressure_field_matches(&result.pressure, tolerance = 1e-6)?;
    test.assert_max_relative_error(&result.pressure, max_error = 1e-4)?;
}

// Baselines stored in tests/regression/baselines/
// Format: JSON with metadata + compressed pressure field
{
    "version": "1.0.0",
    "test_name": "fdtd_homogeneous",
    "config": { ... },
    "baseline": {
        "pressure_field": "base64_compressed_data",
        "checksum": "sha256_hash"
    }
}
```

**Cross-Validation Tests:**
```rust
#[test]
#[ignore] // Requires external tools
fn cross_validate_against_kwave() {
    // Export kwavers config to k-Wave MATLAB script
    // Run k-Wave simulation
    // Import k-Wave results
    // Compare pressure fields
}
```

---

### 7. Axisymmetric Solver Optimization (Priority: Low)

**Problem:** 3D simulations expensive for focused ultrasound

**Solution:** HITU's axisymmetric approach

#### Use Case
- Focused ultrasound (HIFU therapy)
- Cylindrical symmetry: φ-independence
- Reduce 3D (x, y, z) → 2D (r, z)
- Computational savings: ~100x for typical grids

#### Implementation

```rust
// src/solver/forward/fdtd_axisymmetric.rs
pub struct FdtdAxiSymmetric {
    grid_r: Array1<f64>,  // Radial points
    grid_z: Array1<f64>,  // Axial points
    // 2D arrays instead of 3D
}

impl Solver for FdtdAxiSymmetric {
    fn step(&mut self) -> KwaversResult<()> {
        // Solve in (r, z) with appropriate source term modifications
        // Handle r=0 singularity carefully
    }
}
```

**When to Use:**
- ✅ Single-element focused transducers
- ✅ Concentric phased arrays
- ❌ Linear arrays (breaks symmetry)
- ❌ General 3D heterogeneities

---

## Summary of Architectural Lessons

### Critical Findings

1. **Beamforming Belongs in Post-Processing**
   - k-Wave-Python: `reconstruction/beamform`
   - Fullwave: `examples/` only
   - **Never integrated into solvers**

2. **Clinical Workflows Orchestrate, Don't Implement**
   - BabelBrain: 3-step workflow delegates to backends
   - Cleanly separated: imaging ≠ therapy ≠ physics

3. **Backend Abstraction Enables Flexibility**
   - BabelBrain: Metal/OpenCL/CUDA via abstraction
   - k-Wave: MATLAB reference + optional C++/CUDA
   - **wgpu provides this for kwavers**

4. **Physics-First with Optional ML Enhancement**
   - Sound-Speed: Coherence optimization > learned features
   - DBUA: Physics constrains learning (aberration correction)
   - **PINN philosophy aligns with this**

5. **Validation Requires Multiple Axes**
   - Unit tests (algorithms)
   - Regression tests (numerical baselines)
   - Cross-validation (against other tools)
   - Experimental validation (physical measurements)

---

## Recommended Action Plan for Kwavers

### Sprint 213: Beamforming Consolidation
- [ ] Deprecate `domain/sensor/beamforming/` modules
- [ ] Ensure all algorithms exist in `analysis/signal_processing/beamforming/`
- [ ] Update clinical workflows to import from analysis layer only
- [ ] Add deprecation warnings with migration guide

### Sprint 214: Clinical-Solver Decoupling
- [ ] Refactor `clinical/imaging/workflows/` to use `PluginExecutor`
- [ ] Remove direct solver imports from clinical modules
- [ ] Implement step-based workflow pattern (BabelBrain model)
- [ ] Add workflow integration tests (end-to-end without mocking physics)

### Sprint 215: Architecture Enforcement
- [ ] Implement `DependencyChecker` in `architecture.rs`
- [ ] Add CI/CD check for layering violations
- [ ] Document allowed dependency graph
- [ ] Create architectural decision records (ADRs)

### Sprint 216: Regression Testing
- [ ] Create `tests/regression/` directory
- [ ] Implement `RegressionTest` framework
- [ ] Generate baselines for FDTD, PSTD, PINN
- [ ] Add cross-validation tests (k-Wave, Fullwave)

### Sprint 217: Tissue Database
- [ ] Implement `TissueDatabase` with IT'IS Foundation data
- [ ] Add serialization/deserialization (JSON, TOML)
- [ ] Create builder pattern: `Medium::from_tissue()`
- [ ] Document tissue property sources (citations)

### Sprint 218: GPU Backend Refinement
- [ ] Expose wgpu backend selection in configuration
- [ ] Benchmark across Vulkan/Metal/DX12
- [ ] Optimize shader compilation (build-time pre-compilation)
- [ ] Add GPU capability detection and fallback logic

---

## References

1. **j-Wave:** Stanziola, A., et al. "j-Wave: An open-source differentiable wave simulator." GitHub, 2023.
2. **k-Wave:** Treeby, B.E., and Cox, B.T. "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields." J. Biomed. Opt., 2010.
3. **BabelBrain:** Pichardo, S., et al. "Multi-frequency characterization of the speed of sound and attenuation coefficient for longitudinal transmission of freshly excised human skulls." Phys. Med. Biol., 2011.
4. **Fullwave:** Pinton, G.F., et al. "A heterogeneous nonlinear attenuating full-wave model of ultrasound." IEEE Trans. UFFC, 2009.
5. **Sound-Speed Estimation:** Zhang, J., et al. "Sound speed estimation using short-lag spatial coherence." IEEE Trans. UFFC, 2020.
6. **DBUA:** Simmons, W., et al. "Differentiable beamforming for ultrasound autofocusing." arXiv, 2023.
7. **IT'IS Foundation:** Tissue Properties Database V4.1, https://itis.swiss/virtual-population/tissue-properties/

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-23  
**Reviewed By:** Architecture Research (Automated Analysis)  
**Next Review:** Sprint 220 (Post-Refactoring Validation)
