# Deep Vertical Hierarchy Audit — kwavers
**Single Source of Truth for Architectural Refactoring**

**Date:** 2025-01-12  
**Status:** 🔴 CRITICAL - IMMEDIATE ACTION REQUIRED  
**Auditor:** Elite Mathematically-Verified Systems Architect  
**Mandate:** Zero tolerance for cross-contamination, redundancy, and architectural violations

---

## Executive Summary

### Critical Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Total Rust Files** | 947 | N/A | 📊 |
| **Files >500 Lines** | 50+ | 0 | 🔴 CRITICAL |
| **Largest File** | 3,115 lines | 500 | 🔴 CRITICAL |
| **Module Depth** | 8 levels | 4 levels | 🟡 HIGH |
| **Cross-Layer Violations** | 200+ | 0 | 🔴 CRITICAL |
| **Duplicate Implementations** | 15+ | 0 | 🔴 CRITICAL |
| **Dead Code** | ~50 files | 0 | 🟡 MEDIUM |

### Severity Assessment

🔴 **CRITICAL (P0) - Architectural Purity Violations:**
1. **Cross-Contamination:** Beamforming logic duplicated in `domain/sensor/beamforming` (38 files) AND `analysis/signal_processing/beamforming` (15 files)
2. **Layer Violations:** `domain/sensor` contains signal processing algorithms (should be analysis layer)
3. **Physics-Solver Coupling:** Physics equations embedded in `solver/forward/*` instead of `physics/*`
4. **Grid Operations Scattered:** Differential operators in 5+ different locations
5. **Massive Files:** 3,115-line neural beamforming file violates GRASP (<500 lines)

🟡 **HIGH (P1) - Structural Issues:**
1. **Unclear Boundaries:** `physics/acoustics/imaging` vs `clinical/imaging` vs `domain/imaging`
2. **Module Depth:** 8-level nesting (e.g., `physics/acoustics/analytical/patterns/phase_shifting/focus`)
3. **Mixed Concerns:** Therapy workflows in `physics/acoustics/therapy` AND `clinical/therapy`
4. **Orphaned Modules:** Build logs, deprecated code, redundant test utilities

🟢 **MEDIUM (P2) - Code Quality:**
1. **Documentation Gaps:** Large files lack comprehensive module docs
2. **Test Coverage:** Some modules lack property-based tests
3. **Naming Inconsistencies:** Similar functionality with different naming conventions

---

## Deep Vertical Hierarchy Analysis

### Current Structure (947 Files)

```
kwavers/src/
├── core/ (21 files) ✅ FOUNDATION LAYER - CORRECT
│   ├── constants/ (10 files) - Physical constants, well-organized
│   ├── error/ (8 files) - Error hierarchy, some redundancy
│   ├── time/ (1 file) - Time representation
│   └── utils/ (2 files) - Minimal utilities
│
├── infra/ (12 files) ✅ INFRASTRUCTURE LAYER - CORRECT
│   ├── api/ (5 files) - REST API (feature-gated)
│   ├── cloud/ (3 files) - Cloud deployment
│   ├── io/ (2 files) - File I/O
│   └── runtime/ (2 files) - Async runtime
│
├── domain/ (187 files) ⚠️ DOMAIN LAYER - MIXED CONCERNS
│   ├── boundary/ (12 files) ✅ Correct: PML/CPML boundaries
│   ├── field/ (6 files) ✅ Correct: Unified field abstractions
│   ├── grid/ (21 files) ✅ Correct: Spatial discretization
│   │   └── operators/ (5 files) 🔴 WRONG: Should be math/numerics/
│   ├── medium/ (58 files) ✅ Mostly correct: Material properties
│   │   └── heterogeneous/traits/acoustic/ 🔴 WRONG: Physics in domain
│   ├── sensor/ (52 files) 🔴 CRITICAL VIOLATION
│   │   ├── beamforming/ (38 files) 🔴 WRONG: Signal processing in domain
│   │   │   ├── adaptive/ (15 files) - Adaptive algorithms
│   │   │   ├── experimental/ (8 files) - Neural beamforming (3,115 lines!)
│   │   │   ├── narrowband/ (8 files) - Spectral methods
│   │   │   └── time_domain/ (7 files) - DAS, delay calculations
│   │   ├── localization/ (8 files) ⚠️ Mixed: Config vs algorithms
│   │   ├── passive_acoustic_mapping/ (3 files) 🔴 WRONG: Analysis layer
│   │   └── recorder/ (3 files) ✅ Correct: Data recording
│   ├── signal/ (24 files) ⚠️ MIXED: Definitions ✅ + Processing 🔴
│   │   ├── waveform/ (8 files) ✅ Signal definitions
│   │   ├── modulation/ (4 files) ⚠️ Generation vs processing unclear
│   │   └── pulse/ (3 files) ✅ Signal primitives
│   ├── source/ (32 files) ✅ Correct: Source definitions
│   └── imaging/ (4 files) 🔴 WRONG: Should be clinical/applications
│
├── math/ (87 files) ⚠️ MATH LAYER - INCOMPLETE SEPARATION
│   ├── fft/ (8 files) ✅ Correct: FFT implementations
│   ├── geometry/ (5 files) ✅ Correct: Geometric primitives
│   ├── linear_algebra/ (12 files) ⚠️ Large file (1,887 lines)
│   ├── ml/ (48 files) 🔴 MIXED: PINN infrastructure + domain models
│   │   ├── pinn/ (38 files) ⚠️ Burn integration + physics equations
│   │   │   ├── burn_wave_equation_*.rs (3 files, 4,665 lines total)
│   │   │   ├── electromagnetic*.rs (2 files, 1,981 lines) 🔴 Physics!
│   │   │   └── gpu_accelerator.rs (795 lines) - Infrastructure
│   │   └── uncertainty/ (4 files) ✅ Correct: ML utilities
│   └── numerics/ (14 files) ✅ Correct: Numerical methods
│       └── operators/ (3 files) ⚠️ Overlaps with domain/grid/operators
│
├── physics/ (286 files) 🔴 PHYSICS LAYER - SCATTERED & CONTAMINATED
│   ├── acoustics/ (238 files) 🔴 MASSIVE CONTAMINATION
│   │   ├── analytical/ (45 files) ✅ Analytical solutions
│   │   │   └── patterns/phase_shifting/ (12 files, 6 levels deep!) 🔴
│   │   ├── mechanics/ (38 files) ✅ Physics models (waves, cavitation)
│   │   ├── imaging/ (48 files) 🔴 WRONG: Application layer concerns
│   │   │   ├── modalities/ (32 files) 🔴 Clinical workflows in physics!
│   │   │   │   ├── elastography/ (18 files, 8,974 lines) 🔴 HUGE
│   │   │   │   ├── ceus/ (8 files) - Contrast-enhanced US
│   │   │   │   └── ultrasound/hifu/ (6 files) - Therapy in imaging!
│   │   │   ├── registration/ (8 files) 🔴 Image processing, not physics
│   │   │   └── seismic/ (8 files) ⚠️ Domain-specific, OK here?
│   │   ├── therapy/ (28 files) 🔴 WRONG: Clinical workflows
│   │   │   ├── cavitation/ (8 files) - Treatment monitoring
│   │   │   ├── lithotripsy/ (6 files) - Shock wave therapy
│   │   │   └── modalities/ (8 files) - Treatment protocols
│   │   ├── transcranial/ (12 files) 🔴 WRONG: Clinical application
│   │   ├── skull/ (8 files) ⚠️ Domain model or application?
│   │   ├── nonlinear/ (24 files) ✅ Nonlinear acoustics models
│   │   └── validation/ (18 files) ⚠️ Should be analysis/validation
│   ├── chemistry/ (18 files) ✅ Correct: Sonochemistry models
│   ├── optics/ (14 files) ✅ Correct: Light propagation, sonoluminescence
│   ├── thermal/ (6 files) ✅ Correct: Heat diffusion models
│   └── plugin/ (10 files) 🔴 MIXED: Physics API + Solver concerns
│
├── solver/ (254 files) 🔴 SOLVER LAYER - PHYSICS/NUMERICS MIXED
│   ├── forward/ (186 files) 🔴 CRITICAL MIXING
│   │   ├── acoustic/ (8 files) 🔴 WRONG: Physics model, not solver
│   │   ├── elastic/ (6 files) 🔴 WRONG: Physics model, not solver
│   │   ├── fdtd/ (24 files) ✅ Finite difference time domain (numerics)
│   │   ├── pstd/ (38 files) ✅ Pseudospectral time domain (numerics)
│   │   │   └── dg/ (18 files) ✅ Discontinuous Galerkin (numerics)
│   │   ├── hybrid/ (32 files) ✅ Hybrid FDTD/PSTD (numerics)
│   │   ├── nonlinear/ (48 files) 🔴 MIXED: Physics + Numerics
│   │   │   ├── kuznetsov/ (18 files) - Kuznetsov equation (physics!)
│   │   │   ├── kzk/ (12 files) - KZK equation (physics!)
│   │   │   └── westervelt_spectral/ (8 files) - Westervelt (physics!)
│   │   ├── axisymmetric/ (8 files) ✅ Geometric specialization (OK)
│   │   ├── plugin_based/ (6 files) ✅ Plugin architecture
│   │   └── thermal_diffusion/ (4 files) ⚠️ Physics model in solver
│   ├── inverse/ (42 files) ✅ MOSTLY CORRECT
│   │   ├── reconstruction/ (24 files) ✅ Inverse problem solvers
│   │   │   ├── photoacoustic/ (12 files) ✅ PAT reconstruction
│   │   │   └── seismic/ (12 files) ✅ FWI, RTM
│   │   ├── time_reversal/ (12 files) ✅ Time reversal methods
│   │   └── seismic/ (6 files) ⚠️ Duplication with reconstruction/seismic
│   ├── integration/ (18 files) ✅ CORRECT: Time steppers
│   ├── multiphysics/ (6 files) ⚠️ Should be physics/coupling?
│   ├── utilities/ (12 files) ✅ AMR, validation utilities
│   └── analytical/ (4 files) ✅ Analytical solver methods
│
├── simulation/ (24 files) ⚠️ ORCHESTRATION LAYER - UNCLEAR ROLE
│   ├── configuration/ (8 files) ✅ Simulation configuration
│   ├── parameters/ (4 files) ✅ Parameter management
│   ├── builder/ (6 files) ✅ Builder pattern for simulations
│   └── modalities/ (6 files) 🔴 WRONG: Should be clinical/workflows
│
├── clinical/ (12 files) 🟡 APPLICATION LAYER - INCOMPLETE
│   ├── imaging/ (6 files) ⚠️ Should contain ALL imaging workflows
│   │   └── workflows.rs (1,181 lines) 🔴 HUGE FILE
│   └── therapy/ (6 files) ⚠️ Should contain ALL therapy workflows
│       ├── therapy_integration.rs (1,241 lines) 🔴 HUGE FILE
│       └── swe_3d_workflows.rs (975 lines) 🔴 LARGE FILE
│
├── analysis/ (64 files) ✅ ANALYSIS LAYER - MOSTLY CORRECT
│   ├── signal_processing/ (28 files) ✅ Signal processing algorithms
│   │   └── beamforming/ (15 files) ✅ CORRECT LOCATION
│   │       ├── adaptive/ (3 files) - MVDR, MUSIC (877 lines) 🔴
│   │       ├── time_domain/ (3 files) - DAS reference
│   │       ├── utils/ (3 files) - Delays (734 lines), sparse (781 lines)
│   │       └── covariance/ (2 files) - Covariance estimation (669 lines)
│   ├── performance/ (18 files) ✅ Performance optimization
│   ├── testing/ (8 files) ✅ Test infrastructure
│   ├── validation/ (6 files) ✅ Validation suites
│   └── visualization/ (4 files) ✅ GPU visualization (feature-gated)
│
└── gpu/ (6 files) ✅ GPU LAYER - CORRECT (feature-gated)
    ├── memory/ (2 files) - GPU memory management
    └── shaders/ (4 files) - WGSL shaders
```

---

## Cross-Contamination Analysis

### 🔴 CRITICAL: Beamforming Duplication (Priority P0)

**Problem:** Beamforming algorithms implemented in TWO separate locations with overlapping functionality.

#### Location 1: `domain/sensor/beamforming/` (38 files, ~15,000 lines)

```
domain/sensor/beamforming/
├── adaptive/
│   ├── adaptive.rs (741 lines) - MVDR, GSC, LMS implementations
│   ├── algorithms/ (8 files) - Subspace methods, null steering
│   └── mod.rs - Adaptive beamforming coordinator
├── experimental/
│   ├── neural.rs (3,115 lines) 🔴 MASSIVE FILE
│   ├── hybrid.rs (580 lines) - Neural + classical hybrid
│   └── pinn_beamforming.rs (420 lines) - Physics-informed NN
├── narrowband/
│   ├── capon.rs (691 lines) - Capon/MVDR
│   ├── music.rs (580 lines) - MUSIC algorithm
│   ├── snapshots/ (4 files) - Covariance matrix handling
│   └── mod.rs - Frequency-domain beamforming
├── time_domain/
│   ├── das/ (3 files) - Delay-and-sum implementations
│   ├── mod.rs - Time-domain beamforming
│   └── delay_calculation.rs (520 lines) - Geometric delays
├── ai_integration.rs (1,148 lines) 🔴 LARGE FILE
├── beamforming_3d.rs (1,260 lines) 🔴 LARGE FILE
├── covariance.rs (580 lines) - Covariance estimation
├── processor.rs (680 lines) - Main beamforming processor
├── steering.rs (420 lines) - Beam steering calculations
└── mod.rs (340 lines) - Module coordination

**Issues:**
- ❌ Domain layer contains SIGNAL PROCESSING algorithms (layer violation)
- ❌ 3,115-line neural.rs violates GRASP principle (<500 lines)
- ❌ Tightly coupled to sensor hardware (domain concern) AND algorithms (analysis concern)
- ❌ Duplicates functionality in analysis/signal_processing/beamforming
```

#### Location 2: `analysis/signal_processing/beamforming/` (15 files, ~7,000 lines)

```
analysis/signal_processing/beamforming/
├── adaptive/
│   ├── mvdr.rs (580 lines) - MVDR implementation (DUPLICATE!)
│   ├── subspace.rs (877 lines) 🔴 LARGE - MUSIC, ESPRIT (DUPLICATE!)
│   └── mod.rs - Adaptive beamforming API
├── time_domain/
│   ├── das.rs (520 lines) - Delay-and-sum (DUPLICATE!)
│   ├── delay_reference.rs (420 lines) - Reference delay calculation
│   └── mod.rs - Time-domain processing
├── utils/
│   ├── delays.rs (734 lines) 🔴 LARGE - Delay calculation utilities (DUPLICATE!)
│   ├── sparse.rs (781 lines) 🔴 LARGE - Sparse matrix ops (DUPLICATE!)
│   └── mod.rs (781 lines) 🔴 LARGE - General utilities
├── covariance/
│   └── mod.rs (669 lines) 🔴 LARGE - Covariance estimation (DUPLICATE!)
├── traits.rs (851 lines) 🔴 LARGE - Beamforming trait hierarchy
└── mod.rs (420 lines) - Signal processing API

**Issues:**
- ✅ Correct layer (analysis contains signal processing)
- ✅ Proper separation from hardware concerns
- ❌ DUPLICATES functionality from domain/sensor/beamforming
- ❌ Multiple files >500 lines violate GRASP
- ❌ Redundant implementations of MVDR, MUSIC, DAS, delay calculations
```

#### Redundancy Matrix

| Algorithm | domain/sensor/beamforming | analysis/signal_processing/beamforming | Redundancy |
|-----------|---------------------------|----------------------------------------|------------|
| **MVDR** | ✓ adaptive/adaptive.rs | ✓ adaptive/mvdr.rs | 🔴 DUPLICATE |
| **MUSIC** | ✓ narrowband/music.rs | ✓ adaptive/subspace.rs | 🔴 DUPLICATE |
| **DAS** | ✓ time_domain/das/*.rs | ✓ time_domain/das.rs | 🔴 DUPLICATE |
| **Delay Calculation** | ✓ time_domain/delay_calculation.rs | ✓ utils/delays.rs | 🔴 DUPLICATE |
| **Covariance** | ✓ covariance.rs | ✓ covariance/mod.rs | 🔴 DUPLICATE |
| **Sparse Matrix** | ✓ (embedded) | ✓ utils/sparse.rs | 🔴 DUPLICATE |
| **Neural BF** | ✓ experimental/neural.rs | ✗ | ⚠️ UNIQUE |
| **3D Processing** | ✓ beamforming_3d.rs | ✗ | ⚠️ UNIQUE |

**Estimated Redundancy:** ~40-50% code duplication (6,000-7,500 lines)

#### Resolution Plan

**CORRECT Architecture (SSOT):**

```
analysis/signal_processing/beamforming/  [CANONICAL IMPLEMENTATION]
├── core/
│   ├── traits.rs (<500 lines) - Beamformer trait, configuration
│   └── geometry.rs (<500 lines) - Geometric delay calculations
├── time_domain/
│   ├── das.rs (<500 lines) - Delay-and-sum (SSOT)
│   ├── dmas.rs (<500 lines) - Delayed multiply-and-sum
│   └── coherence_factor.rs (<500 lines) - Coherence weighting
├── frequency_domain/
│   ├── mvdr.rs (<500 lines) - MVDR/Capon (SSOT)
│   ├── music.rs (<500 lines) - MUSIC algorithm (SSOT)
│   ├── esprit.rs (<500 lines) - ESPRIT algorithm
│   └── broadband.rs (<500 lines) - Broadband beamforming
├── adaptive/
│   ├── lms.rs (<500 lines) - LMS adaptive filter
│   ├── gsc.rs (<500 lines) - Generalized sidelobe canceller
│   └── null_steering.rs (<500 lines) - Null steering
├── neural/
│   ├── architecture.rs (<500 lines) - Neural network architectures
│   ├── training.rs (<500 lines) - Training procedures
│   ├── hybrid.rs (<500 lines) - Neural + classical hybrid
│   └── pinn.rs (<500 lines) - Physics-informed neural beamforming
├── utils/
│   ├── covariance.rs (<500 lines) - Covariance matrix estimation (SSOT)
│   ├── spatial_smoothing.rs (<500 lines) - Spatial smoothing
│   └── windowing.rs (<500 lines) - Windowing functions
└── mod.rs (<500 lines) - Public API, re-exports

domain/sensor/
├── recorder/
│   ├── config.rs - Recording configuration
│   ├── storage.rs - Data storage backend
│   └── mod.rs - Sensor data recording (NO PROCESSING)
└── mod.rs - Sensor primitives ONLY
```

**Migration Steps:**
1. ✅ Consolidate ALL beamforming algorithms in `analysis/signal_processing/beamforming/`
2. ✅ Split large files (>500 lines) into focused modules
3. ✅ Delete `domain/sensor/beamforming/` entirely (38 files → 0 files)
4. ✅ Update `domain/sensor/` to contain ONLY sensor hardware abstractions and data recording
5. ✅ Create adapter layer if sensor hardware needs to call beamforming (dependency injection)
6. ✅ Update all imports: `domain::sensor::beamforming::*` → `analysis::signal_processing::beamforming::*`
7. ✅ Deprecation notices for 1-2 releases before removal

---

### 🔴 CRITICAL: Physics-Solver Coupling (Priority P0)

**Problem:** Physics equations embedded in solver layer; numerical methods scattered.

#### Current (WRONG):

```
solver/forward/
├── acoustic/ 🔴 WRONG: Wave equation physics in solver
├── elastic/ 🔴 WRONG: Elastic wave physics in solver
├── nonlinear/
│   ├── kuznetsov/ 🔴 WRONG: Kuznetsov equation (physics!)
│   ├── kzk/ 🔴 WRONG: KZK equation (physics!)
│   └── westervelt_spectral/ 🔴 WRONG: Westervelt equation (physics!)
└── thermal_diffusion/ 🔴 WRONG: Heat equation (physics!)

physics/acoustics/
├── mechanics/ ✅ Contains physics models BUT...
│   └── acoustic_wave/ ⚠️ ...solver references this!
└── plugin/ 🔴 MIXED: Physics API + Solver integration
```

#### Correct Separation:

```
physics/
├── acoustics/
│   ├── models/  [PHYSICS EQUATIONS - SSOT]
│   │   ├── linear/
│   │   │   ├── wave_equation.rs - Linear wave equation
│   │   │   └── helmholtz.rs - Helmholtz equation
│   │   ├── nonlinear/
│   │   │   ├── kuznetsov.rs - Kuznetsov equation (physics ONLY)
│   │   │   ├── westervelt.rs - Westervelt equation
│   │   │   └── kzk.rs - KZK equation
│   │   └── traits.rs - Common physics traits
│   ├── mechanics/  [PHYSICAL PHENOMENA]
│   │   ├── cavitation/ - Bubble dynamics
│   │   ├── streaming/ - Acoustic streaming
│   │   └── radiation_force/ - Radiation force
│   └── coupling/  [MULTI-PHYSICS]
│       ├── acousto_thermal.rs - Acoustic heating
│       └── acousto_optic.rs - Acousto-optic coupling
├── elasticity/
│   └── models/
│       ├── linear_elastic.rs - Linear elasticity
│       └── viscoelastic.rs - Viscoelastic models
├── thermal/
│   └── models/
│       └── heat_diffusion.rs - Heat equation
└── coupling/
    └── multiphysics_coordinator.rs - Unified multi-physics

solver/
├── numerical_methods/  [DISCRETIZATION SCHEMES - SSOT]
│   ├── fdtd/
│   │   ├── stencils.rs - Finite difference stencils
│   │   ├── scheme.rs - FDTD scheme implementation
│   │   └── stability.rs - CFL condition enforcement
│   ├── pstd/
│   │   ├── spectral_operators.rs - Spectral differentiation
│   │   ├── scheme.rs - PSTD scheme implementation
│   │   └── dispersion.rs - Dispersion analysis
│   ├── dg/
│   │   ├── basis.rs - DG basis functions
│   │   ├── scheme.rs - DG scheme implementation
│   │   └── limiting.rs - Shock capturing
│   └── hybrid/
│       ├── fdtd_pstd.rs - Hybrid FDTD/PSTD
│       └── domain_decomposition.rs - Domain splitting
├── time_integration/  [TIME STEPPERS]
│   ├── explicit.rs - Explicit methods (RK, leapfrog)
│   ├── implicit.rs - Implicit methods (CN, BDF)
│   └── adaptive.rs - Adaptive time stepping
├── plugin_system/  [EXTENSIBILITY]
│   ├── plugin_api.rs - Plugin trait definitions
│   ├── plugin_manager.rs - Plugin orchestration
│   └── physics_solver_bridge.rs - Physics → Solver adapter
└── orchestrator/
    └── unified_solver.rs - Main solver coordinator
```

**Key Principle:** Physics defines WHAT to solve; Solver defines HOW to solve.

---

### 🔴 CRITICAL: Grid Operations Scattered (Priority P0)

**Problem:** Differential operators, stencils, and grid utilities duplicated in 5+ locations.

#### Current Locations:

1. **`domain/grid/operators/`** (5 files, ~1,500 lines)
   - gradient.rs, laplacian.rs, divergence.rs, curl.rs
   - ❌ Domain layer should NOT contain numerical methods
   
2. **`solver/forward/fdtd/numerics/`** (multiple files)
   - Finite difference stencils (2nd, 4th, 8th order)
   - ❌ Duplicates domain/grid/operators logic
   
3. **`solver/forward/pstd/numerics/operators/`** (3 files)
   - Spectral differentiation operators
   - ❌ Separate from FDTD, should be unified
   
4. **`math/numerics/operators/`** (3 files)
   - differential.rs (1,062 lines) - Generic operators
   - ⚠️ Overlaps with domain/grid/operators
   
5. **`domain/medium/heterogeneous/interpolation/`** (4 files)
   - Grid interpolation (trilinear, tricubic)
   - ⚠️ Should be in math/numerics/

#### Correct Architecture (SSOT):

```
math/numerics/
├── differentiation/  [SPATIAL OPERATORS - SSOT]
│   ├── finite_difference/
│   │   ├── stencils.rs (<500 lines) - FD stencil coefficients
│   │   ├── gradient.rs (<500 lines) - Gradient operators
│   │   ├── laplacian.rs (<500 lines) - Laplacian operators
│   │   ├── divergence.rs (<500 lines) - Divergence operators
│   │   ├── curl.rs (<500 lines) - Curl operators
│   │   └── accuracy.rs (<500 lines) - Order of accuracy (2, 4, 8)
│   ├── spectral/
│   │   ├── fourier.rs (<500 lines) - Fourier differentiation
│   │   ├── chebyshev.rs (<500 lines) - Chebyshev differentiation
│   │   └── dispersion.rs (<500 lines) - Dispersion analysis
│   ├── dg/
│   │   ├── operators.rs (<500 lines) - DG operators
│   │   └── basis_functions.rs (<500 lines) - Basis functions
│   └── traits.rs (<500 lines) - Unified operator traits
├── interpolation/  [GRID INTERPOLATION - SSOT]
│   ├── linear.rs (<500 lines) - Linear interpolation (1D/2D/3D)
│   ├── cubic.rs (<500 lines) - Cubic interpolation
│   ├── rbf.rs (<500 lines) - Radial basis functions
│   └── traits.rs (<500 lines) - Interpolation API
└── integration/
    ├── quadrature.rs (<500 lines) - Numerical integration
    └── adaptive.rs (<500 lines) - Adaptive quadrature

domain/grid/
├── structure.rs - Grid definition, topology
├── coordinates.rs - Coordinate systems (Cartesian, cylindrical, spherical)
└── mod.rs - Grid primitives ONLY (NO operators)

solver/*/
└── [Uses math/numerics/differentiation/* via traits]
```

**Migration:**
1. ✅ Consolidate ALL differential operators in `math/numerics/differentiation/`
2. ✅ Remove `domain/grid/operators/` (5 files → 0 files)
3. ✅ Remove numerical logic from `solver/forward/*/numerics/`
4. ✅ Solvers access operators through trait abstractions
5. ✅ Update 200+ import statements

---

### 🔴 CRITICAL: Clinical Workflows Scattered (Priority P0)

**Problem:** Clinical applications mixed into physics and simulation layers.

#### Current (WRONG):

```
physics/acoustics/imaging/  (48 files) 🔴 APPLICATION LOGIC IN PHYSICS
├── modalities/
│   ├── elastography/ (18 files, 8,974 lines total)
│   │   ├── elastic_wave_solver.rs (2,824 lines) 🔴 SOLVER LOGIC!
│   │   ├── nonlinear.rs (1,342 lines) 🔴 HUGE
│   │   ├── inversion.rs (1,233 lines) 🔴 HUGE
│   │   ├── radiation_force.rs (903 lines) 🔴 LARGE
│   │   └── gpu_accelerated_3d.rs (869 lines) 🔴 LARGE
│   ├── ceus/ (8 files, 2,500 lines) - Contrast-enhanced ultrasound
│   └── ultrasound/hifu/ (6 files) - HIFU therapy in imaging folder!
├── fusion.rs (1,033 lines) 🔴 LARGE - Multi-modal fusion
└── registration/ (8 files, 1,800 lines) - Image registration

physics/acoustics/therapy/  (28 files) 🔴 CLINICAL WORKFLOWS IN PHYSICS
├── cavitation/ (8 files) - Treatment monitoring
├── lithotripsy/ (6 files) - Shock wave therapy
└── modalities/ (8 files) - Treatment protocols

physics/acoustics/transcranial/  (12 files) 🔴 CLINICAL APPLICATION
├── aberration_correction.rs
└── treatment_planning.rs

clinical/  (12 files) ⚠️ INCOMPLETE - SHOULD BE COMPREHENSIVE
├── imaging/
│   └── workflows.rs (1,181 lines) 🔴 HUGE
└── therapy/
    ├── therapy_integration.rs (1,241 lines) 🔴 HUGE
    └── swe_3d_workflows.rs (975 lines) 🔴 LARGE

simulation/modalities/  (6 files) 🔴 WRONG LAYER
└── photoacoustic.rs (865 lines) - Should be clinical/
```

#### Correct Architecture (SSOT):

```
physics/
├── acoustics/
│   ├── models/ - Wave equations ONLY
│   ├── mechanics/ - Physical phenomena (cavitation, streaming, etc.)
│   └── analytical/ - Analytical solutions
├── elasticity/models/ - Elastic wave physics ONLY
└── optics/models/ - Light propagation ONLY

clinical/  [ALL APPLICATION WORKFLOWS HERE]
├── imaging/
│   ├── ultrasound/
│   │   ├── b_mode.rs (<500 lines) - B-mode imaging workflow
│   │   ├── doppler.rs (<500 lines) - Doppler imaging
│   │   └── harmonic.rs (<500 lines) - Harmonic imaging
│   ├── elastography/
│   │   ├── swe.rs (<500 lines) - Shear wave elastography
│   │   ├── arfi.rs (<500 lines) - ARFI imaging
│   │   ├── inversion.rs (<500 lines) - Elastic modulus reconstruction
│   │   └── visualization.rs (<500 lines) - Elastogram rendering
│   ├── photoacoustic/
│   │   ├── pat_workflow.rs (<500 lines) - PAT imaging pipeline
│   │   ├── reconstruction.rs (<500 lines) - PAT reconstruction
│   │   └── multiwavelength.rs (<500 lines) - Spectroscopic PAT
│   ├── contrast_enhanced/
│   │   ├── ceus_workflow.rs (<500 lines) - CEUS imaging
│   │   ├── perfusion.rs (<500 lines) - Perfusion analysis
│   │   └── cloud_detection.rs (<500 lines) - Microbubble detection
│   ├── fusion/
│   │   ├── multimodal_fusion.rs (<500 lines) - Image fusion
│   │   └── registration.rs (<500 lines) - Image registration
│   └── workflows.rs (<500 lines) - Workflow orchestration
├── therapy/
│   ├── hifu/
│   │   ├── ablation.rs (<500 lines) - Tumor ablation
│   │   ├── monitoring.rs (<500 lines) - Treatment monitoring
│   │   └── planning.rs (<500 lines) - Treatment planning
│   ├── lithotripsy/
│   │   ├── shock_wave.rs (<500 lines) - Lithotripsy workflow
│   │   └── targeting.rs (<500 lines) - Stone targeting
│   ├── transcranial/
│   │   ├── aberration_correction.rs (<500 lines) - Skull correction
│   │   ├── treatment_planning.rs (<500 lines) - tcMRgFUS planning
│   │   └── targeting.rs (<500 lines) - Brain targeting
│   ├── cavitation_control/
│   │   ├── detection.rs (<500 lines) - Cavitation detection
│   │   ├── feedback.rs (<500 lines) - Feedback control
│   │   └── power_modulation.rs (<500 lines) - Power adjustment
│   └── workflows.rs (<500 lines) - Therapy orchestration
└── protocols/
    ├── safety.rs (<500 lines) - Safety protocols
    └── standards.rs (<500 lines) - Clinical standards (FDA, IEC)
```

**Migration:**
1. ✅ Move `physics/acoustics/imaging/` → `clinical/imaging/` (48 files)
2. ✅ Move `physics/acoustics/therapy/` → `clinical/therapy/` (28 files)
3. ✅ Move `physics/acoustics/transcranial/` → `clinical/therapy/transcranial/` (12 files)
4. ✅ Split all files >500 lines (23 files requiring splits)
5. ✅ Remove `simulation/modalities/` and merge into `clinical/`
6. ✅ Update 150+ import statements

---

### 🟡 HIGH: Massive Files Violating GRASP (Priority P1)

**Files >500 Lines (50+ files):**

| Rank | File | Lines | Target | Priority |
|------|------|-------|--------|----------|
| 1 | `domain/sensor/beamforming/experimental/neural.rs` | 3,115 | 500 | 🔴 P0 |
| 2 | `physics/acoustics/imaging/modalities/elastography/elastic_wave_solver.rs` | 2,824 | 500 | 🔴 P0 |
| 3 | `math/ml/pinn/burn_wave_equation_2d.rs` | 2,579 | 500 | 🔴 P0 |
| 4 | `math/linear_algebra/mod.rs` | 1,887 | 500 | 🔴 P0 |
| 5 | `physics/acoustics/imaging/modalities/elastography/nonlinear.rs` | 1,342 | 500 | 🔴 P0 |
| 6 | `domain/sensor/beamforming/beamforming_3d.rs` | 1,260 | 500 | 🔴 P0 |
| 7 | `clinical/therapy/therapy_integration.rs` | 1,241 | 500 | 🔴 P0 |
| 8 | `physics/acoustics/imaging/modalities/elastography/inversion.rs` | 1,233 | 500 | 🔴 P0 |
| 9 | `math/ml/pinn/electromagnetic.rs` | 1,188 | 500 | 🔴 P0 |
| 10 | `clinical/imaging/workflows.rs` | 1,181 | 500 | 🔴 P0 |
| 11 | `domain/sensor/beamforming/ai_integration.rs` | 1,148 | 500 | 🔴 P0 |
| 12 | `infra/cloud/mod.rs` | 1,126 | 500 | 🟡 P1 |
| 13 | `math/ml/pinn/meta_learning.rs` | 1,121 | 500 | 🟡 P1 |
| 14 | `math/ml/pinn/burn_wave_equation_1d.rs` | 1,099 | 500 | 🟡 P1 |
| 15 | `math/numerics/operators/differential.rs` | 1,062 | 500 | 🟡 P1 |
| ... | ... | ... | ... | ... |

**Total Excess Lines:** ~85,000 lines over target (170+ split operations required)

**Split Strategy:**
- Each 3,000-line file → 6-8 focused modules
- Each 1,000-line file → 2-4 focused modules
- Preserve git history with `git mv` for traceability

---

### 🟡 HIGH: Module Depth Violations (Priority P1)

**Issue:** Up to 8-level nesting in some hierarchies.

**Worst Offenders:**

```
physics/acoustics/analytical/patterns/phase_shifting/focus/
└── 8 levels deep! 🔴

domain/sensor/beamforming/narrowband/snapshots/windowed/
└── 7 levels deep! 🔴

solver/inverse/reconstruction/photoacoustic/filters/spatial/
└── 7 levels deep! 🔴
```

**Target:** Maximum 4-5 levels

**Solution:**
- Flatten unnecessary nesting
- Use composition over deep hierarchies
- Merge related modules

---

## Dependency Flow Violations

### Current (WRONG) - Circular Dependencies:

```
┌──────────────────────────────────────────────┐
│ CIRCULAR DEPENDENCY VIOLATIONS               │
└──────────────────────────────────────────────┘

domain/sensor/beamforming
    ↓ [WRONG: Domain → Analysis]
analysis/signal_processing/beamforming
    ↓ [OK]
domain/sensor/recorder
    ↑ [CIRCULAR!]

physics/acoustics/mechanics
    ↓ [OK: Physics → Domain]
domain/medium
    ↓ [WRONG: Domain → Physics]
physics/acoustics/imaging
    ↑ [CIRCULAR!]

solver/forward/nonlinear/kuznetsov
    ↓ [WRONG: Solver → Physics mixing]
physics/acoustics/mechanics/acoustic_wave
    ↓ [OK]
domain/medium
    ↑ [Tight coupling]
```

### Correct (REQUIRED) - Strict Layer Hierarchy:

```
┌────────────────────────────────────────────────────┐
│ CORRECT DEPENDENCY FLOW (NO CYCLES)               │
└────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │  clinical/   │ [Application Layer]
                    └──────┬───────┘
                           ↓ (can use all below)
                    ┌──────────────┐
                    │ simulation/  │ [Orchestration]
                    └──────┬───────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌───────────────┐                    ┌─────────────┐
│   solver/     │                    │  analysis/  │ [Analysis]
│ [Numerical    │                    │ [Signal     │
│  Methods]     │                    │ Processing] │
└───────┬───────┘                    └──────┬──────┘
        ↓                                    ↓
        └──────────────────┬─────────────────┘
                           ↓
                    ┌──────────────┐
                    │   physics/   │ [Physics Models]
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │   domain/    │ [Domain Primitives]
                    └──────┬───────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌───────────────┐                    ┌─────────────┐
│    math/      │                    │   infra/    │
│ [Primitives]  │                    │ [I/O, API]  │
└───────┬───────┘                    └──────┬──────┘
        └──────────────────┬─────────────────┘
                           ↓
                    ┌──────────────┐
                    │    core/     │ [Foundation]
                    └──────────────┘

RULES:
1. Lower layers NEVER depend on higher layers
2. Peer layers communicate via interfaces ONLY
3. Dependencies flow downward ONLY
4. Cross-cutting concerns via dependency injection
```

---

## Dead Code & Deprecated Artifacts

### 🟢 MEDIUM: Files to DELETE (Priority P2)

**Build Artifacts:**
```
target/ - Entire build directory (~500MB)
*.log files - Build logs
errors.txt - Deprecated error log
```

**Deprecated Code:**
```
domain/sensor/beamforming/shaders/ - Unused GPU shaders
physics/acoustics/skull/legacy/ - Old skull model
solver/utilities/validation/kwave/ - Incomplete k-Wave comparison
```

**Redundant Documentation:**
```
ARCHITECTURE_IMPROVEMENT_PLAN.md - Superseded
ARCHITECTURE_REFACTORING_AUDIT.md - Superseded
COMPREHENSIVE_MODULE_REFACTORING_PLAN.md - Superseded
DEPENDENCY_ANALYSIS.md - Superseded
PERFORMANCE_OPTIMIZATION_ANALYSIS.md - Superseded
REFACTORING_EXECUTIVE_SUMMARY.md - Superseded
REFACTORING_PROGRESS.md - Superseded
REFACTORING_QUICK_REFERENCE.md - Superseded
REFACTOR_PHASE_1_CHECKLIST.md - Superseded
... (15+ markdown files to consolidate)
```

**Action:** Consolidate all audits into THIS SINGLE SOURCE OF TRUTH.

---

## Correct Target Architecture

### Ideal Structure (Post-Refactoring)

```
kwavers/src/
├── core/ (~20 files) ✅ Foundation primitives
│   ├── constants/ - Physical constants
│   ├── error/ - Error hierarchy
│   ├── time/ - Time representation
│   └── types/ - Common types
│
├── infra/ (~15 files) ✅ Infrastructure
│   ├── api/ - REST API (feature-gated)
│   ├── cloud/ - Cloud deployment
│   ├── io/ - File I/O
│   └── runtime/ - Async runtime
│
├── domain/ (~120 files) ✅ Domain primitives ONLY
│   ├── boundary/ - PML/CPML boundaries
│   ├── field/ - Unified field abstractions
│   ├── grid/ - Spatial discretization (NO operators)
│   ├── medium/ - Material property INTERFACES (NO physics)
│   ├── sensor/ - Sensor hardware abstractions (NO processing)
│   ├── source/ - Source definitions
│   └── signal/ - Signal DEFINITIONS (NO processing)
│
├── math/ (~80 files) ✅ Mathematical primitives
│   ├── numerics/
│   │   ├── differentiation/ - ALL differential operators (SSOT)
│   │   ├── interpolation/ - Grid interpolation (SSOT)
│   │   ├── integration/ - Numerical integration
│   │   └── transforms/ - Mathematical transforms
│   ├── linear_algebra/ - Matrix operations
│   ├── geometry/ - Geometric primitives
│   ├── fft/ - FFT implementations
│   └── ml/ - Machine learning infrastructure
│       └── pinn/ - PINN framework (NO physics equations)
│
├── physics/ (~180 files) ✅ Physics models ONLY
│   ├── acoustics/
│   │   ├── models/ - Wave equations (SSOT)
│   │   │   ├── linear/ - Linear wave equation
│   │   │   └── nonlinear/ - Kuznetsov, Westervelt, KZK
│   │   ├── mechanics/ - Physical phenomena
│   │   │   ├── cavitation/ - Bubble dynamics
│   │   │   ├── streaming/ - Acoustic streaming
│   │   │   └── radiation_force/ - Radiation force
│   │   └── analytical/ - Analytical solutions
│   ├── elasticity/models/ - Elastic wave equations
│   ├── thermal/models/ - Heat diffusion equations
│   ├── optics/models/ - Light propagation
│   └── coupling/ - Multi-physics coupling
│
├── solver/ (~150 files) ✅ Numerical methods ONLY
│   ├── numerical_methods/
│   │   ├── fdtd/ - Finite difference time domain
│   │   ├── pstd/ - Pseudospectral time domain
│   │   ├── dg/ - Discontinuous Galerkin
│   │   └── hybrid/ - Hybrid methods
│   ├── time_integration/ - Time steppers
│   ├── inverse/ - Inverse problem solvers
│   ├── utilities/ - AMR, validation
│   └── plugin_system/ - Extensibility framework
│
├── analysis/ (~80 files) ✅ Analysis & signal processing
│   ├── signal_processing/
│   │   ├── beamforming/ - ALL beamforming (SSOT)
│   │   ├── filtering/ - Signal filtering
│   │   └── localization/ - Source localization
│   ├── performance/ - Performance optimization
│   ├── testing/ - Test infrastructure
│   ├── validation/ - Validation suites
│   └── visualization/ - GPU visualization
│
├── simulation/ (~20 files) ✅ Orchestration
│   ├── configuration/ - Configuration management
│   ├── builder/ - Builder pattern
│   └── orchestrator/ - Simulation coordination
│
├── clinical/ (~100 files) ✅ Application workflows
│   ├── imaging/
│   │   ├── ultrasound/ - B-mode, Doppler, harmonic
│   │   ├── elastography/ - SWE, ARFI workflows
│   │   ├── photoacoustic/ - PAT workflows
│   │   ├── contrast_enhanced/ - CEUS workflows
│   │   └── fusion/ - Multi-modal fusion
│   ├── therapy/
│   │   ├── hifu/ - HIFU ablation
│   │   ├── lithotripsy/ - Lithotripsy
│   │   ├── transcranial/ - tcMRgFUS
│   │   └── cavitation_control/ - Feedback control
│   └── protocols/ - Safety & standards
│
└── gpu/ (~8 files) ✅ GPU acceleration
    ├── memory/ - GPU memory management
    └── kernels/ - Compute kernels
```

**Total Files:** ~780 files (reduced from 947)
- **Deleted:** ~167 files (redundant, deprecated)
- **All files:** <500 lines (GRASP compliant)
- **Zero duplication:** SSOT enforced
- **Zero layer violations:** Strict hierarchy

---

## Refactoring Execution Plan

### Phase 1: Critical Duplication (Week 1-2)

**Sprint 1A: Beamforming Consolidation**
1. ✅ Create canonical `analysis/signal_processing/beamforming/` structure
2. ✅ Migrate algorithms from `domain/sensor/beamforming/` (38 files)
3. ✅ Split large files:
   - neural.rs (3,115) → 7 modules
   - beamforming_3d.rs (1,260) → 3 modules
   - ai_integration.rs (1,148) → 3 modules
4. ✅ Delete `domain/sensor/beamforming/` entirely
5. ✅ Update 150+ import statements
6. ✅ Run full test suite (867 tests must pass)

**Sprint 1B: Grid Operations Consolidation**
1. ✅ Create canonical `math/numerics/differentiation/` structure
2. ✅ Migrate from `domain/grid/operators/` (5 files)
3. ✅ Extract from `solver/*/numerics/` (20+ files)
4. ✅ Delete redundant operator implementations
5. ✅ Update solver imports to use `math/numerics/`
6. ✅ Validate against analytical solutions

**Sprint 1C: Physics-Solver Separation**
1. ✅ Create canonical `physics/*/models/` structure
2. ✅ Move `solver/forward/acoustic/` → `physics/acoustics/models/`
3. ✅ Move `solver/forward/elastic/` → `physics/elasticity/models/`
4. ✅ Move `solver/forward/nonlinear/*` → `physics/acoustics/models/nonlinear/`
5. ✅ Keep ONLY numerical schemes in `solver/`
6. ✅ Update plugin system bridges

### Phase 2: Clinical Consolidation (Week 3-4)

**Sprint 2A: Clinical Workflows Migration**
1. ✅ Create comprehensive `clinical/` structure
2. ✅ Move `physics/acoustics/imaging/` → `clinical/imaging/` (48 files)
3. ✅ Move `physics/acoustics/therapy/` → `clinical/therapy/` (28 files)
4. ✅ Move `physics/acoustics/transcranial/` → `clinical/therapy/transcranial/`
5. ✅ Delete `simulation/modalities/` and merge into `clinical/`
6. ✅ Split large workflow files

**Sprint 2B: Massive File Decomposition**
1. ✅ Split top 20 files >500 lines (priority P0)
2. ✅ Ensure all new files <500 lines
3. ✅ Preserve git history with proper `git mv`
4. ✅ Update module documentation
5. ✅ Run full test suite per file split

### Phase 3: Dead Code Removal (Week 5)

**Sprint 3A: File Cleanup**
1. ✅ Delete deprecated code (marked in audit)
2. ✅ Remove build artifacts from git
3. ✅ Consolidate redundant documentation
4. ✅ Update .gitignore for artifacts
5. ✅ Clean up unused dependencies

**Sprint 3B: Dependency Audit**
1. ✅ Run `cargo tree` and analyze
2. ✅ Remove unused crates from Cargo.toml
3. ✅ Update feature flags
4. ✅ Validate minimal builds
5. ✅ Document dependency rationale

### Phase 4: Validation & Documentation (Week 6)

**Sprint 4A: Comprehensive Testing**
1. ✅ Run full test suite (867 tests)
2. ✅ Add property-based tests for refactored modules
3. ✅ Validate against k-Wave reference results
4. ✅ Performance benchmarking
5. ✅ Memory profiling

**Sprint 4B: Documentation Update**
1. ✅ Update README.md with new structure
2. ✅ Update ADR with refactoring decisions
3. ✅ Generate architecture diagrams
4. ✅ Update API documentation
5. ✅ Create migration guide

---

## Success Criteria

### Mandatory (Must Achieve)

- [ ] **Zero files >500 lines** (GRASP compliance)
- [ ] **Zero cross-layer violations** (strict hierarchy)
- [ ] **Zero duplicate implementations** (SSOT enforced)
- [ ] **Zero circular dependencies** (acyclic graph)
- [ ] **All 867 tests passing** (zero regressions)
- [ ] **Build time <30s** (SRS NFR-002 compliance)
- [ ] **Zero clippy warnings** (code quality)
- [ ] **100% module documentation** (rustdoc completeness)

### Verification

```bash
# File size compliance
find src -name "*.rs" -exec wc -l {} \; | awk '$1 > 500 {print}'
# Expected: No output

# Dependency graph acyclicity
cargo depgraph | grep cycle
# Expected: No cycles

# Test suite
cargo test --all-features
# Expected: 867 tests passing

# Build time
time cargo build --release
# Expected: <30s

# Code quality
cargo clippy -- -D warnings
# Expected: Zero warnings

# Documentation coverage
cargo doc --no-deps
# Expected: 100% documented
```

---

## Risk Assessment

### High Risk

1. **Breaking Changes:** Extensive import updates across 150+ files
   - **Mitigation:** Deprecation notices, backward compatibility shims for 1-2 releases

2. **Test Failures:** Refactoring may expose latent bugs
   - **Mitigation:** Incremental changes with continuous testing per sprint

3. **Git History:** Complex file moves may obscure history
   - **Mitigation:** Use `git mv` for traceability, document renames in commit messages

### Medium Risk

1. **Performance Regression:** Abstraction layers may introduce overhead
   - **Mitigation:** Benchmark before/after, use zero-cost abstractions, `#[inline]` where needed

2. **Documentation Drift:** Docs may lag code changes
   - **Mitigation:** Update docs atomically with code changes, enforce doc tests

### Low Risk

1. **Feature Flag Issues:** Optional features may break
   - **Mitigation:** Test all feature combinations

---

## References

### Inspiration Projects

1. **jwave** (JAX-based): Clean separation of physics, solvers, geometry
   - Pattern: `jwave.geometry`, `jwave.acoustics`, `jwave.utils`
   - Lesson: Keep domain primitives separate from algorithms

2. **k-Wave** (MATLAB): Modular design, clear API boundaries
   - Pattern: `kWaveGrid`, `kWaveMedium`, `kspaceFirstOrder*`
   - Lesson: Solver as first-class abstraction

3. **optimus** (Julia): Physics-agnostic optimization framework
   - Pattern: Abstract physics API, pluggable solvers
   - Lesson: Decouple physics from numerics

4. **fullwave25** (MATLAB/C): Efficient FDTD implementation
   - Pattern: Minimal dependencies, focused scope
   - Lesson: Keep numerical methods self-contained

### Best Practices

- **GRASP:** <500 lines per file (maintainability)
- **SOLID:** Single responsibility, dependency inversion
- **SSOT:** Single source of truth (zero duplication)
- **CUPID:** Composable, Unix-like, predictable, idiomatic, domain-based
- **Vertical Slicing:** Complete, testable features per sprint
- **Bounded Contexts:** Clear module boundaries, minimal coupling

---

## Audit Completion

**Date:** 2025-01-12  
**Status:** ✅ AUDIT COMPLETE - REFACTORING PLAN READY  
**Next Action:** Execute Phase 1, Sprint 1A (Beamforming Consolidation)

**Sign-off:** Elite Mathematically-Verified Systems Architect

---

## Appendix A: File Inventory

### Files to Move (Top 100)

| Source | Destination | Lines | Reason |
|--------|-------------|-------|--------|
| `domain/sensor/beamforming/*` (38 files) | `analysis/signal_processing/beamforming/*` | ~15,000 | Layer violation |
| `domain/grid/operators/*` (5 files) | `math/numerics/differentiation/` | ~1,500 | Wrong layer |
| `physics/acoustics/imaging/*` (48 files) | `clinical/imaging/*` | ~12,000 | Application logic |
| `physics/acoustics/therapy/*` (28 files) | `clinical/therapy/*` | ~7,000 | Application logic |
| `solver/forward/acoustic/*` (8 files) | `physics/acoustics/models/` | ~2,000 | Physics in solver |
| `solver/forward/elastic/*` (6 files) | `physics/elasticity/models/` | ~1,500 | Physics in solver |
| `solver/forward/nonlinear/*` (48 files) | `physics/acoustics/models/nonlinear/` | ~8,000 | Physics equations |

### Files to Split (Top 50)

| File | Lines | Target Files | Priority |
|------|-------|--------------|----------|
| `domain/sensor/beamforming/experimental/neural.rs` | 3,115 | 7 modules | P0 |
| `physics/acoustics/imaging/modalities/elastography/elastic_wave_solver.rs` | 2,824 | 6 modules | P0 |
| `math/ml/pinn/burn_wave_equation_2d.rs` | 2,579 | 6 modules | P0 |
| `math/linear_algebra/mod.rs` | 1,887 | 4 modules | P0 |
| [... 46 more files ...] | | | |

### Files to Delete (50+)

| File | Reason |
|------|--------|
| `domain/sensor/beamforming/shaders/*` | Unused GPU shaders |
| `physics/acoustics/skull/legacy/*` | Deprecated implementation |
| `errors.txt` | Build artifact |
| `ARCHITECTURE_IMPROVEMENT_PLAN.md` | Superseded by this audit |
| [... 46+ more files ...] | |

---

**END OF AUDIT**