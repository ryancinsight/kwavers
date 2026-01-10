# Corrected Deep Vertical Hierarchy Audit
**Kwavers Architecture Analysis - Proper Vertical Layering**

**Date**: 2024-01-09  
**Version**: 2.14.0  
**Status**: 🔴 CRITICAL - Architecture Violations  
**Objective**: Establish CORRECT deep vertical hierarchy with clear dependency flow

---

## Executive Summary

**CORRECTION**: The issue is NOT that the hierarchy is too deep. The issue is that:

1. **The hierarchy doesn't reflect actual architectural dependencies**
2. **Shared components are NOT at the bottom where they should be**
3. **Cross-contamination violates vertical dependency flow**
4. **Redundancy exists because common code isn't extracted to shared layers**

### What "Deep Vertical Hierarchy" Actually Means

```
CORRECT Deep Vertical Hierarchy:
┌─────────────────────────────────────────┐
│  Application Layer (Specific)           │ ← Most specific
│  Uses: Everything below                 │
├─────────────────────────────────────────┤
│  Domain Logic Layer                     │
│  Uses: Services, Math, Core             │
├─────────────────────────────────────────┤
│  Service Layer                          │
│  Uses: Math, Core                       │
├─────────────────────────────────────────┤
│  Mathematical Operations                │
│  Uses: Core only                        │
├─────────────────────────────────────────┤
│  Core/Foundation (Shared)               │ ← Most generic
│  Uses: Nothing (pure foundation)        │
└─────────────────────────────────────────┘

Rules:
✅ Higher layers depend on lower layers (downward arrows only)
✅ Lower layers are MORE generic (widely reusable)
✅ Upper layers are MORE specific (domain-specific)
✅ Shared components sink to the lowest appropriate layer
✅ Each layer has single, clear responsibility
```

---

## Problem Analysis: Current vs. Correct Hierarchy

### Current Problem: Inverted Dependencies

```
❌ CURRENT (BROKEN):
core/constants/thermodynamic.rs:
    pub use crate::physics::constants::GAS_CONSTANT;
    
    Foundation layer depending on upper layer!
    This INVERTS the hierarchy!

core/utils/mod.rs:
    pub use crate::math::fft::*;
    
    Foundation importing from middle layer!
    Breaks vertical structure!
```

### Correct Structure: Strict Vertical Flow

```
✅ CORRECT:
core/constants/fundamental.rs:
    pub const GAS_CONSTANT: f64 = 8.314462618;
    
    Constant defined at foundation (lowest layer)
    
physics/constants/mod.rs:
    pub use crate::core::constants::GAS_CONSTANT;
    
    Upper layer RE-EXPORTS from foundation
    Dependency flows DOWNWARD ✓
```

---

## Root Cause: Misunderstanding of "Shared Components"

### The Problem

**Shared components are scattered across WRONG layers:**

| Component | Current Location | Problem | Correct Location |
|-----------|-----------------|---------|------------------|
| Constants | `physics/constants/` | Too high - used by everyone | `core/constants/` |
| FFT | `math/fft/` | Correct! | `math/fft/` ✓ |
| FFT Re-export | `core/utils/` | Core shouldn't know about math | Remove re-export |
| Grid Operators | `domain/grid/operators/` | Too high - used by solvers | `math/numerics/operators/` |
| Sparse Matrix | `core/utils/sparse_matrix/` | Core too generic | `math/linear_algebra/sparse/` |
| Test Helpers | `core/utils/test_helpers.rs` | Core shouldn't know domain | `tests/support/fixtures.rs` |

### The Principle

**Shared Component Placement Rule**:
```
IF component used by multiple modules at SAME level
THEN sink to layer BELOW them

IF component generic (no domain knowledge)
THEN sink to lowest possible layer

IF component domain-specific
THEN keep at appropriate domain layer
```

---

## Correct Deep Vertical Architecture

### Layer 0: Foundation (Core)

```
src/core/
├── constants/              # ALL constants (most widely shared)
│   ├── fundamental.rs      # Universal: c, G, h, k_B, R_GAS
│   ├── physics.rs          # Physics: sound speeds, densities
│   ├── numerical.rs        # Numerical: tolerances, CFL limits
│   └── mod.rs
├── error/                  # Error types (widely shared)
├── time/                   # Time utilities (generic)
└── types/                  # Generic type definitions
    ├── scalar.rs           # Scalar types
    └── indices.rs          # Index types

Dependencies: NONE (pure foundation)
Exports TO: Everyone
Purpose: Most generic, widely reusable foundation
```

**Rule**: Core knows NOTHING about upper layers (math, domain, physics, solver)

---

### Layer 1: Mathematical Operations (Math)

```
src/math/
├── fft/                    # Fast Fourier Transform
│   ├── processor.rs        # Core FFT implementation
│   ├── cache.rs            # FFT plan caching
│   └── operators.rs        # FFT-based operators
├── linear_algebra/         # Matrix operations
│   ├── dense.rs            # Dense matrices
│   ├── sparse/             # Sparse matrices (MOVED FROM core)
│   │   ├── csr.rs          # CSR format
│   │   ├── solver.rs       # Sparse solvers
│   │   └── eigenvalue.rs   # Eigenvalue problems
│   └── decomposition.rs    # Matrix decompositions
├── geometry/               # Geometric operations
│   ├── transformations.rs  # Coordinate transforms
│   ├── distance.rs         # Distance metrics
│   └── interpolation.rs    # Spatial interpolation
├── numerics/               # Numerical methods (SHARED)
│   ├── operators/          # Differential operators (MOVED FROM domain/grid)
│   │   ├── gradient.rs     # Gradient operators
│   │   ├── laplacian.rs    # Laplacian operators
│   │   └── divergence.rs   # Divergence operators
│   ├── integration/        # Numerical integration
│   │   ├── quadrature.rs   # Quadrature rules
│   │   └── ode.rs          # ODE solvers (generic)
│   └── optimization/       # Generic optimizers
└── ml/                     # Machine learning (GENERIC framework only)
    ├── pinn/               # Generic PINN framework
    │   ├── physics_traits.rs  # Abstract interfaces
    │   ├── training.rs     # Training infrastructure
    │   └── inference.rs    # Inference engine
    └── uncertainty/        # Uncertainty quantification

Dependencies: core/ only
Exports TO: domain/, physics/, solver/, analysis/
Purpose: Generic mathematical operations, no domain knowledge
```

**Rule**: Math knows about Core, but NOT about Domain/Physics/Solver

**Key Moves**:
- ✅ Sparse matrices: `core/utils/sparse_matrix/` → `math/linear_algebra/sparse/`
- ✅ Differential operators: `domain/grid/operators/` → `math/numerics/operators/`
- ✅ Generic PINN: stays in `math/ml/pinn/` (abstract framework)

---

### Layer 2: Domain Infrastructure (Domain)

```
src/domain/
├── grid/                   # Computational mesh (infrastructure)
│   ├── structure.rs        # Grid data structure
│   ├── indexing.rs         # Index management
│   └── topology.rs         # Grid topology
│   # REMOVED: operators/ → moved to math/numerics/operators/
├── medium/                 # Material properties (data containers)
│   ├── homogeneous/        # Uniform media
│   ├── heterogeneous/      # Spatially-varying media
│   │   ├── core/           # Core heterogeneous logic
│   │   ├── interpolation/  # Property interpolation
│   │   └── tissue/         # Tissue models (data)
│   └── properties/         # Property accessors
├── sensor/                 # Data acquisition (infrastructure only)
│   ├── recorder/           # Recording infrastructure
│   │   └── storage/        # Data storage
│   └── geometry/           # Sensor geometry
│   # REMOVED: beamforming/ → moved to analysis/signal_processing/
├── source/                 # Source geometry (infrastructure only)
│   ├── geometry/           # Transducer geometry
│   │   ├── array.rs        # Array layouts
│   │   └── focused.rs      # Focused transducers
│   └── waveform/           # Waveform definitions (data)
├── signal/                 # Signal definitions (data)
│   ├── waveform/           # Waveform types
│   └── modulation/         # Modulation schemes
├── boundary/               # Boundary conditions
│   └── cpml/               # CPML implementation
└── field/                  # Field management
    ├── indices.rs          # Field indices (SHARED across all solvers)
    └── mapping.rs          # Field accessors

Dependencies: core/, math/
Exports TO: physics/, solver/, analysis/, clinical/
Purpose: Domain data structures and infrastructure (NO algorithms/physics)
```

**Rule**: Domain provides DATA and INFRASTRUCTURE, not BEHAVIOR

**Key Principle**: 
- Grid stores coordinates → Math provides operators that act on it
- Medium stores properties → Physics provides models that use them
- Sensor stores geometry → Analysis provides algorithms that process data

---

### Layer 3: Physics Models (Physics)

```
src/physics/
├── acoustics/              # Acoustic physics
│   ├── models/             # Physical models (DEEP hierarchy for specificity)
│   │   ├── linear/         # Linear acoustics
│   │   │   ├── wave_equation.rs
│   │   │   └── helmholtz.rs
│   │   ├── nonlinear/      # Nonlinear acoustics
│   │   │   ├── westervelt.rs
│   │   │   ├── kuznetsov.rs
│   │   │   └── kzk.rs
│   │   └── elastic/        # Elastic waves
│   │       ├── elastic_wave.rs
│   │       └── mode_conversion/
│   ├── mechanics/          # Physical phenomena
│   │   ├── cavitation/     # Bubble dynamics
│   │   │   ├── rayleigh_plesset.rs
│   │   │   ├── keller_miksis.rs
│   │   │   ├── gilmore.rs
│   │   │   └── pinn/       # Cavitation-SPECIFIC PINN
│   │   │       ├── model.rs      # Uses math/ml/pinn traits
│   │   │       └── training.rs   # Cavitation-specific training
│   │   ├── streaming/      # Acoustic streaming
│   │   └── absorption/     # Absorption models
│   ├── analytical/         # Analytical solutions
│   │   ├── plane_wave.rs
│   │   ├── spherical_wave.rs
│   │   └── patterns/       # Beam patterns
│   │       ├── gaussian.rs
│   │       ├── bessel.rs
│   │       └── focusing/
│   └── transducer/         # Transducer physics (BEHAVIOR)
│       ├── radiation.rs    # Radiation patterns
│       ├── focusing.rs     # Focusing physics
│       └── apodization.rs  # Apodization effects
├── optics/                 # Optical physics
│   ├── scattering/
│   ├── absorption/
│   └── sonoluminescence/   # Light from cavitation
├── thermal/                # Thermal physics
│   ├── diffusion/
│   └── bioheat/
└── chemistry/              # Chemical physics
    └── kinetics/

Dependencies: core/, math/, domain/
Exports TO: solver/, analysis/, clinical/
Purpose: Physical models, governing equations, constitutive relations
```

**Rule**: Physics provides MODELS and EQUATIONS, not numerical solutions

**Key Structure**:
- Uses `domain/` for data structures (Grid, Medium)
- Uses `math/` for operators (gradient, FFT)
- Provides models to `solver/` for discretization
- Cavitation PINN uses abstract `math/ml/pinn/` framework with physics-specific extensions

---

### Layer 4: Numerical Solvers (Solver)

```
src/solver/
├── forward/                # Forward problem solvers
│   ├── fdtd/               # FDTD solver
│   │   ├── discretization.rs  # Spatial discretization
│   │   ├── time_step.rs       # Time stepping
│   │   └── plugin.rs          # Plugin interface
│   ├── pstd/               # PSTD solver
│   │   ├── spectral.rs        # Spectral methods
│   │   ├── dg/                # Discontinuous Galerkin
│   │   │   ├── basis.rs
│   │   │   ├── flux.rs
│   │   │   └── shock_capturing/
│   │   └── plugin.rs
│   ├── hybrid/             # Hybrid methods
│   │   ├── fdtd_pstd.rs
│   │   └── adaptive/
│   └── elastic/            # Elastic wave solvers
├── inverse/                # Inverse problems
│   ├── reconstruction/     # Reconstruction algorithms
│   │   ├── photoacoustic/  # Photoacoustic reconstruction
│   │   │   ├── time_reversal.rs
│   │   │   ├── back_projection.rs
│   │   │   └── iterative/
│   │   └── seismic/        # Seismic inversion
│   │       ├── fwi/        # Full waveform inversion
│   │       └── rtm/        # Reverse time migration
│   └── time_reversal/      # Time reversal methods
├── integration/            # Time integration (SHARED by all solvers)
│   └── schemes/            # Integration schemes
│       ├── runge_kutta.rs  # RK methods
│       ├── imex.rs         # IMEX schemes
│       └── multi_rate.rs   # Multi-rate methods
├── coupling/               # Multi-physics coupling
│   ├── acoustic_thermal.rs
│   └── acoustic_optical.rs
└── plugin/                 # Plugin architecture
    ├── manager.rs
    └── execution.rs

Dependencies: core/, math/, domain/, physics/
Exports TO: analysis/, clinical/
Purpose: Discretization, time integration, numerical solution
```

**Rule**: Solver discretizes Physics models using Math operators on Domain structures

**Shared Components**:
- Time integration schemes in `solver/integration/` (used by ALL solvers)
- Plugin architecture in `solver/plugin/` (used by ALL solvers)
- Each solver in `solver/forward/` uses these shared components

---

### Layer 5: Analysis & Post-Processing (Analysis)

```
src/analysis/
├── signal_processing/      # Signal analysis
│   ├── beamforming/        # ✅ CANONICAL LOCATION (moved FROM domain)
│   │   ├── time_domain/    # Time-domain beamforming
│   │   │   ├── das/        # Delay-and-sum
│   │   │   ├── dmas/       # Delay-multiply-and-sum
│   │   │   └── shared/     # SHARED delay calculations
│   │   │       ├── delays.rs      # Geometric delay calculation
│   │   │       └── apodization.rs # Apodization functions
│   │   ├── frequency_domain/ # Frequency-domain beamforming
│   │   │   ├── mvdr/       # MVDR (Capon)
│   │   │   ├── music/      # MUSIC algorithm
│   │   │   └── shared/     # SHARED covariance estimation
│   │   │       ├── covariance.rs  # Covariance matrices
│   │   │       └── eigenvalue.rs  # Eigendecomposition
│   │   ├── neural/         # Neural beamforming
│   │   │   ├── network.rs  # Network architecture
│   │   │   └── pinn/       # PINN-based beamforming
│   │   └── core/           # SHARED beamforming utilities
│   │       ├── geometry.rs        # Array geometry
│   │       └── focusing.rs        # Focusing calculations
│   ├── localization/       # Source localization
│   └── pam/                # Passive acoustic mapping
├── validation/             # Validation infrastructure
│   ├── analytical/         # Analytical benchmarks
│   ├── numerical/          # Numerical accuracy
│   └── clinical/           # Clinical validation
├── visualization/          # Visualization
│   ├── renderer/
│   └── data_pipeline/
└── performance/            # Performance analysis
    ├── profiling/
    └── optimization/

Dependencies: core/, math/, domain/, physics/, solver/
Exports TO: clinical/
Purpose: Post-processing, analysis, validation, visualization
```

**Rule**: Analysis processes OUTPUT from solvers, never runs solvers itself

**Key Migration**: 
- `domain/sensor/beamforming/` → `analysis/signal_processing/beamforming/`
- Reason: Beamforming is PROCESSING (analysis), not INFRASTRUCTURE (domain)

**Shared Components in Beamforming**:
- `shared/delays.rs` - Used by DAS, DMAS, etc.
- `shared/covariance.rs` - Used by MVDR, MUSIC, etc.
- `core/geometry.rs` - Used by ALL beamforming methods

---

### Layer 6: Clinical Applications (Clinical)

```
src/clinical/
├── imaging/                # Clinical imaging
│   ├── workflows/          # Complete imaging workflows
│   │   ├── bmode.rs        # B-mode imaging
│   │   ├── doppler.rs      # Doppler imaging
│   │   └── elastography/   # Elastography
│   └── protocols/          # Clinical protocols
├── therapy/                # Clinical therapy
│   ├── hifu/               # HIFU therapy
│   │   ├── planning.rs     # Treatment planning
│   │   └── monitoring.rs   # Real-time monitoring
│   ├── lithotripsy/        # Lithotripsy
│   │   ├── protocol.rs     # Treatment protocol
│   │   ├── bioeffects.rs   # Safety assessment
│   │   ├── stone_fracture.rs
│   │   └── monitoring.rs
│   └── cavitation/         # Cavitation-enhanced therapy
└── safety/                 # Clinical safety
    ├── thermal_index.rs
    └── mechanical_index.rs

Dependencies: ALL layers
Exports TO: User applications, APIs
Purpose: Complete clinical workflows, user-facing functionality
```

**Rule**: Clinical orchestrates ALL lower layers into complete workflows

---

## Critical Fixes Required

### Fix 1: Move Shared Components DOWN

#### 1.1 Constants Migration

```rust
// BEFORE (WRONG):
physics/constants/fundamental.rs:
    pub const GAS_CONSTANT: f64 = 8.314;

core/constants/thermodynamic.rs:
    pub use crate::physics::constants::GAS_CONSTANT;  // ❌ Upward dependency!

// AFTER (CORRECT):
core/constants/fundamental.rs:
    pub const GAS_CONSTANT: f64 = 8.314;  // ✓ Defined at lowest level

physics/constants/mod.rs:
    pub use crate::core::constants::GAS_CONSTANT;  // ✓ Downward re-export
```

**Rationale**: Constants are MOST shared → belong at LOWEST layer (core)

#### 1.2 Sparse Matrix Migration

```bash
# BEFORE (WRONG):
core/utils/sparse_matrix/  # Core shouldn't have linear algebra
    csr.rs
    solver.rs
    eigenvalue.rs

# AFTER (CORRECT):
math/linear_algebra/sparse/  # Math layer for mathematical operations
    csr.rs
    solver.rs
    eigenvalue.rs
```

**Rationale**: Sparse matrices are mathematical operations → belong in math layer

#### 1.3 Differential Operators Migration

```bash
# BEFORE (WRONG):
domain/grid/operators/  # Grid is infrastructure, not operations
    gradient.rs
    laplacian.rs

# AFTER (CORRECT):
math/numerics/operators/  # Generic numerical operators
    gradient.rs
    laplacian.rs
    
# Grid provides the STRUCTURE
domain/grid/structure.rs:
    pub fn delta_x(&self) -> f64 { self.dx }
    
# Math provides the OPERATORS
math/numerics/operators/gradient.rs:
    pub fn gradient_x(grid: &Grid, field: &Array3<f64>) -> Array3<f64> {
        // Uses grid.delta_x() but logic is in math layer
    }
```

**Rationale**: Operators are generic algorithms → belong in math, not domain

---

### Fix 2: Remove Upward Dependencies

#### 2.1 Core → Math/Domain Dependencies

```rust
// BEFORE (WRONG):
core/utils/mod.rs:
    pub use crate::math::fft::*;  // ❌ Core importing from math!

core/utils/test_helpers.rs:
    use crate::domain::grid::Grid;  // ❌ Core importing from domain!

// AFTER (CORRECT):
// Remove these re-exports entirely
// Users import directly:
use crate::math::fft::FFTProcessor;
use crate::domain::grid::Grid;

// Move test helpers:
tests/support/fixtures.rs:
    use crate::domain::grid::Grid;  // ✓ Tests can depend on anything
```

#### 2.2 Math → Physics Dependencies

```rust
// BEFORE (WRONG):
math/ml/pinn/cavitation_coupled.rs:
    use crate::physics::bubble_dynamics::*;  // ❌ Math importing physics!

// AFTER (CORRECT):
// Generic framework in math:
math/ml/pinn/physics_traits.rs:
    pub trait PhysicsModel {
        fn residual(&self, ...);
    }

// Physics-specific implementation in physics:
physics/acoustics/mechanics/cavitation/pinn/model.rs:
    use crate::math::ml::pinn::PhysicsModel;  // ✓ Physics imports math
    
    impl PhysicsModel for CavitationPINN {
        // Cavitation-specific implementation
    }
```

---

### Fix 3: Extract Shared Code to Lower Layers

#### 3.1 Beamforming Shared Utilities

```rust
// PROBLEM: Delay calculation duplicated in DAS, DMAS, MVDR

// BEFORE (DUPLICATED):
analysis/signal_processing/beamforming/time_domain/das.rs:
    fn calculate_delays(...) { /* logic */ }

analysis/signal_processing/beamforming/time_domain/dmas.rs:
    fn calculate_delays(...) { /* same logic */ }

analysis/signal_processing/beamforming/frequency_domain/mvdr.rs:
    fn calculate_delays(...) { /* same logic */ }

// AFTER (SHARED):
analysis/signal_processing/beamforming/shared/delays.rs:
    /// Shared delay calculation for ALL beamforming methods
    pub fn calculate_geometric_delays(
        sensor_positions: &Array2<f64>,
        focal_point: &[f64; 3],
        sound_speed: f64
    ) -> Array1<f64> {
        // Single implementation used by all
    }

// All beamformers use shared implementation:
analysis/signal_processing/beamforming/time_domain/das.rs:
    use super::super::shared::delays::calculate_geometric_delays;
```

**Rationale**: Shared within beamforming → Extract to `beamforming/shared/`

#### 3.2 Time Integration Shared Schemes

```rust
// PROBLEM: RK4 implemented separately in FDTD and PSTD

// BEFORE (DUPLICATED):
solver/forward/fdtd/time_step.rs:
    fn rk4_step(...) { /* logic */ }

solver/forward/pstd/time_step.rs:
    fn rk4_step(...) { /* same logic */ }

// AFTER (SHARED):
solver/integration/schemes/runge_kutta.rs:
    /// Generic RK4 for ANY solver
    pub fn rk4_step<State>(
        state: &State,
        derivative: impl Fn(&State) -> State,
        dt: f64
    ) -> State {
        // Single implementation
    }

// All solvers use shared implementation:
solver/forward/fdtd/plugin.rs:
    use crate::solver::integration::schemes::runge_kutta::rk4_step;
    
solver/forward/pstd/plugin.rs:
    use crate::solver::integration::schemes::runge_kutta::rk4_step;
```

**Rationale**: Shared across solvers → Extract to `solver/integration/schemes/`

---

## Deep Hierarchy Benefits

### Why Deep is GOOD (When Done Right)

1. **Clear Dependency Visualization**
```
physics/acoustics/mechanics/cavitation/rayleigh_plesset.rs
                  ^        ^         ^             ^
                  |        |         |             |
              domain  phenomena   specific      implementation
              
Path SHOWS: This is acoustic physics → mechanical phenomenon → 
            cavitation-specific → Rayleigh-Plesset model
```

2. **Shared Code Extraction**
```
analysis/signal_processing/beamforming/
    ├── shared/           ← Shared by ALL beamforming
    │   ├── delays.rs
    │   └── covariance.rs
    ├── time_domain/      ← Uses shared/
    │   ├── das.rs
    │   └── dmas.rs
    └── frequency_domain/ ← Uses shared/
        ├── mvdr.rs
        └── music.rs
```

3. **Specificity Gradient**
```
Layer 0 (core):      Most generic  → Used by everyone
Layer 1 (math):      Generic math  → Used by physics/solver/analysis
Layer 2 (domain):    Domain infra  → Used by physics/solver/analysis
Layer 3 (physics):   Physics models → Used by solver/analysis
Layer 4 (solver):    Numerical     → Used by analysis
Layer 5 (analysis):  Post-process  → Used by clinical
Layer 6 (clinical):  Most specific → Uses everything
```

---

## Implementation Roadmap

### Phase 1: Fix Upward Dependencies (CRITICAL)

**Duration**: 2-3 days

1. **Move constants DOWN**
   - [ ] Move all constants to `core/constants/`
   - [ ] Remove `physics/constants/` (or make it re-export)
   - [ ] Update all imports

2. **Remove core → math/domain dependencies**
   - [ ] Remove FFT re-exports from `core/utils/`
   - [ ] Move test helpers to `tests/support/`
   - [ ] Update all imports

3. **Fix math → physics coupling**
   - [ ] Create `math/ml/pinn/physics_traits.rs`
   - [ ] Move cavitation PINN to `physics/.../cavitation/pinn/`
   - [ ] Update imports

**Success**: Zero upward dependencies, clean layer separation

---

### Phase 2: Extract Shared Components (HIGH PRIORITY)

**Duration**: 5-7 days

1. **Extract beamforming shared code**
   - [ ] Create `analysis/signal_processing/beamforming/shared/`
   - [ ] Move delay calculations to `shared/delays.rs`
   - [ ] Move covariance estimation to `shared/covariance.rs`
   - [ ] Update all beamformers to use shared code

2. **Extract solver shared code**
   - [ ] Create `solver/integration/schemes/`
   - [ ] Move RK4, IMEX to shared schemes
   - [ ] Update FDTD, PSTD to use shared schemes

3. **Move sparse matrices**
   - [ ] Move `core/utils/sparse_matrix/` → `math/linear_algebra/sparse/`
   - [ ] Update imports

4. **Move differential operators**
   - [ ] Move `domain/grid/operators/` → `math/numerics/operators/`
   - [ ] Update imports

**Success**: Zero code duplication, clear shared components

---

### Phase 3: Deepen Where Needed (MEDIUM PRIORITY)

**Duration**: 5-7 days

1. **Proper physics hierarchy**
   - [ ] Create `physics/acoustics/models/{linear,nonlinear,elastic}/`
   - [ ] Organize by specificity: general → specific
   - [ ] Clear dependency flow within physics

2. **Proper beamforming hierarchy**
   - [ ] Organize: `shared/` (most generic) → `time_domain/` → specific algorithms
   - [ ] Extract common patterns

3. **Proper solver hierarchy**
   - [ ] Clear separation: shared schemes → solver implementations → plugins

**Success**: Hierarchy reflects dependency relationships

---

### Phase 4: Complete Deprecation Removal (MEDIUM PRIORITY)

**Duration**: 3-5 days

1. **Remove deprecated beamforming**
   - [ ] Remove `domain/sensor/beamforming/` entirely
   - [ ] All code migrated to `analysis/signal_processing/beamforming/`

2. **Remove deprecated locations**
   - [ ] Remove other deprecated modules
   - [ ] Clean up re-exports

**Success**: Zero deprecated markers

---

## Validation Checklist

### Architectural Correctness

- [ ] **No upward dependencies**: Each layer only imports from lower layers
- [ ] **Shared components at bottom**: Generic code in lowest appropriate layer
- [ ] **Zero code duplication**: Shared code extracted to common modules
- [ ] **Clear responsibility**: Each layer has single, clear purpose
- [ ] **Dependency visibility**: File structure reflects dependency graph

### Hierarchy Correctness

- [ ] **Specificity gradient**: Generic (bottom) → Specific (top)
- [ ] **Shared first**: Shared modules at lowest level of scope
- [ ] **Clear paths**: File paths reflect conceptual hierarchy
- [ ] **Bounded contexts**: Each module has clear, isolated responsibility

### Code Quality

- [ ] **Zero circular dependencies**
- [ ] **Zero redundant implementations**
- [ ] **Clear module documentation**
- [ ] **Consistent import patterns**

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Upward Dependencies** | 12+ | 0 | 🔴 |
| **Code Duplication** | 15+ instances | 0 | 🔴 |
| **Shared Component Layers** | Wrong (scattered) | Correct (bottom) | 🔴 |
| **Constants Location** | Physics | Core | 🔴 |
| **Layer Violations** | Multiple | Zero | 🔴 |

---

## Conclusion

**The problem is NOT that the hierarchy is too deep.**

**The problem is that:**
1. Dependencies flow in WRONG direction (upward instead of downward)
2. Shared components are in WRONG layers (scattered instead of at bottom)
3. Code is DUPLICATED instead of shared
4. Hierarchy doesn't REFLECT the actual architectural relationships

**The solution is:**
1. ✅ **Fix dependency direction**: Strictly downward only
2. ✅ **Move shared code DOWN**: Generic code at lowest appropriate layer
3. ✅ **Extract duplicates**: Share instead of duplicate
4. ✅ **Deepen strategically**: Add layers where they clarify structure

**Deep vertical hierarchy is CORRECT when:**
- Each layer has clear responsibility
- Dependencies flow one direction (down)
- Shared components sink to appropriate abstraction level
- File paths reflect conceptual organization
- More generic = lower in tree, more specific = higher in tree

---

**End of Corrected Audit**

*This audit CORRECTS the previous misunderstanding and provides the proper approach to deep vertical hierarchical architecture.*