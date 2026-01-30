# Kwavers Architecture Guide

## Overview

Kwavers is a comprehensive ultrasound and optics simulation library built on a **deep vertical hierarchical architecture** with strict **separation of concerns** and **single source of truth (SSOT)** principles.

## Core Architectural Principles

### 1. Deep Vertical Hierarchy (9 Layers)

The codebase is organized into 9 distinct layers, each with specific responsibilities:

```
Layer 0: Core           - Error types, logging, time, fundamental constants
Layer 1: Math           - FFT, linear algebra, geometry, numerics, SIMD
Layer 2: Domain         - Business domain models (14 bounded contexts)
Layer 3: Physics        - Physical laws and constitutive relations (5 domains)
Layer 4: Solver         - Numerical methods (7 solver types)
Layer 5: Simulation     - Orchestration and configuration
Layer 6: Clinical       - Medical applications
Layer 7: Analysis       - Post-processing and signal processing
Layer 8: Infrastructure - I/O, API, cloud deployment
```

**Critical Rule**: Layers may only depend on layers below them. No circular dependencies permitted.

### 2. Single Source of Truth (SSOT)

Every concept, algorithm, or data structure must have exactly ONE authoritative implementation location:

- **Physical Constants**: `core/constants/fundamental.rs`
- **Solver Parameters**: `solver/constants.rs`
- **Wave Equations**: `physics/foundations/wave_equation.rs`
- **Covariance Estimation**: `analysis/signal_processing/beamforming/covariance/`
- **Beamforming Algorithms**: `analysis/signal_processing/beamforming/`

### 3. Separation of Concerns

Each module has a clearly defined responsibility with no overlap:

- **Domain Layer**: Defines WHAT (specifications, interfaces)
- **Physics Layer**: Defines HOW physics works (equations, laws)
- **Solver Layer**: Defines HOW to compute (numerical methods)
- **Analysis Layer**: Defines HOW to interpret results

---

## Layer-by-Layer Responsibilities

### Layer 0: Core (`core/`)

**Purpose**: Foundation primitives used by all other layers

**Modules**:
- `constants/` - Physical and mathematical constants
- `error/` - Error types and result handling
- `log/` - Logging infrastructure
- `time/` - Time representations
- `utils/` - Generic utilities

**Dependencies**: None (foundation layer)

**Rules**:
- No business logic
- No physics
- No numerical algorithms
- Pure infrastructure only

---

### Layer 1: Math (`math/`)

**Purpose**: Pure mathematical operations independent of domain

**Modules**:
- `fft/` - Fast Fourier Transform implementations
- `geometry/` - Geometric operations (distances, angles)
- `linear_algebra/` - Matrix operations, eigendecomposition
- `numerics/` - Numerical methods (interpolation, integration)
- `simd/` - SIMD-accelerated operations

**Dependencies**: Core only

**Rules**:
- Domain-agnostic mathematics only
- No physics concepts
- No simulation-specific logic
- Reusable across domains

**Examples**:
- ✅ Matrix inversion
- ✅ FFT computation
- ✅ Euclidean distance
- ❌ Sound speed calculation (belongs in physics)
- ❌ Beamforming (belongs in analysis)

---

### Layer 2: Domain (`domain/`)

**Purpose**: Business domain models and specifications (14 bounded contexts)

**Bounded Contexts**:

1. **`boundary/`** - Boundary conditions (CPML, PML, absorbing)
2. **`field/`** - Field type system, indices, mapping
3. **`geometry/`** - Spatial domains and coordinate systems
4. **`grid/`** - Computational grids and operators
5. **`imaging/`** - Imaging modality specifications
6. **`medium/`** - Material properties (heterogeneous, homogeneous, anisotropic)
7. **`mesh/`** - Mesh structures (unstructured)
8. **`plugin/`** - Plugin system interfaces
9. **`sensor/`** - Transducer arrays, sensor configurations
10. **`signal/`** - Signal type definitions
11. **`signal_processing/`** - Domain signal processing interfaces
12. **`source/`** - Acoustic/EM source specifications
13. **`tensor/`** - Tensor abstractions
14. **`therapy/`** - Therapeutic application specifications

**Dependencies**: Core, Math

**Rules**:
- Define WHAT, not HOW
- Specifications and interfaces only
- No algorithm implementations
- No physics equations

**Examples**:
- ✅ Define `MediumProperties` struct
- ✅ Define `SensorArray` geometry
- ✅ Define `BoundaryCondition` enum
- ❌ Implement wave propagation (belongs in physics/solver)
- ❌ Implement beamforming algorithm (belongs in analysis)

---

### Layer 3: Physics (`physics/`)

**Purpose**: Physical laws, constitutive relations, and wave equations

**Domains**:

1. **`foundations/`** - Wave equation traits (SSOT for all wave physics)
2. **`acoustics/`** - Acoustic wave physics
   - `mechanics/` - Wave mechanics, nonlinear acoustics
   - `bubble_dynamics/` - Cavitation modeling
   - `imaging/` - Imaging physics
   - `skull/` - Transcranial propagation
3. **`chemistry/`** - Chemical reactions
4. **`electromagnetic/`** - EM wave physics, Maxwell's equations
5. **`optics/`** - Optical physics (subset of EM, 400-700nm)
6. **`thermal/`** - Heat transfer, thermodynamics
7. **`plugin/`** - Physics plugin architecture

**Dependencies**: Core, Math, Domain

**Rules**:
- Define constitutive relations (σ = f(ε), etc.)
- Define governing equations (∇²p = ∂²p/∂t²/c²)
- NO numerical discretization (that's solver layer)
- NO data processing (that's analysis layer)

**Examples**:
- ✅ Define nonlinear wave equation: ∇²p - (1/c²)∂²p/∂t² = nonlinear_term
- ✅ Define Westervelt equation
- ✅ Define attenuation model: α(f) = α₀f^β
- ❌ FDTD discretization (belongs in solver)
- ❌ Image reconstruction (belongs in analysis)

---

### Layer 4: Solver (`solver/`)

**Purpose**: Numerical methods for solving physics equations

**Solver Types**:

1. **`forward/`** - Forward problem solvers
   - `fdtd/` - Finite-Difference Time-Domain
   - `pstd/` - Pseudospectral Time-Domain
   - `hybrid/` - Hybrid methods
2. **`inverse/`** - Inverse problems (reconstruction, time-reversal)
3. **`analytical/`** - Analytical solutions
4. **`integration/`** - Time integration schemes
5. **`multiphysics/`** - Multi-physics coupling
6. **`validation/`** - Solver validation suites
7. **`utilities/`** - Adaptive mesh refinement

**Dependencies**: Core, Math, Domain, Physics

**Rules**:
- Implement HOW to solve physics equations
- Discretization, time-stepping, spatial operators
- NO physics definitions (get from physics layer)
- NO result interpretation (that's analysis layer)

**Examples**:
- ✅ FDTD stencil implementation
- ✅ Pseudospectral derivative operators
- ✅ Time integration (RK4, leapfrog)
- ❌ Define wave speed (get from physics layer)
- ❌ Compute SNR of output (belongs in analysis)

---

### Layer 5: Simulation (`simulation/`)

**Purpose**: Orchestrate solvers with domain configurations

**Modules**:
- `builder/` - Configuration builders
- `core/` - Main simulation loop
- `factory/` - Object factories
- `imaging/` - Imaging simulation orchestration
- `modalities/` - Simulation modalities
- `multi_physics/` - Multi-physics orchestration
- `therapy/` - Therapy simulation orchestration

**Dependencies**: Core, Math, Domain, Physics, Solver

**Rules**:
- Coordinate solver execution
- Manage simulation lifecycle
- Handle configuration
- NO direct sensor data processing (use analysis layer)

**Examples**:
- ✅ Build simulation from config
- ✅ Run solver for N timesteps
- ✅ Coordinate multi-physics coupling
- ❌ Implement beamforming (belongs in analysis)

---

### Layer 6: Clinical (`clinical/`)

**Purpose**: Medical and therapeutic applications

**Applications**:

1. **`imaging/`** - Diagnostic imaging workflows
   - `doppler/` - Doppler imaging
   - `functional_ultrasound/` - fUS workflows
   - `phantoms/` - Phantom definitions
   - `photoacoustic/` - PA imaging
   - `spectroscopy/` - Spectroscopic analysis
   - `workflows/` - Clinical workflows
2. **`safety/`** - Safety monitoring (IEC 60601-2-37 compliance)
3. **`therapy/`** - Therapeutic applications (HIFU, lithotripsy)

**Dependencies**: Core, Math, Domain, Physics, Solver, Simulation

**Rules**:
- Clinical decision support
- Workflow orchestration
- Safety compliance
- Should use Simulation layer facade, NOT direct solver access

**Current Issue**: Some files directly use solver layer (layer violation). Should go through simulation.

---

### Layer 7: Analysis (`analysis/`)

**Purpose**: Post-processing, signal processing, and result interpretation

**Modules**:

1. **`conservation/`** - Physics validation
2. **`imaging/`** - Image analysis and processing
3. **`ml/`** - Machine learning models
4. **`performance/`** - Performance profiling
5. **`plotting/`** - Visualization (with plotly feature)
6. **`signal_processing/`** - Advanced signal processing
   - `beamforming/` - Beamforming algorithms (SSOT)
   - `clutter_filter/` - Clutter removal
   - `filtering/` - Signal filtering
   - `localization/` - Source localization
   - `pam/` - Passive acoustic mapping
7. **`testing/`** - Test utilities
8. **`validation/`** - Validation suites
9. **`visualization/`** - Advanced visualization (optional GPU)

**Dependencies**: Core, Math, Domain, Physics, Solver (for types only)

**Rules**:
- Process simulation results
- Extract features, compute metrics
- Implement signal processing algorithms
- NO solver implementation
- NO physics equation definitions

**Examples**:
- ✅ Delay-and-sum beamforming
- ✅ MVDR/Capon spatial spectrum
- ✅ SNR calculation
- ✅ Image quality metrics
- ❌ Time-stepping (belongs in solver)

---

### Layer 8: Infrastructure (`infra/`)

**Purpose**: External interfaces and deployment

**Modules**:
- `api/` - REST API, routing
- `cloud/` - Cloud deployment (AWS, Azure, GCP)
- `io/` - File I/O, serialization
- `runtime/` - Runtime management

**Dependencies**: All layers

**Rules**:
- External interfaces only
- Deployment and scaling
- Serialization formats
- NO business logic

---

## Critical Architectural Patterns

### SSOT Enforcement

Every algorithm must have exactly one implementation:

**Bad** (duplication):
```rust
// In domain/sensor/beamforming.rs
fn compute_covariance(...) { ... }

// In analysis/signal_processing/beamforming.rs
fn compute_covariance(...) { ... }  // DUPLICATE!
```

**Good** (SSOT):
```rust
// In analysis/signal_processing/beamforming/covariance.rs (SSOT)
pub fn estimate_covariance(...) { ... }

// Domain layer uses the SSOT
use crate::analysis::signal_processing::beamforming::covariance;
```

### Layer Isolation

Higher layers depend on lower layers only:

**Bad** (circular dependency):
```rust
// In domain/sensor/mod.rs
use crate::analysis::beamforming::process; // ❌ domain depends on analysis
```

**Good** (proper layering):
```rust
// In analysis/beamforming/mod.rs
use crate::domain::sensor::SensorArray; // ✅ analysis depends on domain
```

### Facade Pattern for Layer Access

Clinical layer should access solvers through simulation facade:

**Bad** (layer violation):
```rust
// In clinical/therapy/mod.rs
use crate::solver::forward::FDTDSolver; // ❌ skips simulation layer
let solver = FDTDSolver::new(...);
```

**Good** (proper layering):
```rust
// In clinical/therapy/mod.rs
use crate::simulation::SimulationRunner; // ✅ use facade
let simulation = SimulationRunner::new(...);
```

---

## Module Responsibility Quick Reference

| Module | Defines | Implements | Uses |
|--------|---------|------------|------|
| Core | Infrastructure | Logging, errors | - |
| Math | Mathematics | FFT, linear algebra | Core |
| Domain | Business models | Specifications | Core, Math |
| Physics | Physical laws | Equations | Core, Math, Domain |
| Solver | Numerical methods | FDTD, PSTD | Core, Math, Domain, Physics |
| Simulation | Orchestration | Sim loops | Core, Math, Domain, Physics, Solver |
| Clinical | Medical workflows | Decision support | Simulation (not Solver!) |
| Analysis | Post-processing | Signal processing | Domain, Physics types |
| Infra | External I/O | API, cloud | All layers |

---

## Common Pitfalls and How to Avoid Them

### 1. Algorithm Duplication

**Problem**: Same algorithm implemented in multiple places

**Solution**: Identify the correct layer for the algorithm's responsibility:
- Domain algorithms → stay in domain
- Physics algorithms → move to physics
- Signal processing → move to analysis
- Numerical methods → move to solver

### 2. Layer Violations

**Problem**: Higher layers accessed from lower layers

**Solution**: 
- Refactor to use dependency injection
- Create interfaces in lower layer, implement in higher layer
- Use event/callback patterns

**Example - Materials Module Issue** (⚠️ FOUND 2026-01-29):
```rust
// ❌ WRONG: Material properties in physics layer
physics/materials/
├── MaterialProperties struct
├── tissue catalog
└── fluid catalog

// ✅ RIGHT: Material properties in domain layer
domain/medium/properties/
├── MaterialProperties struct
├── tissue catalog
└── fluid catalog
```

Material properties are **specifications** (WHAT materials have), not physics equations (HOW materials behave). See `MATERIALS_MODULE_REFACTORING_PLAN.md` for details.

### 3. Mixing Concerns

**Problem**: Physics equations in solver, business logic in analysis, etc.

**Solution**:
- Separate WHAT (domain) from HOW (solver/analysis)
- Separate physics EQUATIONS from numerical METHODS
- Keep data models separate from algorithms
- Keep property SPECIFICATIONS separate from physics EQUATIONS

### 4. Implicit Dependencies

**Problem**: Hidden coupling through global state or shared mutable data

**Solution**:
- Make all dependencies explicit in function signatures
- Use dependency injection for complex dependencies
- Avoid global mutable state

---

## Migration Guide

When refactoring code to match this architecture:

### Step 1: Identify Current Location
- Where does the code live now?
- What are its dependencies?

### Step 2: Identify Correct Location
- Is it a specification (domain)?
- Is it a physical law (physics)?
- Is it a numerical method (solver)?
- Is it post-processing (analysis)?

### Step 3: Check for Duplication
- Does this already exist elsewhere?
- If yes, consolidate to SSOT
- If no, create in correct layer

### Step 4: Update Imports
- Change all import paths
- Add backward-compat re-exports if needed
- Mark old location as deprecated

### Step 5: Verify Layer Dependencies
- Does the new location only depend on lower layers?
- Remove any upward dependencies

---

## Architecture Validation Checklist

Use this checklist to verify architectural compliance:

- [ ] No circular dependencies between layers
- [ ] Each algorithm has exactly one implementation (SSOT)
- [ ] Domain layer has no algorithm implementations
- [ ] Physics layer has no numerical discretization
- [ ] Solver layer has no physics equation definitions
- [ ] Analysis layer has no solver implementations
- [ ] Clinical layer uses simulation facade, not direct solver access
- [ ] All constants consolidated in core/constants or solver/constants
- [ ] No backward-compat re-exports older than 1 minor version
- [ ] All deprecation warnings include migration path

---

## References

- **Domain-Driven Design**: Eric Evans, "Domain-Driven Design: Tackling Complexity in the Heart of Software"
- **Clean Architecture**: Robert C. Martin, "Clean Architecture: A Craftsman's Guide to Software Structure and Design"
- **Hexagonal Architecture**: Alistair Cockburn, "Hexagonal Architecture"

## Maintenance

This architecture guide is a living document. Update it when:
- Adding new layers
- Adding new bounded contexts
- Making significant architectural changes
- Resolving architectural debt

Last Updated: 2026-01-29
