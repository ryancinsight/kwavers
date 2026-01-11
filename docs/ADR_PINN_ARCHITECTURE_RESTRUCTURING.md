# ADR: PINN Architectural Restructuring and Domain Layer Separation

**Status:** Accepted  
**Date:** 2025-01-29  
**Sprint:** 186 Phase 3 - Deep Vertical Architecture  
**Deciders:** System Architect, Lead Developer  
**Related:** Sprint 186 Comprehensive Audit, GRASP Compliance Initiative

---

## Context and Problem Statement

During Sprint 186's comprehensive audit, a critical architectural flaw was discovered:

**Problem:** Physics-Informed Neural Networks (PINNs) were implemented in `analysis/ml/pinn/`, causing three violations:

1. **Redundancy:** PINNs solve wave equations (same PDEs as forward solvers), duplicating physics implementations
2. **Framework mismatch:** Burn's autodiff overhead was spreading into non-ML code paths via shared abstractions
3. **Conceptual misplacement:** PINNs are **inverse solvers**, not post-processing analysis tools

**Core Question:**
> "How do we structure PINNs, forward solvers, and ML analysis tools to eliminate redundancy while supporting both ndarray (CPU, no autodiff) and Burn (GPU, autodiff) backends?"

---

## Decision Drivers

### Mathematical Correctness
- PINNs and forward solvers both solve **identical wave equations** (acoustic, elastic, electromagnetic)
- Physics specifications (PDEs, boundary conditions, material properties) must be single source of truth
- Validation logic must be reusable across solver types

### Performance Requirements
- Forward solvers need zero autodiff overhead (pure ndarray, optimized numerical kernels)
- PINNs require autodiff for computing PDE residuals through neural networks
- GPU acceleration should be available for PINNs without forcing all code onto GPU

### Architectural Purity (GRASP/SOLID)
- **Single Responsibility:** Each module has one reason to change
- **Separation of Concerns:** Domain specifications ≠ Solver implementations ≠ Post-processing
- **Dependency Inversion:** Both forward and inverse solvers depend on domain abstractions, not each other
- **Open/Closed:** New solver types (e.g., spectral methods, hybrid solvers) shouldn't require changing existing code

### Zero Technical Debt
- No placeholder implementations, dummy data, or "simplified" code paths
- No error masking or silent failures
- All abstractions must be mathematically justified and fully implemented

---

## Considered Options

### Option 1: Keep PINNs in `analysis/ml/` (Status Quo - Rejected)

**Structure:**
```
analysis/ml/pinn/
    wave_equation_2d.rs  # 2,578 lines - reimplements wave equation
solver/forward/
    acoustic/            # Separate wave equation implementation
```

**Pros:**
- No immediate refactoring cost
- PINNs grouped with other ML tools

**Cons:**
- ❌ Violates Single Responsibility (analysis tools should not solve PDEs)
- ❌ Physics specifications duplicated between forward solvers and PINNs
- ❌ No shared validation or material property abstractions
- ❌ Burn dependencies leak into analysis module (incorrect layer)

**Verdict:** Rejected. Violates architectural principles and creates maintainability debt.

---

### Option 2: Rewrite Everything in Burn with NdArray Backend (Rejected)

**Structure:**
```
solver/
    forward/  # All use Burn's NdArray backend
    inverse/pinn/  # Use Burn with autodiff
```

**Pros:**
- Single framework (Burn) for all tensor operations
- Potential for future GPU migration of forward solvers
- Unified API for tensor operations

**Cons:**
- ❌ Massive rewrite of validated forward solvers (high risk)
- ❌ Autodiff overhead even when disabled (measurable performance cost)
- ❌ Forces mental model of "everything is a tensor graph" when unnecessary
- ❌ Loses zero-copy interop with existing ndarray ecosystem (rayon, ndarray-linalg, etc.)

**Verdict:** Rejected. Too disruptive, performance costs, no clear benefit for forward solvers.

---

### Option 3: Domain Layer Separation with Hybrid Backend (Accepted ✓)

**Structure:**
```
domain/                         # Shared abstractions (no numerics)
    geometry/                   # Spatial domains (RectangularDomain, SphericalDomain)
    physics/                    # Wave equation TRAIT specifications
        wave_equation.rs        # trait WaveEquation, AcousticWaveEquation, ElasticWaveEquation
    tensor/                     # Unified tensor abstraction
        mod.rs                  # TensorView, TensorMut, NdArrayTensor
    medium/                     # Material property traits (existing)

solver/forward/                 # Numerical solvers (ndarray-based)
    acoustic/
        fdtd.rs                 # Implements domain::physics::AcousticWaveEquation
    elastic/
        swe/                    # Implements domain::physics::ElasticWaveEquation

solver/inverse/                 # Inverse problem solvers
    pinn/                       # Physics-Informed Neural Networks (Burn-based)
        elastic_2d/
            geometry.rs         # Collocation sampling, interface conditions
            model.rs            # Neural network architecture (future)
            loss.rs             # Physics-informed loss (future)
        acoustic_2d/            # Future
    optimization/               # Gradient-based parameter estimation
    time_reversal/              # Existing time-reversal methods

analysis/                       # Post-processing only
    ml/
        inference/              # Deploy pre-trained models
        uncertainty/            # Uncertainty quantification
        preprocessing/          # Feature extraction
```

**Pros:**
- ✅ **Zero redundancy:** Physics specifications defined once in `domain/physics` traits
- ✅ **Optimal performance:** Forward solvers use pure ndarray (no autodiff overhead)
- ✅ **Framework flexibility:** PINNs use Burn; forward solvers use ndarray; conversion layer available
- ✅ **Correct semantics:** PINNs are solvers, not analysis tools (placed in `solver/inverse/`)
- ✅ **Shared validation:** Both solver types implement same traits → same test suite
- ✅ **Type safety:** Compiler enforces physics constraints via trait bounds

**Cons:**
- 🔶 Requires restructuring existing PINN code (one-time cost, acceptable)
- 🔶 Conversion overhead at PINN-forward solver boundaries (mitigated by zero-copy when possible)

**Verdict:** **Accepted.** Best balance of correctness, performance, and maintainability.

---

## Decision

### Architectural Structure

#### Layer 1: Domain Specifications (Bottom Layer)

**Purpose:** Define WHAT physics equations and geometric domains are, independent of HOW they're solved.

```rust
// domain/physics/wave_equation.rs
pub trait WaveEquation: Send + Sync {
    fn domain(&self) -> &Domain;
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;
    fn apply_boundary_conditions(&mut self, field: &mut ArrayD<f64>);
    fn cfl_timestep(&self) -> f64;
}

pub trait ElasticWaveEquation: WaveEquation {
    fn lame_lambda(&self) -> ArrayD<f64>;
    fn lame_mu(&self) -> ArrayD<f64>;
    fn density(&self) -> ArrayD<f64>;
    fn stress_from_displacement(&self, u: &ArrayD<f64>) -> ArrayD<f64>;
}
```

**Mathematical Foundation:**
- All wave equations are second-order hyperbolic PDEs: ∂²u/∂t² = L[u] + f
- Traits specify L (spatial operator), boundary conditions, material properties
- Solver-agnostic: applies to FD, FEM, spectral, PINN, analytical solutions

**Design Principles:**
- No implementation details (no grids, no discretization, no autodiff)
- Pure trait interfaces with mathematical semantics
- Composable: traits can be combined via bounds (e.g., `T: ElasticWaveEquation + Serializable`)

---

#### Layer 2: Solver Implementations (Middle Layer)

**Forward Solvers:** Numerical discretization (FD, FEM, spectral)
```rust
// solver/forward/elastic/swe/mod.rs
pub struct StaggeredElasticSolver {
    grid: Grid,
    materials: ElasticMedium,
    // ... ndarray-based state
}

impl ElasticWaveEquation for StaggeredElasticSolver {
    fn spatial_operator(&self, u: &ArrayD<f64>) -> ArrayD<f64> {
        // Finite difference stencil application
    }
}
```

**Inverse Solvers (PINNs):** Neural network approximation
```rust
// solver/inverse/pinn/elastic_2d/mod.rs
pub struct ElasticPINN2D<B: Backend> {
    network: NeuralNet<B>,
    domain: Domain,
    materials: ElasticProperties,
}

impl<B: Backend> ElasticWaveEquation for ElasticPINN2D<B> {
    fn spatial_operator(&self, u: &ArrayD<f64>) -> ArrayD<f64> {
        // Autodiff through neural network
    }
}
```

**Key Insight:** Both implement **same trait**, enabling:
- Shared validation (test against analytical solutions)
- Drop-in replacement (forward solver ↔ PINN)
- Hybrid solvers (PINN in complex regions, FD in simple regions)

---

#### Layer 3: Tensor Abstraction (Cross-Cutting)

**Problem:** Forward solvers use ndarray; PINNs use Burn tensors. Need zero-copy conversion.

**Solution:** Unified tensor interface with backend-specific implementations

```rust
// domain/tensor/mod.rs
pub trait TensorView: Send + Sync {
    fn to_ndarray_f64(&self) -> ArrayD<f64>;
    fn shape(&self) -> Shape;
}

pub struct NdArrayTensor { data: ArrayD<f64> }  // Default: zero overhead

#[cfg(feature = "burn-ndarray")]
pub struct BurnTensor<B: Backend> { data: Tensor<B> }  // Optional: autodiff
```

**Zero-Copy Strategy:**
- When Burn uses NdArray backend: share memory directly (no allocation)
- When Burn uses GPU backend: explicit copy with Device-to-Host transfer
- Feature gates: `default = ["ndarray"]`, optional `["burn-ndarray", "burn-wgpu"]`

---

### Migration Strategy

#### Phase 1: Foundation (Completed in Sprint 186)
- ✅ Created `domain/physics/` with wave equation traits
- ✅ Created `domain/geometry/` with shared geometric primitives
- ✅ Created `domain/tensor/` with unified tensor abstraction
- ✅ Created `solver/inverse/pinn/` directory structure
- ✅ Moved PINN geometry from `analysis/ml/pinn/wave_equation_2d/geometry.rs` to `solver/inverse/pinn/elastic_2d/geometry.rs`
- ✅ Extended with PINN-specific collocation sampling (stratified, LHS, adaptive refinement)

#### Phase 2: PINN Extraction (Next Sprint)
1. Extract remaining `burn_wave_equation_2d.rs` components to `solver/inverse/pinn/elastic_2d/`:
   - `config.rs` — hyperparameters, training configuration
   - `model.rs` — neural network architecture
   - `loss.rs` — physics-informed loss functions (PDE residual, BC, data fitting)
   - `training.rs` — training loop, optimizer, learning rate scheduling
   - `inference.rs` — trained model deployment

2. Implement `ElasticWaveEquation` trait for `ElasticPINN2D`

3. Add integration tests comparing PINN vs forward solver on benchmark problems

#### Phase 3: Forward Solver Refactor (Future Sprint)
1. Refactor existing forward solvers to implement `domain::physics` traits
2. Keep internal ndarray implementation unchanged
3. Expose trait interfaces for interoperability

#### Phase 4: Validation & Benchmarking
1. Shared test suite for all `ElasticWaveEquation` implementations
2. Performance benchmarks (forward solvers should remain optimal)
3. Convergence studies (PINN vs analytical vs forward solvers)

---

## Consequences

### Positive

✅ **Mathematical Correctness**
- Single source of truth for wave equation specifications
- Compiler-enforced physics constraints via traits
- Shared validation across all solver types

✅ **Performance**
- Forward solvers: zero autodiff overhead (pure ndarray)
- PINNs: full autodiff + GPU acceleration (Burn)
- Zero-copy tensor conversion when using compatible backends

✅ **Maintainability**
- Clear separation: domain specs vs solver implementations vs post-processing
- GRASP compliant: each module has single responsibility
- No circular dependencies or upward layer violations

✅ **Extensibility**
- New solver types (spectral, hybrid, analytical) implement same traits
- New physics (electromagnetic, coupled multi-physics) add new traits
- Framework changes (e.g., new Burn version) isolated to solver layer

✅ **Reusability**
- Material property definitions shared (domain/medium/)
- Geometric domains shared (domain/geometry/)
- Boundary conditions shared (domain/physics/)

### Negative (Mitigated)

🔶 **Migration Cost**
- One-time restructuring effort for existing PINN code
- **Mitigation:** Incremental migration; existing code continues to work during transition

🔶 **Conversion Overhead**
- Potential cost when converting ndarray ↔ Burn tensors
- **Mitigation:** Use Burn's NdArray backend for zero-copy when possible; explicit conversion only at solver boundaries

🔶 **Learning Curve**
- Contributors must understand trait-based architecture
- **Mitigation:** Comprehensive documentation (this ADR + module-level docs); examples for both forward and inverse solvers

### Risks and Safeguards

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PINN performance regression | Low | Medium | Benchmark before/after; Burn uses optimized backends (WGPU, CUDA) |
| Forward solver disruption | Very Low | High | Forward solvers unchanged internally; only add trait implementations |
| Incomplete abstraction | Medium | Medium | Iterative refinement; start with acoustic/elastic, generalize later |
| Burn framework instability | Low | Medium | Feature-gate Burn dependencies; fallback to pure ndarray for critical paths |

---

## Validation Criteria

### Structural Validation (Completed ✓)
- [x] `domain/physics/` exists with wave equation traits
- [x] `domain/geometry/` exists with shared geometric primitives
- [x] `domain/tensor/` exists with unified tensor abstraction
- [x] `solver/inverse/pinn/` exists with elastic_2d submodule
- [x] PINN geometry moved from `analysis/ml/` to `solver/inverse/`
- [x] Build succeeds with zero errors (warnings acceptable)

### Functional Validation (In Progress)
- [ ] Forward solver implements `ElasticWaveEquation` trait
- [ ] PINN implements `ElasticWaveEquation` trait
- [ ] Both pass shared validation test suite
- [ ] Tensor conversion works (ndarray → Burn → ndarray roundtrip)

### Performance Validation (Future)
- [ ] Forward solver performance unchanged (≤ 1% regression)
- [ ] PINN training time acceptable (< 10 min for benchmark problem)
- [ ] Zero-copy tensor conversion when using Burn NdArray backend

### Maintainability Validation (Ongoing)
- [ ] No circular dependencies (validated by cargo build)
- [ ] File size compliance (all files < 500 lines)
- [ ] Module documentation complete (rustdoc build --no-deps)

---

## Implementation Status

### Completed (Sprint 186, Session 4)

**Domain Layer:**
- `domain/physics/wave_equation.rs` (333 lines) — Core trait specifications
  - `WaveEquation` trait (abstract interface)
  - `AcousticWaveEquation` trait (scalar pressure field)
  - `ElasticWaveEquation` trait (vector displacement field)
  - `Domain`, `BoundaryCondition`, `SourceTerm` types
  - Unit tests for domain construction and boundary classification

- `domain/geometry/mod.rs` (594 lines) — Geometric domain abstractions
  - `GeometricDomain` trait (point-in-domain, normal computation, sampling)
  - `RectangularDomain` (1D/2D/3D Cartesian domains)
  - `SphericalDomain` (2D circular, 3D spherical domains)
  - Boundary normal computation, interior/boundary sampling
  - Unit tests for containment, classification, sampling

- `domain/tensor/mod.rs` (374 lines) — Unified tensor abstraction
  - `TensorView`, `TensorMut` traits (backend-agnostic interface)
  - `NdArrayTensor` (default CPU implementation)
  - `Shape`, `DType`, `Backend` enums
  - Conversion utilities (ndarray ↔ tensor)
  - Unit tests for tensor creation, mutation, conversion

- `domain/physics/mod.rs` (146 lines) — Physics module root with documentation

**Solver Layer:**
- `solver/inverse/pinn/mod.rs` (177 lines) — PINN framework documentation
  - Architectural rationale (why PINNs are solvers, not analysis tools)
  - Mathematical foundation (physics-informed loss formulation)
  - Framework integration strategy (Burn backends, tensor interop)
  - Module organization roadmap

- `solver/inverse/pinn/elastic_2d/mod.rs` (212 lines) — Elastic 2D PINN module
  - Mathematical formulation (2D elastic wave PDE)
  - PINN loss function specification (PDE + BC + IC + data terms)
  - Usage examples (forward problem, inverse problem)
  - Submodule structure (geometry, config, model, loss, training, inference)

- `solver/inverse/pinn/elastic_2d/geometry.rs` (509 lines) — PINN-specific geometry
  - `InterfaceCondition` enum (elastic continuity, sliding contact, acoustic-elastic)
  - `MultiRegionDomain` (heterogeneous media with interface conditions)
  - `CollocationSampler` (uniform, Latin hypercube, Sobol, adaptive)
  - `AdaptiveRefinement` (residual-based mesh refinement)
  - Unit tests for sampling strategies and multi-region domains

- `solver/inverse/mod.rs` (updated) — Added PINN exports

**Updated Modules:**
- `domain/mod.rs` — Added geometry, physics, tensor submodules with re-exports
- Module exports and documentation updated throughout

**Build Status:**
```
✓ cargo build --lib succeeds (0 errors, ~26 warnings - unused imports only)
✓ No circular dependencies
✓ All new modules pass rustdoc build
```

### In Progress (Next Sprint)

1. **Complete PINN elastic_2d submodules:**
   - `config.rs` — Training hyperparameters, collocation point configuration
   - `model.rs` — Neural network architecture (MLP, adaptive activations)
   - `loss.rs` — Physics-informed loss computation (PDE residual, BC enforcement)
   - `training.rs` — Training loop, optimizer, learning rate scheduler
   - `inference.rs` — Model deployment, quantization, runtime evaluation

2. **Trait implementation:**
   - Implement `ElasticWaveEquation` for `ElasticPINN2D<B: Backend>`
   - Add integration tests comparing PINN vs analytical solutions

3. **Forward solver refactor:**
   - Implement `ElasticWaveEquation` for existing `StaggeredElasticSolver`
   - Verify no performance regression

### Future Work

1. **Additional PINN solvers:**
   - `solver/inverse/pinn/acoustic_2d/` (2D acoustic PINN)
   - `solver/inverse/pinn/elastic_3d/` (3D elastic PINN)

2. **Hybrid solvers:**
   - Combine PINN (complex geometry) + FD (simple geometry) in same simulation

3. **Parameter estimation:**
   - Joint optimization of network weights + material properties
   - Uncertainty quantification for inverse problems

---

## References

### Architectural Principles
- SOLID Principles (Single Responsibility, Open/Closed, Dependency Inversion)
- GRASP Patterns (Low Coupling, High Cohesion, Information Expert)
- Clean Architecture (Domain layer at center, frameworks at periphery)

### Scientific References
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

- Karniadakis, G. E., et al. (2021). "Physics-informed machine learning." *Nature Reviews Physics*, 3(6), 422-440.

- Haghighat, E., et al. (2021). "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics." *Computer Methods in Applied Mechanics and Engineering*, 379, 113741.

### Implementation References
- Burn Deep Learning Framework: https://github.com/tracel-ai/burn
- ndarray Rust Crate: https://github.com/rust-ndarray/ndarray
- k-Wave MATLAB Toolbox: http://www.k-wave.org (reference for validation)

---

## Glossary

**PINN:** Physics-Informed Neural Network — neural network trained to satisfy PDE residuals via automatic differentiation

**Forward Problem:** Given material properties and sources, compute wave field (u_θ approximates solution)

**Inverse Problem:** Given observed wave field, estimate material properties (optimize both θ and material params)

**Collocation Points:** Spatial/temporal locations where PDE residuals are evaluated during training

**Autodiff:** Automatic differentiation — compute gradients via chain rule through computational graph

**Zero-Copy:** Tensor conversion that shares memory rather than allocating new buffer (requires compatible memory layouts)

**Backend:** Computational engine (NdArray CPU, WGPU GPU, CUDA GPU) for tensor operations

**Domain Layer:** Lowest architectural layer containing specifications and abstractions, independent of implementation

**Solver Layer:** Middle layer containing implementations of numerical methods (forward solvers, inverse solvers, analytical solvers)

---

**Approved by:** System Architect  
**Implemented by:** Lead Developer  
**Review Date:** 2025-01-29  
**Next Review:** After Phase 2 completion (PINN extraction finalized)