# Phase 4 Action Plan: Resolving Blocking Issues

**Status**: BLOCKED - Compilation errors prevent test execution  
**Priority**: P0 - Must resolve before proceeding  
**Estimated Resolution Time**: 8-12 hours

---

## Critical Path to Unblock

### Task 1: Resolve Sync Trait Violation [P0, 2-4 hours]

**Problem**: `ElasticPINN2DSolver<B>` cannot implement `WaveEquation` because Burn tensors are `!Sync`

**Root Cause**:
```rust
// WaveEquation trait requires Sync
pub trait WaveEquation: Send + Sync { ... }

// But Burn tensors internally use std::cell::OnceCell which is !Sync
pub struct ElasticPINN2DSolver<B> {
    model: ElasticPINN2D<B>,  // Contains Burn tensors
    ...
}
```

**Recommended Solution**: Separate trait hierarchy (Option C)

**Implementation**:

```rust
// src/domain/physics/wave_equation.rs

/// Traditional numerical solvers (FDTD, FEM, spectral)
pub trait WaveEquation: Send + Sync {
    fn domain(&self) -> &Domain;
    fn cfl_timestep(&self) -> f64;
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;
    // ... existing methods
}

/// Neural network and autodiff-based solvers (PINN)
/// Relaxed Sync constraint due to autodiff framework requirements
pub trait AutodiffWaveEquation: Send {
    fn domain(&self) -> &Domain;
    fn cfl_timestep(&self) -> f64;
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;
    // ... same methods as WaveEquation
}

/// Elastic wave equation for traditional solvers
pub trait ElasticWaveEquation: WaveEquation {
    fn lame_lambda(&self) -> ArrayD<f64>;
    fn lame_mu(&self) -> ArrayD<f64>;
    fn density(&self) -> ArrayD<f64>;
    // ... existing methods
}

/// Elastic wave equation for autodiff-based solvers
pub trait AutodiffElasticWaveEquation: AutodiffWaveEquation {
    fn lame_lambda(&self) -> ArrayD<f64>;
    fn lame_mu(&self) -> ArrayD<f64>;
    fn density(&self) -> ArrayD<f64>;
    // ... same methods as ElasticWaveEquation
}
```

**Changes Required**:
1. Add `AutodiffWaveEquation` and `AutodiffElasticWaveEquation` traits to `wave_equation.rs`
2. Change `ElasticPINN2DSolver` to implement `AutodiffElasticWaveEquation` instead of `ElasticWaveEquation`
3. Update validation framework to accept both trait types:
   ```rust
   // Overload for traditional solvers
   pub fn validate_material_properties<T: ElasticWaveEquation>(solver: &T) -> ValidationResult

   // Overload for autodiff solvers
   pub fn validate_material_properties_autodiff<T: AutodiffElasticWaveEquation>(solver: &T) -> ValidationResult
   ```
4. Update PINN tests to use autodiff-specific validation functions

**Acceptance Criteria**:
- ✅ Compilation succeeds for `ElasticPINN2DSolver`
- ✅ Both trait hierarchies compile
- ✅ Validation framework supports both solver types
- ✅ No `Sync` violations

---

### Task 2: Fix Burn Optimizer API [P0, 2-3 hours]

**Problem**: Burn 0.19 changed optimizer API - `Adam` no longer generic, `GradientsParams` removed

**Errors**:
```
error[E0107]: struct takes 0 generic arguments but 1 generic argument was supplied
   --> src\solver\inverse\pinn\elastic_2d\training.rs:224:10
    |
224 |     Adam(Adam<B>),
    |          ^^^^--- help: remove the unnecessary generics

error[E0432]: unresolved import
   |
89  | use burn::optim::{..., GradientsParams, ...};
    |                        ^^^^^^^^^^^^^^^^ not found in `optim`
```

**Solution**: Update to Burn 0.19 API

**Implementation**:

```rust
// src/solver/inverse/pinn/elastic_2d/training.rs

use burn::{
    module::AutodiffModule,
    optim::{Adam, AdamConfig, Optimizer, Sgd, SgdConfig},  // Remove GradientsParams
    tensor::backend::AutodiffBackend,
};

// Update enum (no generic parameters)
enum OptimizerWrapper {
    Adam(Adam),
    Sgd(Sgd),
}

impl OptimizerWrapper {
    fn step<B: AutodiffBackend>(
        &mut self,
        lr: f64,
        model: <ElasticPINN2D<B::InnerBackend> as burn::module::Module<B::InnerBackend>>::Record,
        grads: burn::module::Gradients,
    ) -> <ElasticPINN2D<B::InnerBackend> as burn::module::Module<B::InnerBackend>>::Record {
        match self {
            OptimizerWrapper::Adam(opt) => {
                // Burn 0.19 API: opt.step(lr, model, grads)
                opt.step(lr, model, grads)
            }
            OptimizerWrapper::Sgd(opt) => opt.step(lr, model, grads),
        }
    }
}
```

**Files to Update**:
- `src/solver/inverse/pinn/elastic_2d/training.rs` (primary changes)
- Check for any other uses of old Burn API

**Acceptance Criteria**:
- ✅ Optimizer wrapper compiles
- ✅ Training loop compiles
- ✅ No Burn API errors

---

### Task 3: Verify Test Execution [P0, 1 hour]

**Goal**: Confirm validation framework works correctly

**Steps**:
1. Compile with features: `cargo build --features pinn`
2. Run PINN validation tests: `cargo test --test pinn_elastic_validation --features pinn`
3. Run framework tests: `cargo test --test elastic_wave_validation_framework`
4. Verify all tests pass

**Expected Results**:
- Material property validation: PASS (all checks green)
- Wave speed validation: PASS (L² error < 1e-12)
- Plane wave PDE residual: PASS (residual < 1e-6)
- CFL timestep: PASS (dt within stability bounds)

**If Tests Fail**: Document failures, analyze root cause, fix issues

---

## Medium Priority Tasks

### Task 4: Implement Autodiff Stress Gradients [P1, 3-4 hours]

**Current State**: Placeholder finite-difference implementation in `loss.rs`

**Required**:
```rust
// Replace placeholders with Burn autodiff
pub fn compute_stress_divergence_autodiff<B: AutodiffBackend>(
    model: &ElasticPINN2D<B>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    t: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // Forward pass with gradient tracking
    let u = model.forward(x.clone(), y.clone(), t.clone());
    let ux = u.slice([0..u.dims()[0], 0..1]);
    let uy = u.slice([0..u.dims()[0], 1..2]);

    // Compute strain via autodiff
    let exx = ux.grad(&x);  // ∂uₓ/∂x
    let eyy = uy.grad(&y);  // ∂uᵧ/∂y
    let exy = (ux.grad(&y) + uy.grad(&x)) * 0.5;

    // Stress from strain (Hooke's law)
    let lambda = model.lambda.unwrap().val();
    let mu = model.mu.unwrap().val();
    let tr_e = exx.clone() + eyy.clone();
    let sxx = lambda * tr_e.clone() + 2.0 * mu * exx;
    let syy = lambda * tr_e + 2.0 * mu * eyy;
    let sxy = 2.0 * mu * exy;

    // Stress divergence via autodiff
    let div_x = sxx.grad(&x) + sxy.grad(&y);
    let div_y = sxy.grad(&x) + syy.grad(&y);

    (div_x, div_y)
}
```

**Benefits**:
- Higher accuracy (no finite-difference truncation error)
- Faster (GPU-accelerated)
- Consistent with Burn's autodiff framework

---

### Task 5: Add More Analytical Solutions [P2, 4-6 hours each]

**Priority Order**:
1. **Lamb's Problem** - Point force on half-space
2. **Rayleigh Waves** - Surface wave propagation
3. **Point Source Green's Function** - Fundamental solution

**Implementation Pattern** (same as `PlaneWaveSolution`):
```rust
pub struct LambSolution {
    pub source_location: [f64; 2],
    pub source_time: f64,
    pub lambda: f64,
    pub mu: f64,
    pub rho: f64,
}

impl LambSolution {
    pub fn displacement(&self, x: f64, y: f64, t: f64) -> [f64; 2] { ... }
    pub fn velocity(&self, x: f64, y: f64, t: f64) -> [f64; 2] { ... }
    // ... analytical derivatives
}
```

---

## Timeline

```
Week 1:
  Day 1-2: Task 1 (Sync trait resolution)     [4 hours]
  Day 2-3: Task 2 (Burn API fix)              [3 hours]
  Day 3:   Task 3 (Test execution)            [1 hour]
  
Week 2:
  Day 1-2: Task 4 (Autodiff stress)           [4 hours]
  Day 3-5: Task 5 (Analytical solutions)      [12 hours, 3 solutions]

Total: ~24 hours of focused work
```

---

## Success Criteria

### Phase 4 Complete When:
- ✅ All PINN validation tests pass (`cargo test --features pinn`)
- ✅ No compilation errors in PINN modules
- ✅ Validation framework works for both traditional and autodiff solvers
- ✅ At least 3 analytical solutions implemented and tested
- ✅ PDE residuals < 1e-6 for all analytical solutions
- ✅ Energy conservation error < 1e-10 for test cases
- ✅ Documentation updated (rustdoc, README, phase summary)

---

## Rollback Plan

If architectural changes cause issues:

1. **Sync Trait Separation Fails**:
   - Revert to single trait hierarchy
   - Accept that PINN cannot implement WaveEquation directly
   - Use adapter pattern: `WaveEquationAdapter<T: AutodiffElasticWaveEquation>`

2. **Burn API Update Fails**:
   - Pin to Burn 0.18.x in Cargo.toml
   - Document version constraint
   - Defer upgrade to future phase

3. **Test Failures Unresolvable**:
   - Document known limitations
   - Mark failing tests as `#[ignore]` with TODO comments
   - Create backlog items for investigation

---

## Communication

**Status Updates**: After each task completion
**Blockers**: Escalate immediately if stuck > 2 hours
**Documentation**: Update this plan as tasks complete

**Next Review**: After Task 3 (test execution) - reassess timeline and priorities

---

## Notes

- Keep changes minimal and focused
- One task at a time, verify before proceeding
- All code must maintain mathematical correctness
- No shortcuts, no placeholders in critical paths
- Document all architectural decisions