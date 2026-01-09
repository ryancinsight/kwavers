# Phase 1: Foundation & Math Layer — Execution Checklist

**Sprint:** Architecture Refactoring Phase 1  
**Duration:** Week 1 (5 working days)  
**Status:** 🔴 NOT STARTED  
**Priority:** P0 CRITICAL

---

## Overview

Phase 1 establishes the mathematical foundation and removes dead code. This is the most critical phase as it creates the base abstractions that all other layers will depend on.

**Goals:**
1. Remove all dead/deprecated code
2. Establish unified `math/numerics/operators/` module
3. Consolidate all numerical operations from scattered locations
4. Define clear trait-based interfaces
5. Ensure zero performance regression

---

## Pre-Flight Checklist

### Baseline Capture
- [ ] Run full test suite and capture results
  ```bash
  cargo test --all-features 2>&1 | tee baseline_tests_phase1.log
  ```
- [ ] Run benchmarks for performance baseline
  ```bash
  cargo bench 2>&1 | tee baseline_bench_phase1.log
  ```
- [ ] Generate current documentation
  ```bash
  cargo doc --all-features --no-deps
  ```
- [ ] Create backup branch
  ```bash
  git checkout -b refactor/phase1-foundation
  git push -u origin refactor/phase1-foundation
  ```

---

## Day 1: Dead Code Removal & Structure Setup

### Task 1.1: Delete Deprecated Files ⏱️ 1 hour

**Files to DELETE:**
- [ ] `src/domain/sensor/beamforming/adaptive/algorithms_old.rs` (2,199 lines)
- [ ] Search and delete all `*_old.rs` files:
  ```bash
  find src -name "*_old.rs" -type f
  ```
- [ ] Search and delete all `*_backup.rs` files:
  ```bash
  find src -name "*_backup.rs" -type f
  ```
- [ ] Search and delete all `*_deprecated.rs` files:
  ```bash
  find src -name "*_deprecated.rs" -type f
  ```

**Verification:**
```bash
cargo check --all-features
# Should compile (may have warnings about unused imports)
```

### Task 1.2: Create Math Module Structure ⏱️ 2 hours

**Create directory structure:**
```bash
mkdir -p src/math/numerics/operators
mkdir -p src/math/numerics/integration
mkdir -p src/math/numerics/transforms
```

**Create skeleton files:**

- [ ] `src/math/numerics/mod.rs`
  ```rust
  //! Numerical methods primitives
  //! 
  //! This module provides the foundational numerical operations used throughout kwavers.
  //! All numerical algorithms should be implemented here to avoid duplication.
  
  pub mod operators;
  pub mod integration;
  pub mod transforms;
  ```

- [ ] `src/math/numerics/operators/mod.rs`
  ```rust
  //! Numerical operators
  //! 
  //! Core numerical operations: differentiation, spectral transforms, interpolation
  
  pub mod differential;
  pub mod spectral;
  pub mod interpolation;
  
  // Re-export main traits
  pub use differential::DifferentialOperator;
  pub use spectral::SpectralOperator;
  pub use interpolation::Interpolator;
  ```

- [ ] Update `src/math/mod.rs` to include numerics:
  ```rust
  pub mod fft;
  pub mod geometry;
  pub mod linear_algebra;
  pub mod ml;
  pub mod numerics;  // NEW
  ```

**Verification:**
```bash
cargo check --lib
```

### Task 1.3: Define Core Traits ⏱️ 2 hours

- [ ] Create `src/math/numerics/operators/traits.rs` with foundational traits:

```rust
//! Core operator traits
//!
//! Defines the interface for all numerical operators in kwavers.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use crate::core::error::KwaversResult;

/// Trait for differential operators (finite difference, spectral differentiation)
pub trait DifferentialOperator: Send + Sync {
    /// Apply operator in X direction
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Apply operator in Y direction
    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Apply operator in Z direction
    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Get operator order (e.g., 2 for second-order accurate)
    fn order(&self) -> usize;
    
    /// Get stencil width
    fn stencil_width(&self) -> usize;
}

/// Trait for spectral operators (FFT-based operations)
pub trait SpectralOperator: Send + Sync {
    /// Apply operator in k-space
    fn apply_kspace(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Get wavenumber grid
    fn wavenumber_grid(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>);
}

/// Trait for interpolation operators
pub trait Interpolator: Send + Sync {
    /// Interpolate 1D data
    fn interpolate_1d(&self, data: ArrayView1<f64>, target_points: ArrayView1<f64>) 
        -> KwaversResult<Array1<f64>>;
    
    /// Interpolate 3D data
    fn interpolate_3d(&self, data: ArrayView3<f64>, 
                      target_x: ArrayView1<f64>,
                      target_y: ArrayView1<f64>,
                      target_z: ArrayView1<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Get interpolation order
    fn order(&self) -> usize;
}

/// Marker trait for operators that preserve energy/conservation laws
pub trait ConservativeOperator: DifferentialOperator {}

/// Marker trait for operators that are adjoint-consistent
pub trait AdjointConsistentOperator: DifferentialOperator {}
```

**Verification:**
```bash
cargo check --lib
```

---

## Day 2: Differential Operators Migration

### Task 2.1: Audit Current FD Implementations ⏱️ 2 hours

**Find all finite difference implementations:**
```bash
grep -r "finite.*difference\|stencil" src/solver/forward/fdtd/numerics/ --include="*.rs"
grep -r "fn.*gradient\|fn.*divergence\|fn.*laplacian" src/domain/grid/operators/ --include="*.rs"
```

**Document findings:**
- [ ] List all FD stencils found in `solver/forward/fdtd/numerics/`
- [ ] List all grid operators in `domain/grid/operators/`
- [ ] Identify which are duplicates
- [ ] Note any special cases or edge conditions

### Task 2.2: Create Unified Differential Module ⏱️ 4 hours

- [ ] Create `src/math/numerics/operators/differential.rs`

**Structure:**
```rust
//! Finite difference operators
//!
//! Unified implementation of all finite difference stencils used in kwavers.
//! All spatial derivatives should use these operators to ensure consistency.

use ndarray::{Array3, ArrayView3, Axis};
use crate::core::error::{KwaversResult, NumericalError};
use super::traits::DifferentialOperator;

/// Second-order accurate central difference operator
#[derive(Debug, Clone)]
pub struct CentralDifference2 {
    dx: f64,
    dy: f64,
    dz: f64,
}

impl CentralDifference2 {
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing.into());
        }
        Ok(Self { dx, dy, dz })
    }
}

impl DifferentialOperator for CentralDifference2 {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut result = Array3::zeros((nx, ny, nz));
        
        // Interior points: central difference
        for i in 1..nx-1 {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i,j,k]] = (field[[i+1,j,k]] - field[[i-1,j,k]]) / (2.0 * self.dx);
                }
            }
        }
        
        // Boundary points: forward/backward difference
        for j in 0..ny {
            for k in 0..nz {
                result[[0,j,k]] = (field[[1,j,k]] - field[[0,j,k]]) / self.dx;
                result[[nx-1,j,k]] = (field[[nx-1,j,k]] - field[[nx-2,j,k]]) / self.dx;
            }
        }
        
        Ok(result)
    }
    
    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        // Similar implementation for Y direction
        todo!("Implement Y direction")
    }
    
    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        // Similar implementation for Z direction
        todo!("Implement Z direction")
    }
    
    fn order(&self) -> usize { 2 }
    fn stencil_width(&self) -> usize { 3 }
}

// Additional operators:
// - FourthOrderCentralDifference
// - StaggeredGridOperator
// - UpwindDifference
// - etc.
```

- [ ] Implement all required operators from audit
- [ ] Add comprehensive unit tests
- [ ] Add property-based tests for conservation
- [ ] Document each operator with literature references

**Verification:**
```bash
cargo test --lib math::numerics::operators::differential
```

### Task 2.3: Migrate FDTD Numerics ⏱️ 2 hours

**Source files:**
- `src/solver/forward/fdtd/numerics/finite_difference.rs`
- `src/solver/forward/fdtd/numerics/staggered_grid.rs`

**Actions:**
- [ ] Copy implementations to new `differential.rs`
- [ ] Adapt to new trait interface
- [ ] Remove redundant code
- [ ] Update `solver/forward/fdtd/` to import from `math::numerics`
- [ ] Mark old files as deprecated (keep temporarily for reference)

**Verification:**
```bash
cargo test solver::forward::fdtd
```

---

## Day 3: Spectral Operators Migration

### Task 3.1: Audit Spectral Implementations ⏱️ 1 hour

**Find all spectral operations:**
```bash
grep -r "fft\|spectral\|k_space\|wavenumber" src/solver/forward/pstd/numerics/ --include="*.rs"
```

- [ ] List all spectral operators
- [ ] Identify FFT dependencies
- [ ] Note any special windowing or filtering

### Task 3.2: Create Unified Spectral Module ⏱️ 4 hours

- [ ] Create `src/math/numerics/operators/spectral.rs`

**Structure:**
```rust
//! Spectral operators
//!
//! FFT-based differential and filtering operators for pseudospectral methods.

use ndarray::{Array1, Array3, ArrayView3};
use rustfft::{FftPlanner, num_complex::Complex};
use crate::core::error::KwaversResult;
use super::traits::SpectralOperator;

/// Pseudospectral derivative operator using FFT
#[derive(Debug)]
pub struct PseudospectralDerivative {
    nx: usize,
    ny: usize,
    nz: usize,
    kx: Array1<f64>,
    ky: Array1<f64>,
    kz: Array1<f64>,
    // FFT workspace
}

impl PseudospectralDerivative {
    pub fn new(nx: usize, ny: usize, nz: usize, 
               dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        // Generate wavenumber grids
        let kx = Self::wavenumber_vector(nx, dx);
        let ky = Self::wavenumber_vector(ny, dy);
        let kz = Self::wavenumber_vector(nz, dz);
        
        Ok(Self { nx, ny, nz, kx, ky, kz })
    }
    
    fn wavenumber_vector(n: usize, d: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let dk = 2.0 * std::f64::consts::PI / (n as f64 * d);
        
        for i in 0..n/2 {
            k[i] = i as f64 * dk;
        }
        for i in n/2..n {
            k[i] = (i as i64 - n as i64) as f64 * dk;
        }
        k
    }
}

impl SpectralOperator for PseudospectralDerivative {
    fn apply_kspace(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        // 1. Forward FFT
        // 2. Multiply by ik
        // 3. Inverse FFT
        todo!("Implement spectral derivative")
    }
    
    fn wavenumber_grid(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (self.kx.clone(), self.ky.clone(), self.kz.clone())
    }
}

// Additional operators:
// - SpectralFilter (anti-aliasing)
// - SpectralSmoothing
// - KSpaceCorrection (dispersion correction)
```

- [ ] Implement spectral derivative
- [ ] Implement spectral filters
- [ ] Add comprehensive tests
- [ ] Benchmark against current implementation

**Verification:**
```bash
cargo test --lib math::numerics::operators::spectral
cargo bench spectral_operators
```

### Task 3.3: Migrate PSTD Numerics ⏱️ 3 hours

**Source files:**
- `src/solver/forward/pstd/numerics/operators/spectral.rs`

**Actions:**
- [ ] Copy implementations to new `spectral.rs`
- [ ] Adapt to new trait interface
- [ ] Update `solver/forward/pstd/` to import from `math::numerics`
- [ ] Remove redundant code
- [ ] Verify no performance regression

**Verification:**
```bash
cargo test solver::forward::pstd
cargo bench pstd_solver
```

---

## Day 4: Interpolation Operators Migration

### Task 4.1: Audit Interpolation Implementations ⏱️ 1 hour

**Find all interpolation code:**
```bash
find src/domain/medium/heterogeneous/interpolation/ -name "*.rs"
grep -r "interpolate\|lerp\|trilinear" src/ --include="*.rs"
```

- [ ] List all interpolation methods
- [ ] Identify use cases (medium properties, sensor data, etc.)
- [ ] Note any special requirements (conservative, monotonic, etc.)

### Task 4.2: Create Unified Interpolation Module ⏱️ 4 hours

- [ ] Create `src/math/numerics/operators/interpolation.rs`

**Structure:**
```rust
//! Interpolation operators
//!
//! Unified interpolation methods for spatial data.

use ndarray::{Array1, Array3, ArrayView1, ArrayView3};
use crate::core::error::KwaversResult;
use super::traits::Interpolator;

/// Trilinear interpolation
#[derive(Debug, Clone)]
pub struct TrilinearInterpolator {
    dx: f64,
    dy: f64,
    dz: f64,
}

impl TrilinearInterpolator {
    pub fn new(dx: f64, dy: f64, dz: f64) -> Self {
        Self { dx, dy, dz }
    }
}

impl Interpolator for TrilinearInterpolator {
    fn interpolate_3d(&self, data: ArrayView3<f64>,
                      target_x: ArrayView1<f64>,
                      target_y: ArrayView1<f64>,
                      target_z: ArrayView1<f64>) -> KwaversResult<Array3<f64>> {
        // Standard trilinear interpolation
        todo!("Implement trilinear interpolation")
    }
    
    fn interpolate_1d(&self, data: ArrayView1<f64>, target_points: ArrayView1<f64>) 
        -> KwaversResult<Array1<f64>> {
        // Linear interpolation
        todo!("Implement 1D interpolation")
    }
    
    fn order(&self) -> usize { 1 }
}

// Additional interpolators:
// - CubicSplineInterpolator
// - SpectralInterpolator
// - ConservativeInterpolator (for medium boundaries)
```

- [ ] Implement trilinear interpolation
- [ ] Implement cubic spline interpolation
- [ ] Add property tests (e.g., interpolation at grid points = original values)
- [ ] Document interpolation order and accuracy

**Verification:**
```bash
cargo test --lib math::numerics::operators::interpolation
```

### Task 4.3: Migrate Medium Interpolation ⏱️ 3 hours

**Source files:**
- `src/domain/medium/heterogeneous/interpolation/*.rs`

**Actions:**
- [ ] Move implementations to `math/numerics/operators/interpolation.rs`
- [ ] Update `domain/medium/heterogeneous/` to use new interpolation
- [ ] Remove old interpolation module
- [ ] Verify medium property interpolation works correctly

**Verification:**
```bash
cargo test domain::medium::heterogeneous
```

---

## Day 5: Integration, Testing, and Documentation

### Task 5.1: Update All References ⏱️ 2 hours

**Find all usage of old implementations:**
```bash
# Find FDTD references
grep -r "solver::forward::fdtd::numerics" src/ --include="*.rs"

# Find PSTD references  
grep -r "solver::forward::pstd::numerics::operators" src/ --include="*.rs"

# Find interpolation references
grep -r "domain::medium::heterogeneous::interpolation" src/ --include="*.rs"
```

- [ ] Update all imports to use `math::numerics::operators::`
- [ ] Remove or deprecate old modules
- [ ] Update re-exports in module files

### Task 5.2: Comprehensive Testing ⏱️ 3 hours

- [ ] Run full unit test suite
  ```bash
  cargo test --lib --all-features
  ```

- [ ] Run integration tests
  ```bash
  cargo test --test '*' --all-features
  ```

- [ ] Run benchmarks and compare to baseline
  ```bash
  cargo bench
  diff baseline_bench_phase1.log <benchmark_output>
  ```

- [ ] Verify conservation properties
  ```bash
  cargo test --test energy_conservation_test
  ```

- [ ] Check for performance regressions (tolerance: ±5%)

### Task 5.3: Documentation Updates ⏱️ 2 hours

- [ ] Update module-level documentation
- [ ] Add examples to each operator trait
- [ ] Document literature references for each method
- [ ] Update `docs/adr.md` with Phase 1 decisions
- [ ] Add migration notes to CHANGELOG

**Example documentation:**
```rust
//! # Differential Operators
//!
//! This module provides finite difference operators for spatial derivatives.
//!
//! ## Literature References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on 
//!   arbitrarily spaced grids." Mathematics of Computation, 51(184), 699-706.
//! - Shubin, G. R., & Bell, J. B. (1987). "A modified equation approach to 
//!   constructing fourth order methods for acoustic wave propagation."
//!
//! ## Examples
//!
//! ```rust
//! use kwavers::math::numerics::operators::{DifferentialOperator, CentralDifference2};
//!
//! let op = CentralDifference2::new(0.001, 0.001, 0.001)?;
//! let gradient = op.apply_x(field.view())?;
//! ```
```

### Task 5.4: Phase 1 Sign-off ⏱️ 1 hour

- [ ] Verify all Day 1-4 tasks completed
- [ ] All tests passing
- [ ] No performance regressions
- [ ] Documentation complete
- [ ] Code review checklist:
  - [ ] All files <500 lines
  - [ ] Trait-based abstractions
  - [ ] Zero unsafe code (unless justified)
  - [ ] Comprehensive error handling
  - [ ] Literature citations

- [ ] Commit and push Phase 1
  ```bash
  git add src/math/numerics/
  git commit -m "Phase 1: Establish math/numerics foundation
  
  - Created unified differential operators
  - Created unified spectral operators
  - Created unified interpolation operators
  - Migrated FDTD numerics
  - Migrated PSTD numerics
  - Migrated medium interpolation
  - All tests passing
  - Zero performance regression"
  
  git push origin refactor/phase1-foundation
  ```

- [ ] Create pull request with Phase 1 summary
- [ ] Update `ARCHITECTURE_REFACTORING_AUDIT.md` Phase 1 status: ✅ COMPLETE

---

## Success Criteria

### Must Have (Blockers for Phase 2)
- ✅ All tests passing
- ✅ Zero performance regression (within ±5%)
- ✅ All dead code removed
- ✅ `math/numerics/operators/` module complete and tested
- ✅ All solver modules using new math layer

### Should Have
- ✅ All files <500 lines
- ✅ Comprehensive documentation
- ✅ Literature references cited
- ✅ Property-based tests for conservation laws

### Nice to Have
- Performance improvements documented
- Benchmark comparisons in PR
- Migration guide for external users

---

## Rollback Plan

If critical issues arise:

1. **Performance Regression >10%**
   - Investigate bottleneck with profiling
   - Optimize hot path
   - If unsolvable, revert specific operator

2. **Test Failures**
   - Identify failing test
   - Fix bug or update test
   - If widespread, revert to baseline

3. **Compilation Issues**
   - Fix import errors
   - Update module re-exports
   - Verify feature flags

4. **Emergency Rollback**
   ```bash
   git revert <commit-hash>
   git push origin refactor/phase1-foundation
   ```

---

## Daily Standup Template

**What I completed yesterday:**
- [ ] Task X.Y completed
- [ ] N tests passing
- [ ] Issue: <description>

**What I'm working on today:**
- [ ] Task X.Y in progress
- [ ] Expected completion: <time>

**Blockers:**
- None / <describe blocker>

---

## Notes & Lessons Learned

### Technical Decisions
- [ ] Document any deviations from plan
- [ ] Note any unexpected challenges
- [ ] Record performance insights

### Process Improvements
- [ ] Time estimates vs actual
- [ ] Testing strategy effectiveness
- [ ] Documentation clarity

---

**Phase 1 Status:** 🔴 NOT STARTED → 🟡 IN PROGRESS → 🟢 COMPLETE

**Next Phase:** Phase 2 - Domain Layer Purification (Week 2)