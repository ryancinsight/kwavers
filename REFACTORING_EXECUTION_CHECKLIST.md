# Refactoring Execution Checklist — kwavers
**Tactical Execution Plan for Deep Vertical Hierarchy Refactoring**

**Date:** 2025-01-12  
**Status:** 🔴 READY FOR EXECUTION  
**Owner:** Elite Mathematically-Verified Systems Architect  
**Mandate:** Zero tolerance for errors, complete verification at each step

---

## Overview

This checklist provides step-by-step execution instructions for the architectural refactoring outlined in `DEEP_VERTICAL_HIERARCHY_AUDIT.md`. Each task is atomic, verifiable, and reversible.

**Total Effort:** 6 weeks, 4 phases, 10 sprints  
**Test Requirement:** 867 tests must pass after EVERY sprint  
**File Size Limit:** <500 lines per file (GRASP compliance)

---

## Pre-Refactoring Checklist

### ✅ Prerequisites

- [ ] **Backup created:** Full repository backup to external location
- [ ] **Git branch created:** `refactor/deep-vertical-hierarchy` from `main`
- [ ] **Baseline tests passing:** `cargo test --all-features` (867/867 passing)
- [ ] **Baseline benchmarks:** `cargo bench` (record results)
- [ ] **Baseline build time:** `time cargo build --release` (record time)
- [ ] **Documentation reviewed:** All audit documents read and understood
- [ ] **Team notification:** All stakeholders informed of refactoring schedule

### ✅ Tooling Setup

```bash
# Install required tools
cargo install cargo-depgraph
cargo install cargo-modules
cargo install tokei  # Line counting

# Verify baseline metrics
tokei src/
cargo test --all-features 2>&1 | tee baseline_tests.log
cargo bench 2>&1 | tee baseline_benchmarks.log
time cargo build --release 2>&1 | tee baseline_build.log

# Create safety snapshot
git tag refactor-baseline-2025-01-12
git push origin refactor-baseline-2025-01-12
```

---

## Phase 1: Critical Duplication Removal (Week 1-2)

### Sprint 1A: Beamforming Consolidation (Days 1-3)

**Objective:** Eliminate beamforming duplication by consolidating in `analysis/signal_processing/beamforming/`

#### Step 1A.1: Create Canonical Structure

- [ ] **Create directories:**
  ```bash
  mkdir -p src/analysis/signal_processing/beamforming/core
  mkdir -p src/analysis/signal_processing/beamforming/time_domain
  mkdir -p src/analysis/signal_processing/beamforming/frequency_domain
  mkdir -p src/analysis/signal_processing/beamforming/adaptive
  mkdir -p src/analysis/signal_processing/beamforming/neural
  mkdir -p src/analysis/signal_processing/beamforming/utils
  ```

- [ ] **Create stub modules:**
  ```bash
  touch src/analysis/signal_processing/beamforming/core/traits.rs
  touch src/analysis/signal_processing/beamforming/core/geometry.rs
  touch src/analysis/signal_processing/beamforming/core/mod.rs
  ```

#### Step 1A.2: Split Large Files First

**Task:** Split `domain/sensor/beamforming/experimental/neural.rs` (3,115 lines → 7 modules)

- [ ] **Create target structure:**
  ```bash
  mkdir -p src/analysis/signal_processing/beamforming/neural
  touch src/analysis/signal_processing/beamforming/neural/architecture.rs
  touch src/analysis/signal_processing/beamforming/neural/training.rs
  touch src/analysis/signal_processing/beamforming/neural/inference.rs
  touch src/analysis/signal_processing/beamforming/neural/hybrid.rs
  touch src/analysis/signal_processing/beamforming/neural/pinn.rs
  touch src/analysis/signal_processing/beamforming/neural/evaluation.rs
  touch src/analysis/signal_processing/beamforming/neural/mod.rs
  ```

- [ ] **Extract components:**
  - [ ] Architecture definitions → `architecture.rs` (<500 lines)
  - [ ] Training procedures → `training.rs` (<500 lines)
  - [ ] Inference logic → `inference.rs` (<500 lines)
  - [ ] Hybrid methods → `hybrid.rs` (<500 lines)
  - [ ] PINN integration → `pinn.rs` (<500 lines)
  - [ ] Evaluation metrics → `evaluation.rs` (<500 lines)
  - [ ] Public API → `mod.rs` (<200 lines)

- [ ] **Verify line counts:**
  ```bash
  wc -l src/analysis/signal_processing/beamforming/neural/*.rs
  # All files must be <500 lines
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::neural
  ```

**Task:** Split `domain/sensor/beamforming/beamforming_3d.rs` (1,260 lines → 3 modules)

- [ ] **Create target structure:**
  ```bash
  touch src/analysis/signal_processing/beamforming/spatial/volume_beamformer.rs
  touch src/analysis/signal_processing/beamforming/spatial/array_geometry_3d.rs
  touch src/analysis/signal_processing/beamforming/spatial/mod.rs
  ```

- [ ] **Extract components:**
  - [ ] Volume beamforming → `volume_beamformer.rs` (<500 lines)
  - [ ] 3D array geometry → `array_geometry_3d.rs` (<500 lines)
  - [ ] Public API → `mod.rs` (<300 lines)

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::spatial
  ```

**Task:** Split `domain/sensor/beamforming/ai_integration.rs` (1,148 lines → 3 modules)

- [ ] **Create target structure:**
  ```bash
  touch src/analysis/signal_processing/beamforming/neural/model_loader.rs
  touch src/analysis/signal_processing/beamforming/neural/feature_extraction.rs
  touch src/analysis/signal_processing/beamforming/neural/postprocessing.rs
  ```

- [ ] **Extract components:**
  - [ ] Model loading → `model_loader.rs` (<400 lines)
  - [ ] Feature extraction → `feature_extraction.rs` (<400 lines)
  - [ ] Post-processing → `postprocessing.rs` (<400 lines)

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::neural
  ```

#### Step 1A.3: Migrate Time-Domain Beamforming

**Task:** Consolidate DAS implementations

- [ ] **Review existing implementations:**
  ```bash
  grep -r "delay_and_sum\|DelayAndSum\|DAS" src/domain/sensor/beamforming/time_domain/
  grep -r "delay_and_sum\|DelayAndSum\|DAS" src/analysis/signal_processing/beamforming/time_domain/
  ```

- [ ] **Create canonical DAS implementation:**
  ```rust
  // src/analysis/signal_processing/beamforming/time_domain/das.rs
  
  /// Canonical Delay-And-Sum beamformer implementation (SSOT)
  pub struct DelayAndSumBeamformer {
      // Implementation
  }
  ```

- [ ] **Migrate tests:**
  ```bash
  git mv tests/beamforming/das_test.rs tests/signal_processing/das_test.rs
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/sensor/beamforming/time_domain/das/*.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::time_domain::das
  ```

**Task:** Consolidate delay calculation utilities

- [ ] **Compare implementations:**
  ```bash
  diff -u src/domain/sensor/beamforming/time_domain/delay_calculation.rs \
          src/analysis/signal_processing/beamforming/utils/delays.rs
  ```

- [ ] **Create canonical implementation:**
  ```rust
  // src/analysis/signal_processing/beamforming/core/geometry.rs
  
  /// Geometric delay calculation (SSOT)
  pub fn calculate_delays(
      sensor_positions: &Array2<f64>,
      focus_point: &[f64; 3],
      sound_speed: f64,
  ) -> Array1<f64> {
      // Implementation
  }
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/sensor/beamforming/time_domain/delay_calculation.rs
  git rm src/analysis/signal_processing/beamforming/utils/delays.rs
  ```

- [ ] **Update imports:**
  ```bash
  # Use sed or manual find-replace
  find src -name "*.rs" -exec sed -i 's/domain::sensor::beamforming::time_domain::delay_calculation/analysis::signal_processing::beamforming::core::geometry/g' {} +
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::core
  ```

#### Step 1A.4: Migrate Frequency-Domain Beamforming

**Task:** Consolidate MVDR implementations

- [ ] **Compare implementations:**
  ```bash
  diff -u src/domain/sensor/beamforming/adaptive/adaptive.rs \
          src/analysis/signal_processing/beamforming/adaptive/mvdr.rs
  ```

- [ ] **Create canonical MVDR:**
  ```rust
  // src/analysis/signal_processing/beamforming/frequency_domain/mvdr.rs
  
  /// Minimum Variance Distortionless Response (MVDR) beamformer (SSOT)
  pub struct MvdrBeamformer {
      // Implementation
  }
  ```

- [ ] **Delete duplicates:**
  ```bash
  # Keep analysis version, delete domain version
  git rm src/domain/sensor/beamforming/adaptive/adaptive.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::frequency_domain::mvdr
  ```

**Task:** Consolidate MUSIC implementations

- [ ] **Compare implementations:**
  ```bash
  diff -u src/domain/sensor/beamforming/narrowband/music.rs \
          src/analysis/signal_processing/beamforming/adaptive/subspace.rs
  ```

- [ ] **Create canonical MUSIC:**
  ```rust
  // src/analysis/signal_processing/beamforming/frequency_domain/music.rs
  
  /// Multiple Signal Classification (MUSIC) algorithm (SSOT)
  pub struct MusicBeamformer {
      // Implementation
  }
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/sensor/beamforming/narrowband/music.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::frequency_domain::music
  ```

#### Step 1A.5: Migrate Covariance Estimation

**Task:** Consolidate covariance matrix utilities

- [ ] **Compare implementations:**
  ```bash
  diff -u src/domain/sensor/beamforming/covariance.rs \
          src/analysis/signal_processing/beamforming/covariance/mod.rs
  ```

- [ ] **Create canonical covariance estimation:**
  ```rust
  // src/analysis/signal_processing/beamforming/utils/covariance.rs (<500 lines)
  
  /// Spatial covariance matrix estimation (SSOT)
  pub fn estimate_covariance_matrix(
      data: &Array3<Complex<f64>>,
      spatial_smoothing: bool,
  ) -> Array2<Complex<f64>> {
      // Implementation
  }
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/sensor/beamforming/covariance.rs
  git rm src/analysis/signal_processing/beamforming/covariance/mod.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib analysis::signal_processing::beamforming::utils::covariance
  ```

#### Step 1A.6: Delete domain/sensor/beamforming

- [ ] **Verify all functionality migrated:**
  ```bash
  # List remaining files
  find src/domain/sensor/beamforming -name "*.rs"
  # Should only be deprecated stubs with migration notices
  ```

- [ ] **Create deprecation notices:**
  ```rust
  // src/domain/sensor/beamforming/mod.rs
  
  #[deprecated(
      since = "2.15.0",
      note = "Beamforming moved to analysis::signal_processing::beamforming"
  )]
  pub use crate::analysis::signal_processing::beamforming as new_beamforming;
  ```

- [ ] **Delete after 1-2 releases:**
  ```bash
  # In future release (2.17.0+):
  git rm -r src/domain/sensor/beamforming/
  ```

#### Step 1A.7: Update Imports (150+ files)

- [ ] **Generate import map:**
  ```bash
  grep -r "use crate::domain::sensor::beamforming" src/ > import_map.txt
  wc -l import_map.txt  # Count affected files
  ```

- [ ] **Automated replacement:**
  ```bash
  find src -name "*.rs" -exec sed -i \
    's/domain::sensor::beamforming/analysis::signal_processing::beamforming/g' {} +
  ```

- [ ] **Manual verification:** Review 10 random files for correctness

- [ ] **Compile check:**
  ```bash
  cargo check --all-features
  # Must succeed with zero errors
  ```

#### Step 1A.8: Verification

- [ ] **Run full test suite:**
  ```bash
  cargo test --all-features
  # Expected: 867/867 tests passing
  ```

- [ ] **Check file sizes:**
  ```bash
  find src/analysis/signal_processing/beamforming -name "*.rs" -exec wc -l {} \; | \
    awk '$1 > 500 {print "VIOLATION: " $2 " has " $1 " lines"}'
  # Expected: No output
  ```

- [ ] **Verify no duplication:**
  ```bash
  # Check for duplicate function names
  grep -r "fn delay_and_sum" src/ | wc -l
  # Expected: 1 (SSOT)
  ```

- [ ] **Performance benchmark:**
  ```bash
  cargo bench --bench beamforming_benchmarks
  # Compare with baseline (no regression >5%)
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 1A: Beamforming consolidation complete

  - Consolidated 38 files from domain/sensor/beamforming
  - Migrated to analysis/signal_processing/beamforming
  - Split 3 large files (3,115, 1,260, 1,148 lines)
  - All files now <500 lines (GRASP compliant)
  - Zero duplication (SSOT enforced)
  - All 867 tests passing
  - Zero performance regression"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

### Sprint 1B: Grid Operations Consolidation (Days 4-6)

**Objective:** Eliminate grid operator duplication by consolidating in `math/numerics/differentiation/`

#### Step 1B.1: Create Canonical Structure

- [ ] **Create directories:**
  ```bash
  mkdir -p src/math/numerics/differentiation/finite_difference
  mkdir -p src/math/numerics/differentiation/spectral
  mkdir -p src/math/numerics/differentiation/dg
  mkdir -p src/math/numerics/interpolation
  ```

- [ ] **Create stub modules:**
  ```bash
  touch src/math/numerics/differentiation/finite_difference/stencils.rs
  touch src/math/numerics/differentiation/finite_difference/gradient.rs
  touch src/math/numerics/differentiation/finite_difference/laplacian.rs
  touch src/math/numerics/differentiation/finite_difference/divergence.rs
  touch src/math/numerics/differentiation/finite_difference/curl.rs
  touch src/math/numerics/differentiation/finite_difference/mod.rs
  touch src/math/numerics/differentiation/traits.rs
  touch src/math/numerics/differentiation/mod.rs
  ```

#### Step 1B.2: Migrate Finite Difference Operators

**Task:** Consolidate gradient operators

- [ ] **Review existing implementations:**
  ```bash
  find src -name "gradient*.rs" -o -name "*gradient.rs" | xargs grep -l "pub fn gradient"
  ```

- [ ] **Compare implementations:**
  ```bash
  # Compare domain/grid/operators/gradient.rs with solver implementations
  diff -u src/domain/grid/operators/gradient.rs \
          src/solver/forward/fdtd/numerics/gradient.rs
  ```

- [ ] **Create canonical gradient operator:**
  ```rust
  // src/math/numerics/differentiation/finite_difference/gradient.rs (<500 lines)
  
  use ndarray::{Array3, ArrayView3};
  
  /// Finite difference gradient operators (SSOT)
  pub trait GradientOperator {
      fn gradient_x(&self, field: ArrayView3<f64>, dx: f64) -> Array3<f64>;
      fn gradient_y(&self, field: ArrayView3<f64>, dy: f64) -> Array3<f64>;
      fn gradient_z(&self, field: ArrayView3<f64>, dz: f64) -> Array3<f64>;
  }
  
  /// Second-order accurate gradient
  pub struct SecondOrderGradient;
  
  impl GradientOperator for SecondOrderGradient {
      fn gradient_x(&self, field: ArrayView3<f64>, dx: f64) -> Array3<f64> {
          // Central difference implementation
      }
      // ...
  }
  
  /// Fourth-order accurate gradient
  pub struct FourthOrderGradient;
  
  impl GradientOperator for FourthOrderGradient {
      // Implementation
  }
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/grid/operators/gradient.rs
  git rm src/domain/grid/operators/gradient_optimized.rs
  # Remove from solver implementations after verification
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib math::numerics::differentiation::finite_difference::gradient
  ```

**Task:** Consolidate laplacian operators

- [ ] **Create canonical laplacian:**
  ```rust
  // src/math/numerics/differentiation/finite_difference/laplacian.rs (<500 lines)
  
  /// Finite difference Laplacian operators (SSOT)
  pub trait LaplacianOperator {
      fn laplacian(&self, field: ArrayView3<f64>, dx: f64, dy: f64, dz: f64) -> Array3<f64>;
  }
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/grid/operators/laplacian.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib math::numerics::differentiation::finite_difference::laplacian
  ```

**Task:** Consolidate divergence and curl operators

- [ ] **Create canonical divergence:**
  ```rust
  // src/math/numerics/differentiation/finite_difference/divergence.rs (<500 lines)
  ```

- [ ] **Create canonical curl:**
  ```rust
  // src/math/numerics/differentiation/finite_difference/curl.rs (<500 lines)
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/domain/grid/operators/divergence.rs
  git rm src/domain/grid/operators/curl.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib math::numerics::differentiation::finite_difference
  ```

#### Step 1B.3: Consolidate Stencil Definitions

**Task:** Create unified stencil coefficient repository

- [ ] **Create stencil module:**
  ```rust
  // src/math/numerics/differentiation/finite_difference/stencils.rs (<500 lines)
  
  /// Finite difference stencil coefficients (SSOT)
  
  /// Second-order central difference stencil
  pub const CENTRAL_2ND_ORDER: [f64; 3] = [-0.5, 0.0, 0.5];
  
  /// Fourth-order central difference stencil
  pub const CENTRAL_4TH_ORDER: [f64; 5] = [
      1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0
  ];
  
  /// Eighth-order central difference stencil
  pub const CENTRAL_8TH_ORDER: [f64; 9] = [
      -1.0/280.0, 4.0/105.0, -1.0/5.0, 4.0/5.0, 0.0,
      -4.0/5.0, 1.0/5.0, -4.0/105.0, 1.0/280.0
  ];
  
  // Forward/backward stencils, Laplacian stencils, etc.
  ```

- [ ] **Extract from solver implementations:**
  ```bash
  grep -r "stencil\|coefficient" src/solver/forward/fdtd/
  # Consolidate all found stencils
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib math::numerics::differentiation::finite_difference::stencils
  ```

#### Step 1B.4: Migrate Spectral Operators

**Task:** Consolidate spectral differentiation

- [ ] **Create spectral module:**
  ```rust
  // src/math/numerics/differentiation/spectral/fourier.rs (<500 lines)
  
  use ndarray::{Array3, ArrayView3};
  use crate::math::fft::{Fft3d, get_fft_for_grid};
  
  /// Fourier spectral differentiation (SSOT)
  pub struct FourierDifferentiator {
      fft: Fft3d,
      kx: Array1<f64>,
      ky: Array1<f64>,
      kz: Array1<f64>,
  }
  
  impl FourierDifferentiator {
      pub fn gradient(&self, field: ArrayView3<f64>) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
          // Spectral gradient via FFT
      }
  }
  ```

- [ ] **Extract from PSTD solver:**
  ```bash
  # Review and extract spectral operators
  cat src/solver/forward/pstd/numerics/operators/spectral.rs
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm src/solver/forward/pstd/numerics/operators/spectral.rs
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib math::numerics::differentiation::spectral
  ```

#### Step 1B.5: Migrate Interpolation

**Task:** Consolidate grid interpolation

- [ ] **Create interpolation module:**
  ```rust
  // src/math/numerics/interpolation/linear.rs (<500 lines)
  
  /// Linear interpolation in 1D/2D/3D (SSOT)
  pub fn interpolate_1d(data: &[f64], x: f64) -> f64 { /* ... */ }
  pub fn interpolate_2d(data: &Array2<f64>, x: f64, y: f64) -> f64 { /* ... */ }
  pub fn interpolate_3d(data: &Array3<f64>, x: f64, y: f64, z: f64) -> f64 { /* ... */ }
  ```

- [ ] **Extract from medium:**
  ```bash
  cat src/domain/medium/heterogeneous/interpolation/trilinear.rs
  # Move to math/numerics/interpolation/
  ```

- [ ] **Delete duplicates:**
  ```bash
  git rm -r src/domain/medium/heterogeneous/interpolation/
  ```

- [ ] **Run tests:**
  ```bash
  cargo test --package kwavers --lib math::numerics::interpolation
  ```

#### Step 1B.6: Delete domain/grid/operators

- [ ] **Verify all functionality migrated:**
  ```bash
  ls src/domain/grid/operators/
  # Should be empty or only deprecated stubs
  ```

- [ ] **Delete directory:**
  ```bash
  git rm -r src/domain/grid/operators/
  ```

- [ ] **Update domain/grid/mod.rs:**
  ```rust
  // Remove: pub mod operators;
  // Add deprecation notice if needed
  ```

#### Step 1B.7: Update Solver Imports

- [ ] **Generate import map:**
  ```bash
  grep -r "domain::grid::operators\|solver::.*/numerics/operators" src/ > grid_import_map.txt
  ```

- [ ] **Automated replacement:**
  ```bash
  find src -name "*.rs" -exec sed -i \
    's/domain::grid::operators/math::numerics::differentiation::finite_difference/g' {} +
  
  find src -name "*.rs" -exec sed -i \
    's/solver::forward::pstd::numerics::operators/math::numerics::differentiation::spectral/g' {} +
  ```

- [ ] **Compile check:**
  ```bash
  cargo check --all-features
  ```

#### Step 1B.8: Verification

- [ ] **Run full test suite:**
  ```bash
  cargo test --all-features
  # Expected: 867/867 tests passing
  ```

- [ ] **Verify SSOT:**
  ```bash
  grep -r "fn gradient_x" src/ | wc -l
  # Expected: 1-2 (trait definition + implementation)
  ```

- [ ] **Performance benchmark:**
  ```bash
  cargo bench --bench grid_benchmarks
  # Compare with baseline
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 1B: Grid operations consolidation complete

  - Consolidated operators from domain/grid/operators
  - Extracted stencils from solver implementations
  - Unified in math/numerics/differentiation
  - Zero duplication (SSOT enforced)
  - All 867 tests passing"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

### Sprint 1C: Physics-Solver Separation (Days 7-10)

**Objective:** Separate physics equations from numerical solvers

#### Step 1C.1: Create Physics Models Structure

- [ ] **Create directories:**
  ```bash
  mkdir -p src/physics/acoustics/models/linear
  mkdir -p src/physics/acoustics/models/nonlinear
  mkdir -p src/physics/elasticity/models
  mkdir -p src/physics/thermal/models
  ```

#### Step 1C.2: Move Acoustic Wave Models

**Task:** Move linear acoustics from solver to physics

- [ ] **Move acoustic plugin:**
  ```bash
  git mv src/solver/forward/acoustic/plugin.rs \
         src/physics/acoustics/models/linear/wave_equation.rs
  ```

- [ ] **Refactor to physics model:**
  ```rust
  // src/physics/acoustics/models/linear/wave_equation.rs
  
  /// Linear acoustic wave equation (physics ONLY)
  pub struct LinearAcousticWave {
      // No solver logic, only physics
  }
  
  impl LinearAcousticWave {
      pub fn compute_pressure_rate(&self, ...) -> f64 { /* Physics */ }
      pub fn compute_velocity_rate(&self, ...) -> [f64; 3] { /* Physics */ }
  }
  ```

- [ ] **Delete old solver file:**
  ```bash
  git rm -r src/solver/forward/acoustic/
  ```

**Task:** Move nonlinear acoustics models

- [ ] **Move Kuznetsov equation:**
  ```bash
  git mv src/solver/forward/nonlinear/kuznetsov/ \
         src/physics/acoustics/models/nonlinear/kuznetsov/
  ```

- [ ] **Refactor to physics model:**
  ```rust
  // src/physics/acoustics/models/nonlinear/kuznetsov/equation.rs
  
  /// Kuznetsov equation for nonlinear acoustics (physics ONLY)
  pub struct KuznetsovEquation {
      // Physics parameters
  }
  
  impl KuznetsovEquation {
      pub fn nonlinear_term(&self, ...) -> f64 { /* B/A nonlinearity */ }
      pub fn absorption_term(&self, ...) -> f64 { /* Attenuation */ }
  }
  ```

- [ ] **Move KZK equation:**
  ```bash
  git mv src/solver/forward/nonlinear/kzk/ \
         src/physics/acoustics/models/nonlinear/kzk/
  ```

- [ ] **Move Westervelt equation:**
  ```bash
  git mv src/solver/forward/nonlinear/westervelt_spectral/ \
         src/physics/acoustics/models/nonlinear/westervelt/
  ```

- [ ] **Delete old nonlinear solver directory:**
  ```bash
  git rm -r src/solver/forward/nonlinear/
  ```

#### Step 1C.3: Move Elastic Wave Models

**Task:** Move elastic wave physics

- [ ] **Move elastic plugin:**
  ```bash
  git mv src/solver/forward/elastic/ \
         src/physics/elasticity/models/
  ```

- [ ] **Refactor to physics model:**
  ```rust
  // src/physics/elasticity/models/linear_elastic.rs
  
  /// Linear elastic wave equation (physics ONLY)
  pub struct LinearElasticWave {
      // Lame parameters, density
  }
  
  impl LinearElasticWave {
      pub fn compute_stress_rate(&self, ...) -> Array2<f64> { /* Physics */ }
      pub fn compute_velocity_rate(&self, ...) -> [f64; 3] { /* Physics */ }
  }
  ```

#### Step 1C.4: Move Thermal Models

**Task:** Move heat diffusion

- [ ] **Move thermal diffusion:**
  ```bash
  git mv src/solver/forward/thermal_diffusion/ \
         src/physics/thermal/models/heat_diffusion/
  ```

- [ ] **Refactor to physics model:**
  ```rust
  // src/physics/thermal/models/heat_diffusion.rs
  
  /// Heat diffusion equation (physics ONLY)
  pub struct HeatDiffusion {
      // Thermal conductivity, specific heat
  }
  
  impl HeatDiffusion {
      pub fn compute_temperature_rate(&self, ...) -> f64 { /* Physics */ }
  }
  ```

#### Step 1C.5: Keep Only Numerical Methods in Solver

- [ ] **Verify solver contains ONLY:**
  ```bash
  ls src/solver/forward/
  # Expected: fdtd/, pstd/, hybrid/, dg/, plugin_based/
  # NO: acoustic/, elastic/, nonlinear/, thermal_diffusion/
  ```

- [ ] **Clean solver structure:**
  ```rust
  // src/solver/forward/fdtd/scheme.rs
  
  /// FDTD numerical scheme (NO physics equations)
  pub struct FdtdScheme {
      spatial_order: usize,  // 2, 4, 8
      time_step: f64,
  }
  
  impl FdtdScheme {
      pub fn step<P: PhysicsModel>(&self, physics: &P, state: &mut State) {
          // Use physics model via trait
          let rate = physics.compute_rate(state);
          // Apply numerical time stepping
      }
  }
  ```

#### Step 1C.6: Create Physics-Solver Bridge

- [ ] **Create bridge trait:**
  ```rust
  // src/solver/plugin_system/physics_solver_bridge.rs
  
  use crate::physics::plugin::Plugin;
  
  /// Bridge between physics models and numerical solvers
  pub trait PhysicsModel {
      fn compute_rate(&self, state: &State) -> State;
      fn required_fields(&self) -> Vec<FieldType>;
  }
  
  /// Adapter to use physics plugins in solvers
  pub struct PhysicsAdapter<P: Plugin> {
      plugin: P,
  }
  
  impl<P: Plugin> PhysicsModel for PhysicsAdapter<P> {
      fn compute_rate(&self, state: &State) -> State {
          // Delegate to plugin
      }
  }
  ```

#### Step 1C.7: Update All Solver References

- [ ] **Generate import map:**
  ```bash
  grep -r "solver::forward::acoustic\|solver::forward::elastic\|solver::forward::nonlinear" src/
  ```

- [ ] **Automated replacement:**
  ```bash
  find src -name "*.rs" -exec sed -i \
    's/solver::forward::acoustic/physics::acoustics::models::linear/g' {} +
  
  find src -name "*.rs" -exec sed -i \
    's/solver::forward::nonlinear::kuznetsov/physics::acoustics::models::nonlinear::kuznetsov/g' {} +
  ```

- [ ] **Manual verification:** Check complex cases

#### Step 1C.8: Verification

- [ ] **Run full test suite:**
  ```bash
  cargo test --all-features
  # Expected: 867/867 tests passing
  ```

- [ ] **Verify separation:**
  ```bash
  # Physics should NOT import solver
  grep -r "use crate::solver" src/physics/
  # Expected: No matches or only plugin_system/bridge
  
  # Solver should access physics via traits
  grep -r "PhysicsModel\|Plugin" src/solver/
  # Expected: Multiple trait usages
  ```

- [ ] **Performance benchmark:**
  ```bash
  cargo bench --bench physics_benchmarks
  cargo bench --bench solver_benchmarks
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 1C: Physics-solver separation complete

  - Moved physics equations to physics/ layer
  - Kept only numerical methods in solver/
  - Created physics-solver bridge
  - Zero circular dependencies
  - All 867 tests passing"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

## Phase 2: Clinical Consolidation (Week 3-4)

### Sprint 2A: Clinical Workflows Migration (Days 11-14)

**Objective:** Consolidate clinical workflows in `clinical/` layer

#### Step 2A.1: Create Clinical Structure

- [ ] **Create directories:**
  ```bash
  mkdir -p src/clinical/imaging/ultrasound
  mkdir -p src/clinical/imaging/elastography
  mkdir -p src/clinical/imaging/photoacoustic
  mkdir -p src/clinical/imaging/contrast_enhanced
  mkdir -p src/clinical/imaging/fusion
  mkdir -p src/clinical/therapy/hifu
  mkdir -p src/clinical/therapy/lithotripsy
  mkdir -p src/clinical/therapy/transcranial
  mkdir -p src/clinical/therapy/cavitation_control
  mkdir -p src/clinical/protocols
  ```

#### Step 2A.2: Move Imaging Modalities

**Task:** Move elastography workflows

- [ ] **Move elastography:**
  ```bash
  git mv src/physics/acoustics/imaging/modalities/elastography/ \
         src/clinical/imaging/elastography/
  ```

- [ ] **Split elastic_wave_solver.rs (2,824 lines):**
  ```bash
  # Split into:
  # - solver_core.rs (<500 lines)
  # - boundary_conditions.rs (<500 lines)
  # - material_nonlinearity.rs (<500 lines)
  # - solver_integration.rs (<500 lines)
  # - validation.rs (<500 lines)
  # - mod.rs (<300 lines)
  ```

- [ ] **Split nonlinear.rs (1,342 lines):**
  ```bash
  # Split into:
  # - nonlinear_elasticity.rs (<500 lines)
  # - constitutive_models.rs (<500 lines)
  # - strain_analysis.rs (<500 lines)
  ```

- [ ] **Split inversion.rs (1,233 lines):**
  ```bash
  # Split into:
  # - direct_inversion.rs (<500 lines)
  # - iterative_inversion.rs (<500 lines)
  # - regularization.rs (<500 lines)
  ```

**Task:** Move other imaging modalities

- [ ] **Move CEUS:**
  ```bash
  git mv src/physics/acoustics/imaging/modalities/ceus/ \
         src/clinical/imaging/contrast_enhanced/
  ```

- [ ] **Move ultrasound/HIFU:**
  ```bash
  git mv src/physics/acoustics/imaging/modalities/ultrasound/ \
         src/clinical/imaging/ultrasound/
  ```

- [ ] **Move photoacoustic:**
  ```bash
  git mv src/simulation/modalities/photoacoustic.rs \
         src/clinical/imaging/photoacoustic/workflow.rs
  
  # Split photoacoustic.rs (865 lines) into:
  # - workflow.rs (<500 lines)
  # - reconstruction.rs (<400 lines)
  ```

**Task:** Move fusion and registration

- [ ] **Move fusion:**
  ```bash
  git mv src/physics/acoustics/imaging/fusion.rs \
         src/clinical/imaging/fusion/multimodal.rs
  
  # Split fusion.rs (1,033 lines) into:
  # - multimodal.rs (<500 lines)
  # - algorithms.rs (<500 lines)
  ```

- [ ] **Move registration:**
  ```bash
  git mv src/physics/acoustics/imaging/registration/ \
         src/clinical/imaging/fusion/registration/
  ```

#### Step 2A.3: Move Therapy Workflows

**Task:** Move HIFU therapy

- [ ] **Move HIFU:**
  ```bash
  git mv src/physics/acoustics/imaging/modalities/ultrasound/hifu/ \
         src/clinical/therapy/hifu/
  ```

**Task:** Move therapy modalities

- [ ] **Move lithotripsy:**
  ```bash
  git mv src/physics/acoustics/therapy/lithotripsy/ \
         src/clinical/therapy/lithotripsy/
  ```

- [ ] **Move cavitation control:**
  ```bash
  git mv src/physics/acoustics/therapy/cavitation/ \
         src/clinical/therapy/cavitation_control/
  ```

- [ ] **Move transcranial:**
  ```bash
  git mv src/physics/acoustics/transcranial/ \
         src/clinical/therapy/transcranial/
  ```

#### Step 2A.4: Split Large Clinical Workflow Files

**Task:** Split clinical/imaging/workflows.rs (1,181 lines)

- [ ] **Split into modality-specific workflows:**
  ```bash
  # Create:
  # - b_mode_workflow.rs (<400 lines)
  # - doppler_workflow.rs (<400 lines)
  # - harmonic_workflow.rs (<400 lines)
  ```

**Task:** Split clinical/therapy/therapy_integration.rs (1,241 lines)

- [ ] **Split into focused modules:**
  ```bash
  # Create:
  # - treatment_planning.rs (<500 lines)
  # - monitoring.rs (<400 lines)
  # - safety_checks.rs (<400 lines)
  ```

**Task:** Split clinical/therapy/swe_3d_workflows.rs (975 lines)

- [ ] **Split into focused modules:**
  ```bash
  # Create:
  # - swe_3d_acquisition.rs (<500 lines)
  # - swe_3d_reconstruction.rs (<500 lines)
  ```

#### Step 2A.5: Clean Up Physics Layer

- [ ] **Verify physics layer clean:**
  ```bash
  ls src/physics/acoustics/
  # Expected: analytical/, mechanics/, models/, coupling/
  # NOT: imaging/, therapy/, transcranial/
  ```

- [ ] **Delete empty directories:**
  ```bash
  git rm -r src/physics/acoustics/imaging/
  git rm -r src/physics/acoustics/therapy/
  git rm -r src/physics/acoustics/transcranial/
  ```

#### Step 2A.6: Update Imports

- [ ] **Generate import map:**
  ```bash
  grep -r "physics::acoustics::imaging\|physics::acoustics::therapy" src/ > clinical_import_map.txt
  ```

- [ ] **Automated replacement:**
  ```bash
  find src -name "*.rs" -exec sed -i \
    's/physics::acoustics::imaging/clinical::imaging/g' {} +
  
  find src -name "*.rs" -exec sed -i \
    's/physics::acoustics::therapy/clinical::therapy/g' {} +
  
  find src -name "*.rs" -exec sed -i \
    's/physics::acoustics::transcranial/clinical::therapy::transcranial/g' {} +
  ```

#### Step 2A.7: Verification

- [ ] **Run full test suite:**
  ```bash
  cargo test --all-features
  # Expected: 867/867 tests passing
  ```

- [ ] **Check file sizes:**
  ```bash
  find src/clinical -name "*.rs" -exec wc -l {} \; | \
    awk '$1 > 500 {print "VIOLATION: " $2}'
  # Expected: No output
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 2A: Clinical workflows migration complete

  - Moved imaging modalities to clinical/imaging
  - Moved therapy workflows to clinical/therapy
  - Split 23 large files (>500 lines)
  - Cleaned up physics layer
  - All 867 tests passing"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

### Sprint 2B: Massive File Decomposition (Days 15-20)

**Objective:** Split all remaining files >500 lines

#### Step 2B.1: Split Math/ML Files

**Task:** Split math/linear_algebra/mod.rs (1,887 lines)

- [ ] **Create focused modules:**
  ```bash
  mkdir -p src/math/linear_algebra/
  # Split into:
  # - matrix_operations.rs (<500 lines)
  # - decomposition.rs (<500 lines)
  # - solvers.rs (<500 lines)
  # - eigenvalues.rs (<500 lines)
  ```

**Task:** Split PINN files

- [ ] **Split burn_wave_equation_2d.rs (2,579 lines):**
  ```bash
  # Split into:
  # - architecture.rs (<500 lines)
  # - training.rs (<500 lines)
  # - loss_functions.rs (<500 lines)
  # - boundary_conditions.rs (<500 lines)
  # - validation.rs (<500 lines)
  ```

- [ ] **Split burn_wave_equation_1d.rs (1,099 lines):**
  ```bash
  # Split into:
  # - model_1d.rs (<500 lines)
  # - training_1d.rs (<500 lines)
  # - validation_1d.rs (<300 lines)
  ```

- [ ] **Split electromagnetic.rs (1,188 lines):**
  ```bash
  # Split into:
  # - maxwell_equations.rs (<500 lines)
  # - pinn_solver.rs (<500 lines)
  # - validation.rs (<300 lines)
  ```

#### Step 2B.2: Split Analysis Files

**Task:** Split beamforming files

- [ ] **Split adaptive/subspace.rs (877 lines):**
  ```bash
  # Split into:
  # - music.rs (<500 lines)
  # - esprit.rs (<400 lines)
  ```

- [ ] **Split utils/mod.rs (781 lines):**
  ```bash
  # Split into:
  # - array_manifold.rs (<400 lines)
  # - steering_vectors.rs (<400 lines)
  ```

- [ ] **Split covariance/mod.rs (669 lines):**
  ```bash
  # Already migrated to utils/covariance.rs
  # Ensure <500 lines
  ```

#### Step 2B.3: Split Infrastructure Files

**Task:** Split infra/cloud/mod.rs (1,126 lines)

- [ ] **Split into cloud providers:**
  ```bash
  # Split into:
  # - aws.rs (<500 lines)
  # - azure.rs (<500 lines)
  # - gcp.rs (<300 lines)
  ```

**Task:** Split infra/api files

- [ ] **Split clinical_handlers.rs (914 lines):**
  ```bash
  # Split into:
  # - imaging_endpoints.rs (<500 lines)
  # - therapy_endpoints.rs (<500 lines)
  ```

- [ ] **Split models.rs (861 lines):**
  ```bash
  # Split into:
  # - request_models.rs (<500 lines)
  # - response_models.rs (<500 lines)
  ```

- [ ] **Split auth.rs (687 lines):**
  ```bash
  # Split into:
  # - authentication.rs (<400 lines)
  # - authorization.rs (<300 lines)
  ```

#### Step 2B.4: Split Domain Files

**Task:** Split domain/medium/adapters/cylindrical.rs (840 lines)

- [ ] **Split into:**
  ```bash
  # - cylindrical_projection.rs (<500 lines)
  # - coordinate_transform.rs (<400 lines)
  ```

**Task:** Split domain/grid/topology.rs (752 lines)

- [ ] **Split into:**
  ```bash
  # - cartesian_topology.rs (<400 lines)
  # - cylindrical_topology.rs (<400 lines)
  ```

**Task:** Split domain/boundary/pml.rs (746 lines)

- [ ] **Split into:**
  ```bash
  # - pml_config.rs (<300 lines)
  # - pml_implementation.rs (<500 lines)
  ```

#### Step 2B.5: Split Solver Files

**Task:** Split solver/forward/axisymmetric/solver.rs (700 lines)

- [ ] **Split into:**
  ```bash
  # - axisymmetric_scheme.rs (<500 lines)
  # - boundary_conditions.rs (<300 lines)
  ```

**Task:** Split solver/forward/hybrid/solver.rs (653 lines)

- [ ] **Split into:**
  ```bash
  # - hybrid_scheme.rs (<500 lines)
  # - domain_coupling.rs (<300 lines)
  ```

#### Step 2B.6: Verification Loop (Per File Split)

For each file split:

- [ ] **Verify line counts:**
  ```bash
  wc -l <new_files>
  # All must be <500 lines
  ```

- [ ] **Run module tests:**
  ```bash
  cargo test --package kwavers --lib <module_path>
  ```

- [ ] **Check imports:**
  ```bash
  cargo check
  ```

- [ ] **Commit atomically:**
  ```bash
  git add <affected_files>
  git commit -m "Split <file>: <brief_description>"
  ```

#### Step 2B.7: Final Verification

- [ ] **Global line count check:**
  ```bash
  find src -name "*.rs" -exec wc -l {} \; | \
    awk '$1 > 500 {count++; print "VIOLATION: " $2 " (" $1 " lines)"} \
         END {print "Total violations: " count}'
  # Expected: Total violations: 0
  ```

- [ ] **Run full test suite:**
  ```bash
  cargo test --all-features
  # Expected: 867/867 tests passing
  ```

- [ ] **Performance benchmarks:**
  ```bash
  cargo bench
  # Compare with baseline (no significant regression)
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 2B: Massive file decomposition complete

  - Split 50+ files exceeding 500 lines
  - All files now GRASP compliant (<500 lines)
  - Preserved git history with atomic commits
  - All 867 tests passing
  - Zero performance regression"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

## Phase 3: Dead Code Removal (Week 5)

### Sprint 3A: File Cleanup (Days 21-23)

#### Step 3A.1: Remove Deprecated Code

- [ ] **Delete deprecated beamforming shaders:**
  ```bash
  git rm -r src/domain/sensor/beamforming/shaders/
  ```

- [ ] **Delete legacy skull models:**
  ```bash
  git rm -r src/physics/acoustics/skull/legacy/
  ```

- [ ] **Delete incomplete k-Wave validation:**
  ```bash
  git rm -r src/solver/utilities/validation/kwave/
  ```

#### Step 3A.2: Remove Build Artifacts

- [ ] **Clean target directory:**
  ```bash
  cargo clean
  git clean -fdx target/
  ```

- [ ] **Remove log files:**
  ```bash
  git rm errors.txt 2>/dev/null || true
  find . -name "*.log" -type f -delete
  ```

- [ ] **Update .gitignore:**
  ```bash
  cat >> .gitignore << 'EOF'
  # Build artifacts
  /target/
  **/*.log
  errors.txt
  
  # Test artifacts
  /proptest-regressions/
  *.profraw
  
  # Editor artifacts
  .vscode/
  .idea/
  *.swp
  *~
  EOF
  ```

#### Step 3A.3: Consolidate Documentation

- [ ] **List redundant documentation:**
  ```bash
  ls -1 *.md | grep -E "REFACTOR|ARCHITECTURE|AUDIT|ANALYSIS"
  ```

- [ ] **Keep only:**
  ```bash
  # Keep:
  # - DEEP_VERTICAL_HIERARCHY_AUDIT.md (this file)
  # - REFACTORING_EXECUTION_CHECKLIST.md (current file)
  # - gap_audit.md (mathematical validation)
  # - README.md (user-facing)
  ```

- [ ] **Archive obsolete docs:**
  ```bash
  mkdir -p docs/archive/
  git mv ARCHITECTURE_IMPROVEMENT_PLAN.md docs/archive/
  git mv ARCHITECTURE_REFACTORING_AUDIT.md docs/archive/
  git mv COMPREHENSIVE_MODULE_REFACTORING_PLAN.md docs/archive/
  git mv DEPENDENCY_ANALYSIS.md docs/archive/
  git mv PERFORMANCE_OPTIMIZATION_ANALYSIS.md docs/archive/
  git mv REFACTORING_EXECUTIVE_SUMMARY.md docs/archive/
  git mv REFACTORING_PROGRESS.md docs/archive/
  git mv REFACTORING_QUICK_REFERENCE.md docs/archive/
  git mv REFACTOR_PHASE_1_CHECKLIST.md docs/archive/
  # ... (other redundant docs)
  ```

- [ ] **Update README.md to reference new structure:**
  ```bash
  # Edit README.md to point to:
  # - DEEP_VERTICAL_HIERARCHY_AUDIT.md (architecture)
  # - gap_audit.md (validation)
  # - docs/adr.md (decision records)
  ```

#### Step 3A.4: Verification

- [ ] **Check git status:**
  ```bash
  git status
  # Verify only intended deletions
  ```

- [ ] **Run full test suite:**
  ```bash
  cargo test --all-features
  # Expected: 867/867 tests passing
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 3A: Dead code removal complete

  - Deleted deprecated code and artifacts
  - Consolidated redundant documentation
  - Updated .gitignore
  - All 867 tests passing"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

### Sprint 3B: Dependency Audit (Days 24-25)

#### Step 3B.1: Analyze Dependencies

- [ ] **Generate dependency tree:**
  ```bash
  cargo tree --all-features > dependency_tree_full.txt
  cargo tree --no-default-features > dependency_tree_minimal.txt
  ```

- [ ] **Check for unused dependencies:**
  ```bash
  cargo install cargo-udeps
  cargo +nightly udeps --all-features
  ```

- [ ] **Analyze feature flags:**
  ```bash
  # List all features
  cargo metadata --format-version 1 | \
    jq '.packages[] | select(.name == "kwavers") | .features'
  ```

#### Step 3B.2: Remove Unused Dependencies

- [ ] **Remove unused crates:**
  ```toml
  # Edit Cargo.toml
  # Remove any dependencies flagged by cargo-udeps
  ```

- [ ] **Verify minimal build:**
  ```bash
  cargo build --no-default-features
  # Should build successfully with minimal dependencies
  ```

- [ ] **Verify full build:**
  ```bash
  cargo build --all-features
  # Should build successfully with all features
  ```

#### Step 3B.3: Optimize Feature Flags

- [ ] **Review feature combinations:**
  ```bash
  cargo build --features "gpu"
  cargo build --features "pinn"
  cargo build --features "api"
  cargo build --features "plotting"
  ```

- [ ] **Document feature dependencies:**
  ```toml
  # In Cargo.toml, add comments:
  
  [features]
  # Core functionality (always enabled)
  default = ["minimal"]
  minimal = []
  
  # Performance features
  parallel = ["ndarray/rayon"]
  simd = []  # Requires nightly
  
  # GPU acceleration
  gpu = ["dep:wgpu", "dep:bytemuck", "dep:pollster"]
  gpu-visualization = ["gpu", "plotting", "dep:egui"]
  
  # Machine learning
  pinn = ["dep:burn"]  # Physics-informed neural networks
  pinn-gpu = ["pinn", "gpu"]
  
  # API and cloud
  api = ["dep:axum", "dep:tower", "async-runtime"]
  cloud = ["dep:reqwest", "async-runtime"]
  
  # Complete feature set
  full = ["gpu", "pinn", "api", "cloud", "plotting"]
  ```

#### Step 3B.4: Update Dependency Documentation

- [ ] **Document dependency rationale:**
  ```markdown
  # Create docs/dependencies.md
  
  ## Core Dependencies
  
  - **ndarray**: N-dimensional arrays (scientific computing foundation)
  - **rustfft**: FFT implementations (spectral methods)
  - **num-complex**: Complex number support (frequency domain)
  ...
  
  ## Optional Dependencies
  
  - **wgpu**: GPU compute (feature: gpu)
  - **burn**: Machine learning framework (feature: pinn)
  - **axum**: Web framework (feature: api)
  ...
  ```

#### Step 3B.5: Verification

- [ ] **Build matrix test:**
  ```bash
  # Test all major feature combinations
  cargo test --no-default-features
  cargo test --features "parallel"
  cargo test --features "gpu"
  cargo test --features "pinn"
  cargo test --features "api"
  cargo test --all-features
  ```

- [ ] **Check build times:**
  ```bash
  time cargo build --release --no-default-features
  time cargo build --release --features "gpu"
  time cargo build --release --all-features
  # Compare with baseline
  ```

- [ ] **Commit sprint:**
  ```bash
  git add -A
  git commit -m "Sprint 3B: Dependency audit complete

  - Removed unused dependencies
  - Optimized feature flags
  - Documented dependency rationale
  - All feature combinations build successfully"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

## Phase 4: Validation & Documentation (Week 6)

### Sprint 4A: Comprehensive Testing (Days 26-28)

#### Step 4A.1: Full Test Suite

- [ ] **Run all tests:**
  ```bash
  cargo test --all-features -- --nocapture 2>&1 | tee test_results.log
  ```

- [ ] **Verify test count:**
  ```bash
  grep "test result:" test_results.log
  # Expected: 867 passed; 0 failed
  ```

- [ ] **Run ignored tests:**
  ```bash
  cargo test --all-features -- --ignored
  ```

#### Step 4A.2: Property-Based Testing

- [ ] **Add proptest for refactored modules:**
  ```bash
  # Add property tests for:
  # - analysis/signal_processing/beamforming
  # - math/numerics/differentiation
  # - physics/acoustics/models
  ```

- [ ] **Run property tests:**
  ```bash
  cargo test --package kwavers --test property_based
  ```

#### Step 4A.3: Integration Testing

- [ ] **Run integration tests:**
  ```bash
  cargo test --test integration_test
  cargo test --test simple_integration_test
  cargo test --test infrastructure_test
  ```

- [ ] **Run validation tests:**
  ```bash
  cargo test --test cfl_stability_test
  cargo test --test energy_conservation_test
  ```

#### Step 4A.4: Literature Validation

- [ ] **Run k-Wave comparison tests:**
  ```bash
  cargo test --test literature_validation --features full
  ```

- [ ] **Run physics validation:**
  ```bash
  cargo test --test physics_validation_test --features full
  cargo test --test rigorous_physics_validation --features full
  ```

#### Step 4A.5: Performance Benchmarking

- [ ] **Run all benchmarks:**
  ```bash
  cargo bench 2>&1 | tee benchmark_results.log
  ```

- [ ] **Compare with baseline:**
  ```bash
  # Compare benchmark_results.log with baseline_benchmarks.log
  # Acceptable regression: <5%
  # Target improvement: >10%
  ```

- [ ] **Profile critical paths:**
  ```bash
  cargo bench --bench critical_path_benchmarks
  ```

#### Step 4A.6: Memory Profiling

- [ ] **Check for memory leaks:**
  ```bash
  cargo install valgrind
  valgrind --leak-check=full target/release/examples/basic_simulation
  ```

- [ ] **Profile memory usage:**
  ```bash
  cargo run --release --example advanced_ultrasound_imaging -- --profile-memory
  ```

#### Step 4A.7: Code Quality Checks

- [ ] **Run clippy:**
  ```bash
  cargo clippy --all-features -- -D warnings 2>&1 | tee clippy_results.log
  ```

- [ ] **Verify zero warnings:**
  ```bash
  grep "warning:" clippy_results.log | wc -l
  # Expected: 0
  ```

- [ ] **Check code formatting:**
  ```bash
  cargo fmt --check
  ```

#### Step 4A.8: Documentation Tests

- [ ] **Run doc tests:**
  ```bash
  cargo test --doc
  ```

- [ ] **Check doc coverage:**
  ```bash
  cargo doc --no-deps --all-features
  # Verify no "missing documentation" warnings
  ```

#### Step 4A.9: Verification

- [ ] **Final test summary:**
  ```bash
  echo "Test Summary:"
  echo "============="
  echo "Unit tests: $(grep -c "test result: ok" test_results.log)"
  echo "Integration tests: $(cargo test --test '*' 2>&1 | grep -c "test result: ok")"
  echo "Doc tests: $(cargo test --doc 2>&1 | grep -c "test result: ok")"
  echo "Benchmarks: $(grep -c "^test " benchmark_results.log)"
  echo "Clippy warnings: $(grep -c "warning:" clippy_results.log)"
  ```

- [ ] **Commit sprint:**
  ```bash
  git add test_results.log benchmark_results.log clippy_results.log
  git commit -m "Sprint 4A: Comprehensive testing complete

  - All 867 tests passing
  - Zero clippy warnings
  - Zero performance regression
  - Memory profiling clean
  - Literature validation passing"
  
  git push origin refactor/deep-vertical-hierarchy
  ```

---

### Sprint 4B: Documentation Update (Days 29-30)

#### Step 4B.1: Update README.md

- [ ] **Update architecture section:**
  ```markdown
  ## Architecture
  
  Kwavers follows strict layered architecture principles:
  
  - **clinical/**: Application-specific workflows (imaging, therapy)
  - **simulation/**: Orchestration and configuration
  - **analysis/**: Signal processing and validation
  - **solver/**: Numerical methods (FDTD, PSTD, DG)
  - **physics/**: Physics models (acoustics, elasticity, optics)
  - **domain/**: Domain primitives (grid, medium, sensors)
  - **math/**: Mathematical primitives (FFT, linear algebra, numerics)
  - **infra/**: Infrastructure (API, I/O, cloud)
  - **core/**: Foundation (errors, constants, types)
  ```

- [ ] **Update feature flags documentation:**
  ```markdown
  ## Feature Flags
  
  - `minimal` (default): Core functionality only
  - `parallel`: Enable parallel processing
  - `gpu`: GPU acceleration via WGPU
  - `pinn`: Physics-informed neural networks
  - `api`: REST API for