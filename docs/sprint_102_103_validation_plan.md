# Sprint 102-103: k-Wave Validation & Documentation Plan
## Comprehensive Testing Strategy for Feature Parity Verification

**Status**: READY FOR EXECUTION  
**Priority**: P0 - CRITICAL  
**Estimated Effort**: 2-3 micro-sprints (2-3 hours total)  
**Analyst**: Senior Rust Engineer

---

## Executive Summary

**Objective**: Establish numerical parity with k-Wave MATLAB through comprehensive validation testing and complete documentation with literature citations.

**Current State**: Feature implementation **COMPLETE** (38 k-space files, 13 absorption files, 32 transducer files), validation **NEEDED**.

**Success Criteria**:
- Numerical accuracy <1% error vs k-Wave MATLAB benchmarks
- 10+ standard k-Wave test cases implemented and passing
- 100% documentation coverage with literature citations
- Performance benchmarks published

---

## Sprint 102: k-Wave Validation Test Suite (P0)

### Objective
Create comprehensive test suite comparing Kwavers implementations against k-Wave MATLAB reference results.

### Deliverables

#### 1. Test Infrastructure Setup
```bash
# Create validation directory structure
tests/kwave_validation/
├── mod.rs                    # Main test module
├── reference_data/          # k-Wave MATLAB reference results
│   ├── power_law_absorption.json
│   ├── dispersion_correction.json
│   ├── k_space_gradient.json
│   └── time_reversal.json
├── test_kspace_operators.rs # k-space operator validation
├── test_absorption.rs       # Absorption model validation
├── test_reconstruction.rs   # Reconstruction algorithm validation
└── test_performance.rs      # Performance benchmarking
```

#### 2. Standard Test Cases (Priority Order)

**Test Case 1: Power-Law Absorption**
```rust
#[test]
fn test_power_law_absorption_vs_kwave() {
    // Test α(ω) = α₀|ω|^y with y ∈ [0.5, 1.0, 1.5, 2.0]
    // Reference: k-Wave example_us_defining_transducer
    // Expected: <0.5% error in attenuation coefficient
}
```

**Test Case 2: k-Space Gradient**
```rust
#[test]
fn test_kspace_gradient_accuracy() {
    // Test ∇p in k-space vs finite difference
    // Reference: k-Wave makeGrid, kspaceFirstOrder2D
    // Expected: <1% error for smooth fields
}
```

**Test Case 3: Dispersion Correction**
```rust
#[test]
fn test_dispersion_correction_causal() {
    // Test causal absorption dispersion correction
    // Reference: Treeby & Cox (2010) Eq. 12
    // Expected: Phase velocity correction <0.1%
}
```

**Test Case 4: Time Reversal Reconstruction**
```rust
#[test]
fn test_time_reversal_photoacoustic() {
    // Test photoacoustic reconstruction accuracy
    // Reference: k-Wave example_pr_2D_TR_circular_sensor
    // Expected: <5% RMS error in reconstructed image
}
```

**Test Case 5: Multi-Relaxation Absorption**
```rust
#[test]
fn test_multi_relaxation_absorption() {
    // Test multi-relaxation absorption model
    // Reference: Szabo (1995) IEEE Trans. UFFC
    // Expected: Frequency response match <2%
}
```

**Test Case 6: Transducer Field Calculation**
```rust
#[test]
fn test_transducer_spatial_impulse_response() {
    // Test SIR calculation vs Tupholme-Stepanishen
    // Reference: Jensen & Svendsen (1992)
    // Expected: Pressure field match <3%
}
```

**Test Case 7: Beamforming Algorithms**
```rust
#[test]
fn test_delay_and_sum_beamforming() {
    // Test DAS beamforming vs k-Wave
    // Reference: Van Veen & Buckley (1988)
    // Expected: Beamformed image PSNR >30dB
}
```

**Test Case 8: Heterogeneous Medium Propagation**
```rust
#[test]
fn test_heterogeneous_medium_kwave_parity() {
    // Test wave propagation in layered media
    // Reference: k-Wave example_sd_homogeneous_medium
    // Expected: Reflection/transmission coefficients <2% error
}
```

**Test Case 9: PML Boundary Conditions**
```rust
#[test]
fn test_pml_absorption_performance() {
    // Test CPML reflection coefficient
    // Reference: Roden & Gedney (2000)
    // Expected: Reflection coefficient <-40dB
}
```

**Test Case 10: Nonlinear Propagation**
```rust
#[test]
fn test_westervelt_equation_kwave() {
    // Test nonlinear wave steepening
    // Reference: Hamilton & Blackstock (1998)
    // Expected: Harmonic generation match <5%
}
```

#### 3. Reference Data Generation

**MATLAB Script** (to run in k-Wave environment):
```matlab
% generate_reference_data.m
% Generate reference results for Kwavers validation

% Test Case 1: Power-law absorption
alpha_coeff = 0.75;  % dB/(MHz^y cm)
alpha_power = 1.5;
[tau, eta] = compute_absorption_operators(grid, alpha_coeff, alpha_power);
save('reference_data/power_law_absorption.mat', 'tau', 'eta');

% ... (similar for all test cases)
```

**Conversion to JSON** (for Rust consumption):
```python
# convert_mat_to_json.py
import scipy.io
import json
import numpy as np

def convert_mat_to_json(mat_file, json_file):
    data = scipy.io.loadmat(mat_file)
    # Convert numpy arrays to lists
    json_data = {k: v.tolist() for k, v in data.items() 
                 if not k.startswith('__')}
    with open(json_file, 'w') as f:
        json.dump(json_data, f)

convert_mat_to_json('power_law_absorption.mat', 
                    'power_law_absorption.json')
```

#### 4. Performance Benchmarking

**Benchmark Suite** (`tests/kwave_validation/test_performance.rs`):
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_kspace_operator(c: &mut Criterion) {
    let mut group = c.benchmark_group("k-space operators");
    
    // Small grid (64³)
    group.bench_function("kspace_gradient_64", |b| {
        b.iter(|| {
            let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
            let op = KSpaceOperator::new(...);
            black_box(op.apply_gradient(&field))
        });
    });
    
    // Large grid (256³)
    group.bench_function("kspace_gradient_256", |b| {
        // ...
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_kspace_operator);
criterion_main!(benches);
```

**Performance Targets**:
- Small grids (64³): <10ms per operation
- Medium grids (128³): <100ms per operation
- Large grids (256³): <1s per operation
- GPU acceleration: 2-5x speedup vs CPU

---

## Sprint 103: Documentation Enhancement (P0)

### Objective
Complete literature-validated documentation with inline LaTeX equations and comprehensive citations.

### Deliverables

#### 1. k-Space Operator Documentation

**File**: `src/solver/kspace_pseudospectral.rs`

**Enhancements**:
```rust
//! # k-Space Pseudospectral Method
//!
//! ## Mathematical Foundation
//!
//! The k-space pseudospectral method solves the linearized wave equation with
//! power-law absorption:
//!
//! ```text
//! ∂p/∂t + c₀·∇·u + α(ω)*p = S
//! ∂u/∂t + ∇p/ρ₀ = 0
//! ```
//!
//! where the absorption operator is defined as:
//!
//! ```text
//! α(ω) = α₀ · |ω|^y
//! ```
//!
//! with y ∈ [0, 3] controlling the frequency dependence.
//!
//! ## Dispersion Correction
//!
//! For causal absorption (y ≠ 2), dispersion correction is required [1]:
//!
//! ```text
//! c_eff(ω) = c₀ · [1 + (α₀·c₀^(y-1) / 2) · |ω|^(y-1) · tan(π·y/2)]
//! ```
//!
//! ## Implementation Details
//!
//! The method uses FFT to transform to k-space where spatial derivatives
//! become multiplications by wavenumbers:
//!
//! ```text
//! ℱ{∂p/∂x} = i·kₓ·ℱ{p}
//! ℱ{∇²p} = -(kₓ² + kᵧ² + k_z²)·ℱ{p}
//! ```
//!
//! # References
//!
//! [1] Treeby, B. E., & Cox, B. T. (2010). "Modeling power law absorption and
//!     dispersion for acoustic propagation using the fractional Laplacian."
//!     The Journal of the Acoustical Society of America, 127(5), 2741-2748.
//!     DOI: 10.1121/1.3377056
//!
//! [2] Treeby, B. E., Jaros, J., Rendell, A. P., & Cox, B. T. (2012).
//!     "Modeling nonlinear ultrasound propagation in heterogeneous media with
//!     power law absorption using a k-space pseudospectral method."
//!     The Journal of the Acoustical Society of America, 131(6), 4324-4336.
//!     DOI: 10.1121/1.4712021
```

#### 2. Absorption Model Documentation

**File**: `src/solver/kwave_parity/absorption.rs`

**Enhancements**:
```rust
//! # Power-Law Absorption Models
//!
//! ## Physical Basis
//!
//! Biological tissues exhibit frequency-dependent absorption following an
//! empirical power law [1]:
//!
//! ```text
//! α(f) = α₀ · f^y
//! ```
//!
//! where:
//! - α(f): absorption coefficient [dB/(MHz^y cm)]
//! - f: frequency [MHz]
//! - y: power law exponent (typically 1.0-1.5 for soft tissue)
//!
//! ## Multi-Relaxation Model
//!
//! The multi-relaxation model accounts for multiple relaxation processes [2]:
//!
//! ```text
//! α(ω) = Σᵢ wᵢ·ω²·τᵢ / (1 + ω²·τᵢ²)
//! ```
//!
//! # References
//!
//! [1] Szabo, T. L. (1995). "Time domain wave equations for lossy media
//!     obeying a frequency power law." The Journal of the Acoustical Society
//!     of America, 96(1), 491-500. DOI: 10.1121/1.410434
//!
//! [2] Treeby, B. E., & Cox, B. T. (2010). "Modeling power law absorption..."
//!     JASA 127(5), 2741-2748.
```

#### 3. User Migration Guide

**File**: `docs/migration_guide_kwave.md` (NEW)

**Structure**:
```markdown
# k-Wave to Kwavers Migration Guide

## Overview
This guide helps k-Wave MATLAB users transition to Kwavers Rust library.

## Quick Comparison

| Task | k-Wave MATLAB | Kwavers Rust |
|------|---------------|--------------|
| Grid creation | `kgrid = kWaveGrid(128, 1e-3, ...)` | `let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3)?;` |
| Medium setup | `medium.sound_speed = 1500` | `let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);` |
| Solver | `sensor_data = kspaceFirstOrder2D(...)` | `let mut solver = KWaveSolver::new(grid, medium, config)?;` |

## Common Patterns

### Power-Law Absorption
**k-Wave MATLAB**:
```matlab
medium.alpha_coeff = 0.75;
medium.alpha_power = 1.5;
```

**Kwavers Rust**:
```rust
let config = KWaveConfig {
    absorption_mode: AbsorptionMode::PowerLaw {
        alpha_coeff: 0.75,
        alpha_power: 1.5,
    },
    ..Default::default()
};
```

### Transducer Modeling
**k-Wave MATLAB**:
```matlab
transducer = kWaveTransducer(kgrid, ...);
```

**Kwavers Rust**:
```rust
let geometry = TransducerGeometry {
    element_positions: positions,
    element_sizes: sizes,
    element_normals: normals,
    apodization: Some(apod_weights),
    delays: Some(delays),
};
```

## Performance Tips
1. Use GPU acceleration: Enable `gpu` feature
2. Optimize grid size: Powers of 2 for FFT efficiency
3. Parallel execution: Enable `rayon` for multi-threading

## Resources
- API documentation: [docs.rs/kwavers](https://docs.rs/kwavers)
- Examples: `examples/kwave_replication_suite_fixed.rs`
- Validation tests: `tests/kwave_validation/`
```

#### 4. API Documentation Completion

**Target**: 100% public API documentation with examples

**Checklist**:
- [ ] All public functions have rustdoc comments
- [ ] All modules have module-level documentation
- [ ] Mathematical formulas in LaTeX format
- [ ] Code examples for complex APIs
- [ ] Literature references with DOI links
- [ ] Performance characteristics documented
- [ ] Safety invariants for unsafe blocks

---

## Success Metrics

### Quantitative Targets

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Numerical Accuracy | <1% error | k-Wave benchmark comparison |
| Test Coverage | >95% | Tarpaulin analysis |
| Documentation Coverage | 100% | `cargo doc` + manual review |
| Performance | 2-5x k-Wave | Criterion benchmarks |
| Build Time | <60s | CI/CD tracking |

### Qualitative Assessment

- [ ] All 10 test cases passing with numerical parity
- [ ] Performance benchmarks published
- [ ] Documentation complete with literature citations
- [ ] Migration guide usable by k-Wave users
- [ ] API examples executable and correct

---

## Implementation Timeline

**Sprint 102** (1-hour micro-sprint):
- Set up test infrastructure (20 min)
- Implement test cases 1-5 (30 min)
- Initial validation run (10 min)

**Sprint 103** (1-hour micro-sprint):
- Complete test cases 6-10 (20 min)
- Performance benchmarking (20 min)
- Documentation enhancement (20 min)

**Sprint 103 continuation** (1-hour micro-sprint):
- Complete API documentation (30 min)
- Write migration guide (20 min)
- Final validation and report (10 min)

---

## Risks & Mitigation

### Risk 1: Reference Data Availability
- **Risk**: May not have access to k-Wave MATLAB
- **Mitigation**: Use published k-Wave paper results, k-wave-python equivalents
- **Contingency**: Contact k-Wave authors for reference datasets

### Risk 2: Numerical Precision
- **Risk**: <1% error target may be too strict for some test cases
- **Mitigation**: Use appropriate tolerances per physics domain
- **Fallback**: Document known precision limits with justification

### Risk 3: Performance Benchmarking
- **Risk**: Hardware variations affect benchmarks
- **Mitigation**: Normalize to problem size, report relative speedup
- **Standard**: Use consistent CI/CD hardware for reproducibility

---

## Completion Criteria

**Sprint 102-103 COMPLETE when**:
1. ✅ 10+ k-Wave validation tests passing
2. ✅ Numerical accuracy <1% documented
3. ✅ Performance benchmarks published
4. ✅ Documentation 100% complete with citations
5. ✅ Migration guide written and reviewed
6. ✅ All deliverables committed to repository

---

*Plan Version: 1.0*  
*Created: Sprint 101*  
*Status: READY FOR EXECUTION*  
*Estimated Effort: 2-3 hours total*
