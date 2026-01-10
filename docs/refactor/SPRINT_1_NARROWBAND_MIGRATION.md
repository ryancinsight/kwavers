# Sprint 1: Narrowband Beamforming Migration

**Sprint Goal:** Migrate narrowband beamforming algorithms from `domain::sensor::beamforming::narrowband` to `analysis::signal_processing::beamforming::narrowband` (SSOT)

**Status:** 🟡 Ready to Execute  
**Priority:** P0 — Blocking Deep Vertical Hierarchy Goals  
**Estimated Effort:** 12-16 hours  
**Sprint Duration:** 1 week  
**Owner:** Elite Mathematically-Verified Systems Architect  

---

## Sprint Objectives

### Primary Goals
1. ✅ **Complete SSOT for narrowband algorithms** — Migrate all narrowband beamforming from domain to analysis layer
2. ✅ **Zero code duplication** — Remove duplicate implementations after validation
3. ✅ **100% test coverage** — All migrated algorithms have comprehensive tests
4. ✅ **Zero regressions** — All existing tests pass, performance maintained

### Success Criteria
- [ ] All narrowband algorithms migrated to `analysis::signal_processing::beamforming::narrowband`
- [ ] Property-based tests validate mathematical equivalence
- [ ] Performance benchmarks show <5% change
- [ ] Zero uses of `domain::sensor::beamforming::narrowband` in internal code
- [ ] Full test suite passes (867 tests)
- [ ] `cargo clippy -- -D warnings` passes

---

## Architecture Context

### Current State (Incorrect)
```text
domain/sensor/beamforming/narrowband/     [DEPRECATED — 49 files, ~4k LOC]
├── capon.rs                              Narrowband Capon spatial spectrum
├── mod.rs                                Module exports
├── steering_narrowband.rs                Narrowband steering vectors
└── snapshots/                            Snapshot extraction utilities
    ├── mod.rs
    └── windowed/
        └── mod.rs

analysis/signal_processing/beamforming/narrowband/  [INCOMPLETE — Placeholder only]
└── mod.rs                                Placeholder + documentation
```

### Target State (Correct)
```text
analysis/signal_processing/beamforming/narrowband/  [CANONICAL SSOT]
├── mod.rs                                Public API + documentation
├── capon.rs                              Narrowband Capon spatial spectrum
├── steering.rs                           Narrowband steering vectors
└── snapshots/
    ├── mod.rs                            Snapshot extraction utilities
    ├── extraction.rs                     Core extraction logic
    └── windowing.rs                      Windowing functions

domain/sensor/beamforming/narrowband/     [TO BE REMOVED]
└── mod.rs                                Compatibility facade (deprecated re-exports)
```

### Layer Responsibilities

**Analysis Layer** (SSOT):
- ✅ Owns: Narrowband Capon algorithm
- ✅ Owns: Snapshot extraction from time-series data
- ✅ Owns: Narrowband steering vector calculation
- ✅ Owns: STFT-based frequency-domain conversion

**Domain Layer** (Hardware Primitives):
- ✅ Owns: Sensor array geometry (positions, orientations)
- ✅ Owns: Time-series data recording
- ❌ Does NOT own: Signal processing algorithms

---

## Task Breakdown

### Phase 1: Source Analysis & Preparation (2 hours)

#### Task 1.1: Inventory Source Files
- [ ] Read `domain/sensor/beamforming/narrowband/mod.rs` — identify all public exports
- [ ] Read `domain/sensor/beamforming/narrowband/capon.rs` — analyze Capon implementation
- [ ] Read `domain/sensor/beamforming/narrowband/steering_narrowband.rs` — analyze steering vectors
- [ ] Read `domain/sensor/beamforming/narrowband/snapshots/` — analyze snapshot utilities
- [ ] Document all public functions, types, and constants

**Deliverable:** `NARROWBAND_SOURCE_INVENTORY.md` with complete API surface

#### Task 1.2: Identify Dependencies
- [ ] Analyze imports in narrowband module (what does it depend on?)
- [ ] Identify circular dependencies (if any)
- [ ] Check for domain-specific coupling (sensor types, grid dependencies)
- [ ] Document migration order (bottom-up to avoid circular deps)

**Deliverable:** Dependency graph, migration order plan

#### Task 1.3: Analyze Consumer Usage
- [ ] `grep -r "domain::sensor::beamforming::narrowband" src/` — find all consumers
- [ ] Classify consumers by layer (analysis, domain, examples, tests)
- [ ] Identify critical consumers (blocking migration)
- [ ] Document migration priority (high-impact first)

**Deliverable:** Consumer analysis report, prioritized update list

---

### Phase 2: Algorithm Migration (6-8 hours)

#### Task 2.1: Migrate Capon Spatial Spectrum
**File:** `domain/sensor/beamforming/narrowband/capon.rs` → `analysis/.../narrowband/capon.rs`

**Steps:**
1. Create `src/analysis/signal_processing/beamforming/narrowband/capon.rs`
2. Copy implementation from domain layer
3. Remove domain-specific dependencies (replace with canonical utilities)
4. Update function signatures for consistency with analysis layer API
5. Add comprehensive Rustdoc with mathematical foundation
6. Validate against existing tests

**Mathematical Foundation:**
```text
Capon Spatial Spectrum (Minimum Variance):

P_MVDR(θ, φ) = 1 / (a^H R^-1 a)

where:
- a = steering vector for direction (θ, φ)
- R = sample covariance matrix
- ^H = Hermitian (conjugate transpose)
```

**Implementation Requirements:**
- ✅ Zero-copy: Use `ArrayView` for covariance matrix
- ✅ Numerically stable: Diagonal loading for regularization
- ✅ Error handling: Return `KwaversResult` for invalid inputs
- ✅ Validation: Check covariance is Hermitian, positive semi-definite
- ✅ Performance: No unnecessary allocations

**Test Coverage:**
- [ ] Unit test: Diagonal matrix (known solution)
- [ ] Unit test: Identity covariance (uniform spectrum)
- [ ] Unit test: Single source scenario (peak at source angle)
- [ ] Property test: Output is always positive
- [ ] Property test: Output is invariant to covariance scaling
- [ ] Integration test: Compare against MATLAB reference implementation

#### Task 2.2: Migrate Narrowband Steering Vectors
**File:** `domain/sensor/beamforming/narrowband/steering_narrowband.rs` → `analysis/.../narrowband/steering.rs`

**Steps:**
1. Create `src/analysis/signal_processing/beamforming/narrowband/steering.rs`
2. Migrate `NarrowbandSteeringVector` type
3. Migrate narrowband steering vector calculation
4. Unify with broadband steering vectors in `utils/mod.rs` where applicable
5. Add phase-wrapping utilities for narrowband case
6. Document mathematical relationship to broadband case

**Mathematical Foundation:**
```text
Narrowband Steering Vector:

a(θ, φ, f) = exp(-j 2π f τᵢ(θ, φ) / c)

where:
- τᵢ = time delay to sensor i for direction (θ, φ)
- f = operating frequency (narrowband assumption)
- c = speed of sound

Narrowband Assumption: Signal bandwidth << center frequency
```

**Implementation Requirements:**
- ✅ Complex exponential calculation (accurate phase)
- ✅ Support for arbitrary array geometries (3D positions)
- ✅ Normalization options (unit norm, max element = 1)
- ✅ Validation: Check direction vector is normalized

**Test Coverage:**
- [ ] Unit test: Linear array, broadside (all phases zero)
- [ ] Unit test: Linear array, endfire (linear phase progression)
- [ ] Unit test: Planar array, arbitrary direction
- [ ] Property test: Norm is consistent with normalization mode
- [ ] Property test: Phase wrapping correct (modulo 2π)

#### Task 2.3: Migrate Snapshot Extraction
**Files:** `domain/sensor/beamforming/narrowband/snapshots/` → `analysis/.../narrowband/snapshots/`

**Steps:**
1. Create `src/analysis/signal_processing/beamforming/narrowband/snapshots/mod.rs`
2. Migrate snapshot extraction utilities
3. Migrate windowing functions
4. Migrate STFT-based frequency extraction
5. Remove any sensor-specific coupling
6. Add comprehensive documentation

**Snapshot Extraction Methods:**
1. **Direct Extraction:** Sample at specific time index
2. **STFT-based:** Extract frequency bin from short-time Fourier transform
3. **Windowed:** Apply window function before extraction
4. **Spatial Smoothing:** Forward-backward averaging

**Implementation Requirements:**
- ✅ Configurable window functions (Hamming, Hanning, Blackman)
- ✅ Configurable overlap (50%, 75%, etc.)
- ✅ Zero-padding options for frequency resolution
- ✅ Normalization (preserve power, preserve amplitude)
- ✅ Multi-channel support (vectorized over sensors)

**Test Coverage:**
- [ ] Unit test: Single-frequency signal extracted correctly
- [ ] Unit test: Window function applied correctly
- [ ] Unit test: STFT matches FFT for stationary signals
- [ ] Property test: Power preserved with correct normalization
- [ ] Integration test: Extract snapshots from synthetic time-series

#### Task 2.4: Update Module Structure
**File:** `src/analysis/signal_processing/beamforming/narrowband/mod.rs`

**Steps:**
1. Replace placeholder with complete implementation
2. Define public API (re-exports)
3. Add comprehensive module-level documentation
4. Document usage examples
5. Document migration from old location

**Public API Design:**
```rust
// analysis/signal_processing/beamforming/narrowband/mod.rs

//! Narrowband (frequency-domain) beamforming algorithms.
//!
//! This module implements beamforming algorithms for narrowband signals,
//! where the signal bandwidth is much smaller than the center frequency.

pub mod capon;
pub mod steering;
pub mod snapshots;

// Convenience re-exports
pub use capon::{
    capon_spatial_spectrum,
    capon_spatial_spectrum_point,
    CaponSpectrumConfig,
};

pub use steering::{
    narrowband_steering_vector,
    NarrowbandSteeringConfig,
};

pub use snapshots::{
    extract_narrowband_snapshot,
    extract_stft_snapshot,
    SnapshotConfig,
    WindowFunction,
};
```

---

### Phase 3: Test Migration & Validation (3-4 hours)

#### Task 3.1: Migrate Existing Tests
- [ ] Identify tests in `domain/sensor/beamforming/narrowband/` that test migrated code
- [ ] Copy tests to `analysis/signal_processing/beamforming/narrowband/`
- [ ] Update imports to canonical location
- [ ] Validate all tests pass
- [ ] Remove duplicate tests from domain layer

**Test Organization:**
```text
src/analysis/signal_processing/beamforming/narrowband/
├── capon.rs
│   └── #[cfg(test)] mod tests { ... }
├── steering.rs
│   └── #[cfg(test)] mod tests { ... }
└── snapshots/
    └── mod.rs
        └── #[cfg(test)] mod tests { ... }
```

#### Task 3.2: Add Property-Based Tests
- [ ] Add proptest for Capon spatial spectrum (positivity, scaling invariance)
- [ ] Add proptest for steering vectors (norm properties, phase wrapping)
- [ ] Add proptest for snapshot extraction (power conservation, linearity)

**Example Property Test:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn capon_spectrum_always_positive(
        n_sensors in 4..16usize,
        steering_angle in -PI..PI,
    ) {
        let cov = create_test_covariance(n_sensors);
        let steering = narrowband_steering_vector(n_sensors, steering_angle);
        
        let spectrum = capon_spatial_spectrum(&cov, &steering)
            .expect("Capon failed");
        
        prop_assert!(spectrum > 0.0);
        prop_assert!(spectrum.is_finite());
    }
}
```

#### Task 3.3: Add Integration Tests
- [ ] End-to-end narrowband beamforming pipeline
- [ ] Multi-source localization scenario
- [ ] Compare against MATLAB reference (if available)
- [ ] Validate against analytical solutions (simple geometries)

**Integration Test Scenarios:**
1. **Single Source Detection:** Known source angle, verify peak in spatial spectrum
2. **Two-Source Resolution:** Two closely-spaced sources, verify resolution limit
3. **SNR Sensitivity:** Performance degradation with decreasing SNR
4. **Array Geometry Validation:** Linear, planar, circular arrays

---

### Phase 4: Consumer Updates (2-3 hours)

#### Task 4.1: Update Internal Consumers
- [ ] Update imports in analysis layer (if narrowband used by other modules)
- [ ] Update imports in domain layer (if any consumers remain)
- [ ] Update imports in tests
- [ ] Validate zero breaking changes

**Consumer Update Template:**
```rust
// Before (deprecated):
use crate::domain::sensor::beamforming::narrowband::{
    capon_spatial_spectrum,
    extract_narrowband_snapshot,
};

// After (canonical):
use crate::analysis::signal_processing::beamforming::narrowband::{
    capon_spatial_spectrum,
    extract_narrowband_snapshot,
};
```

#### Task 4.2: Update Examples (if applicable)
- [ ] Check if any examples use narrowband beamforming
- [ ] Update imports to canonical location
- [ ] Validate examples run correctly
- [ ] Update example documentation

---

### Phase 5: Performance Validation (1-2 hours)

#### Task 5.1: Establish Performance Baseline
- [ ] Run existing benchmarks (if any) on deprecated implementation
- [ ] Record baseline performance metrics
- [ ] Identify critical performance paths

#### Task 5.2: Benchmark Canonical Implementation
- [ ] Create benchmark suite for narrowband algorithms
- [ ] Benchmark Capon spatial spectrum (various matrix sizes)
- [ ] Benchmark snapshot extraction (various signal lengths)
- [ ] Benchmark steering vector calculation

**Benchmark Suite:**
```rust
// benches/narrowband_beamforming.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::analysis::signal_processing::beamforming::narrowband::*;

fn bench_capon_spatial_spectrum(c: &mut Criterion) {
    let n = 64;
    let cov = create_test_covariance(n);
    let steering = narrowband_steering_vector(n, 0.0);

    c.bench_function("capon_spatial_spectrum_64", |b| {
        b.iter(|| {
            black_box(capon_spatial_spectrum(
                black_box(&cov),
                black_box(&steering),
            ).unwrap())
        });
    });
}

criterion_group!(benches, bench_capon_spatial_spectrum);
criterion_main!(benches);
```

#### Task 5.3: Compare Performance
- [ ] Compare canonical vs deprecated performance
- [ ] Acceptance criterion: <5% change
- [ ] If regression detected, profile and optimize
- [ ] Document performance characteristics

**Performance Acceptance Criteria:**
- ✅ Capon spatial spectrum: <5% change
- ✅ Snapshot extraction: <10% change (acceptable for correctness improvements)
- ✅ Steering vector calculation: <2% change (should be identical)
- ✅ Memory allocations: No increase

---

### Phase 6: Cleanup & Documentation (1 hour)

#### Task 6.1: Remove Deprecated Implementation
- [ ] Delete `domain/sensor/beamforming/narrowband/capon.rs`
- [ ] Delete `domain/sensor/beamforming/narrowband/steering_narrowband.rs`
- [ ] Delete `domain/sensor/beamforming/narrowband/snapshots/`
- [ ] Keep `domain/sensor/beamforming/narrowband/mod.rs` as compatibility facade

**Compatibility Facade:**
```rust
// domain/sensor/beamforming/narrowband/mod.rs

#![deprecated(
    since = "2.15.0",
    note = "Use `analysis::signal_processing::beamforming::narrowband` instead"
)]

//! ⚠️ DEPRECATED: Moved to `analysis::signal_processing::beamforming::narrowband`

#[deprecated(since = "2.15.0")]
pub use crate::analysis::signal_processing::beamforming::narrowband::{
    capon_spatial_spectrum,
    narrowband_steering_vector,
    extract_narrowband_snapshot,
};
```

#### Task 6.2: Update Documentation
- [ ] Update `docs/adr.md` with migration decision
- [ ] Update README with new import paths
- [ ] Update `BEAMFORMING_ARCHITECTURE_ANALYSIS.md` status
- [ ] Add migration example to `BEAMFORMING_MIGRATION_GUIDE.md`

#### Task 6.3: Update Backlog
- [ ] Mark Sprint 1 tasks complete in `docs/backlog.md`
- [ ] Update Sprint 2 tasks based on learnings
- [ ] Document any deferred items or risks discovered

---

## Validation Checklist

### Pre-Sprint Validation
- [ ] Current codebase builds cleanly (`cargo build --all-features`)
- [ ] All tests passing (`cargo test --all-features`)
- [ ] Baseline benchmarks recorded
- [ ] Dependency graph documented

### In-Sprint Validation (After Each Task)
- [ ] Code compiles without errors
- [ ] All tests pass (unit, integration, property-based)
- [ ] `cargo clippy -- -D warnings` passes
- [ ] Rustdoc builds without warnings (`cargo doc --all-features`)

### Sprint Completion Validation
- [ ] Full test suite passes (867 tests)
- [ ] Zero regressions detected
- [ ] Performance benchmarks meet acceptance criteria
- [ ] All narrowband algorithms migrated to canonical location
- [ ] Zero uses of deprecated imports in internal code
- [ ] Documentation updated and accurate
- [ ] Compatibility facade in place
- [ ] Code review completed

---

## Risk Management

### Risk R1: Algorithm Divergence
**Risk:** Canonical and deprecated implementations have diverged (different behavior)

**Probability:** Medium  
**Impact:** High (mathematical incorrectness)

**Mitigation:**
1. Property-based cross-validation (compare outputs on random inputs)
2. Use existing test suite as oracle (both should pass same tests)
3. If divergence found, determine which is correct via literature validation
4. Document any intentional behavior changes

**Action Plan if Triggered:**
- Pause migration
- Investigate divergence root cause
- Consult literature for correct behavior
- Fix incorrect implementation
- Add regression test to prevent future divergence

### Risk R2: Performance Regression
**Risk:** Canonical implementation is slower than deprecated version

**Probability:** Low  
**Impact:** Medium (performance critical for real-time imaging)

**Mitigation:**
1. Benchmark before migration (establish baseline)
2. Profile canonical implementation if regression detected
3. Optimize hot paths (SIMD, cache-aware algorithms)
4. Accept <5% regression if correctness improved

**Action Plan if Triggered:**
- Profile with `cargo flamegraph` or `perf`
- Identify hot spots
- Optimize (vectorization, memory layout, algorithm choice)
- Re-benchmark
- Document trade-offs in ADR if performance sacrificed for correctness

### Risk R3: Circular Dependencies
**Risk:** Narrowband module depends on code that depends on narrowband (cycle)

**Probability:** Low  
**Impact:** Critical (blocks compilation)

**Mitigation:**
1. Analyze dependencies before migration (Task 1.2)
2. Migrate in bottom-up order (dependencies first)
3. Break cycles by moving shared code to lower layer (utils, math)

**Action Plan if Triggered:**
- Identify cycle using `cargo` error messages
- Move shared code to appropriate lower layer
- Update imports
- Validate cycle broken

### Risk R4: Missing Test Coverage
**Risk:** Migrated code has insufficient test coverage, bugs go undetected

**Probability:** Medium  
**Impact:** High (silent incorrectness)

**Mitigation:**
1. Require 100% function coverage for migrated code
2. Add property-based tests for mathematical invariants
3. Cross-validate against deprecated implementation
4. Add integration tests for end-to-end scenarios

**Action Plan if Triggered:**
- Run `cargo tarpaulin` to identify uncovered lines
- Add tests for uncovered branches
- Add property-based tests for edge cases
- Validate coverage meets threshold (≥95%)

---

## Success Metrics

### Quantitative Metrics
- ✅ **Code Duplication:** 0 duplicate narrowband implementations
- ✅ **Test Pass Rate:** 100% (867/867 tests passing)
- ✅ **Performance Change:** <5% on critical paths
- ✅ **Test Coverage:** ≥95% line coverage for narrowband module
- ✅ **Build Warnings:** 0 compiler warnings, 0 clippy warnings

### Qualitative Metrics
- ✅ **Architectural Purity:** Clear layer separation, no domain algorithms
- ✅ **Code Quality:** Comprehensive Rustdoc, clean API design
- ✅ **Maintainability:** Single source of truth, easy to extend
- ✅ **Testability:** Property-based tests validate mathematical correctness

---

## Timeline

### Week 1 Schedule

**Day 1 (2 hours):** Phase 1 — Source Analysis & Preparation
- Morning: Task 1.1, 1.2 — Inventory and dependency analysis
- Afternoon: Task 1.3 — Consumer analysis

**Day 2 (4 hours):** Phase 2 — Algorithm Migration (Part 1)
- Morning: Task 2.1 — Migrate Capon spatial spectrum
- Afternoon: Task 2.2 — Migrate narrowband steering vectors

**Day 3 (4 hours):** Phase 2 — Algorithm Migration (Part 2)
- Morning: Task 2.3 — Migrate snapshot extraction
- Afternoon: Task 2.4 — Update module structure

**Day 4 (3 hours):** Phase 3 — Test Migration & Validation
- Morning: Task 3.1 — Migrate existing tests
- Afternoon: Task 3.2, 3.3 — Add property-based and integration tests

**Day 5 (3 hours):** Phase 4-6 — Cleanup & Finalization
- Morning: Task 4.1, 4.2 — Update consumers
- Afternoon: Task 5.1-5.3 — Performance validation
- Evening: Task 6.1-6.3 — Cleanup and documentation

**Day 6 (Buffer):** Final validation and code review

---

## Sprint Retrospective Template

### What Went Well
- [ ] List successful outcomes
- [ ] Identify effective practices
- [ ] Note smooth collaborations

### What Could Be Improved
- [ ] List challenges encountered
- [ ] Identify bottlenecks
- [ ] Note areas for improvement

### Lessons Learned
- [ ] Technical learnings
- [ ] Process improvements
- [ ] Risk management insights

### Action Items for Next Sprint
- [ ] Apply lessons to Sprint 2
- [ ] Update process based on learnings
- [ ] Adjust estimates based on actual effort

---

## Appendix: Quick Reference

### File Locations
```text
Source (OLD):
  domain/sensor/beamforming/narrowband/capon.rs
  domain/sensor/beamforming/narrowband/steering_narrowband.rs
  domain/sensor/beamforming/narrowband/snapshots/

Target (NEW):
  analysis/signal_processing/beamforming/narrowband/capon.rs
  analysis/signal_processing/beamforming/narrowband/steering.rs
  analysis/signal_processing/beamforming/narrowband/snapshots/
```

### Key Commands
```bash
# Build and test
cargo build --all-features
cargo test --all-features
cargo clippy -- -D warnings

# Benchmarks
cargo bench --bench narrowband_beamforming

# Coverage
cargo tarpaulin --all-features --out Html

# Documentation
cargo doc --all-features --no-deps --open
```

### Import Update Pattern
```rust
// OLD (deprecated):
use crate::domain::sensor::beamforming::narrowband::*;

// NEW (canonical):
use crate::analysis::signal_processing::beamforming::narrowband::*;
```

---

**Sprint Status:** 🟡 Ready to Execute  
**Next Action:** Begin Phase 1 — Source Analysis & Preparation  
**Estimated Completion:** 1 week from start date  

---

*This sprint plan adheres to the Elite Mathematically-Verified Systems Architect principles: mathematical correctness first, architectural purity enforced, zero tolerance for duplication.*