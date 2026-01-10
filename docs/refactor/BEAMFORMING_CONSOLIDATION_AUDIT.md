# Beamforming Consolidation Audit

**Phase**: 1 (Critical Consolidation)  
**Sprint**: 4 (Final)  
**Date**: 2026-01-15  
**Status**: 🔍 AUDIT IN PROGRESS  
**Severity**: HIGH (4-way duplication, architectural layer violation)

---

## Executive Summary

Beamforming algorithms are duplicated across **4 distinct locations** in the codebase, representing the most complex cross-contamination pattern in Phase 1. This audit identifies all implementations, analyzes their purpose, and provides a consolidation strategy.

### Duplication Sites

1. **`analysis/signal_processing/beamforming/`** (Target consolidation location)
   - Status: Partially migrated, modern structure
   - Algorithms: DAS, MVDR, MUSIC, ESMV
   - Files: 8 files, ~2,000 LOC
   - Assessment: ✅ Correct layer, should be canonical

2. **`domain/sensor/beamforming/`** (Primary duplication source)
   - Status: Large, mature, feature-rich
   - Algorithms: DAS, MVDR, Capon, MUSIC, ESMV, neural, experimental
   - Files: 49 files, ~8,000 LOC
   - Assessment: ❌ Layer violation (domain should not contain analysis algorithms)

3. **`domain/source/transducers/phased_array/beamforming.rs`** (Transmit beamforming)
   - Status: Specialized for transmit focusing/steering
   - Algorithms: Focus delays, steering delays, plane wave
   - Files: 1 file, ~150 LOC
   - Assessment: ⚠️ Legitimate use case, but algorithm duplicates steering logic

4. **`core/utils/sparse_matrix/beamforming.rs`** (Matrix operations)
   - Status: Utility functions for beamforming matrices
   - Algorithms: Delay-sum matrix construction, sparse steering matrices
   - Files: 1 file, ~120 LOC
   - Assessment: ❌ Inappropriate location (domain logic in core utilities)

5. **`domain/sensor/passive_acoustic_mapping/beamforming_config.rs`** (PAM policy)
   - Status: Configuration wrapper, delegates to shared beamforming
   - Algorithms: None (policy only)
   - Files: 1 file, ~160 LOC
   - Assessment: ✅ Correct pattern (policy over shared algorithms)

**Total Scope**: ~60 files, ~10,500 LOC affected

---

## Detailed Inventory

### 1. `analysis/signal_processing/beamforming/` ✅ TARGET

**Location**: `src/analysis/signal_processing/beamforming/`  
**Status**: 🟢 Correct architectural layer  
**Role**: Post-processing and analysis algorithms

#### Structure

```
analysis/signal_processing/beamforming/
├── adaptive/
│   ├── mod.rs              (130 LOC) - Trait definitions, MinimumVariance, MUSIC, ESMV
│   ├── mvdr.rs             (280 LOC) - MVDR/Capon implementation
│   └── subspace.rs         (320 LOC) - MUSIC, ESMV implementations
├── time_domain/
│   ├── mod.rs              (450 LOC) - DAS core, delay calculation
│   ├── das.rs              (DEPRECATED - merged into mod.rs)
│   └── delay_reference.rs  (DEPRECATED - merged into mod.rs)
├── mod.rs                  (290 LOC) - Module docs, exports
└── test_utilities.rs       (530 LOC) - Test helpers
```

**Total**: 8 files, ~2,000 LOC

#### Algorithms Implemented

1. **Time-Domain DAS** (`time_domain/mod.rs`)
   - `delay_and_sum()`: Core DAS algorithm
   - `relative_delays_s()`: Delay calculation
   - `alignment_shifts_s()`: Sample alignment
   - `DelayReference` enum: Reference sensor policy

2. **Adaptive Beamforming** (`adaptive/mod.rs`, `adaptive/mvdr.rs`)
   - `MinimumVariance`: MVDR/Capon beamformer
   - Diagonal loading for stability
   - Trait: `AdaptiveBeamformer`

3. **Subspace Methods** (`adaptive/subspace.rs`)
   - `MUSIC`: Multiple signal classification
   - `EigenspaceMV`: Eigenspace minimum variance (ESMV)
   - Source number estimation

#### Assessment

**Strengths**:
- ✅ Correct architectural layer (analysis, not domain)
- ✅ Clean trait-based design
- ✅ Modern Rust idioms (zero-copy, ArrayView)
- ✅ Well-documented with references
- ✅ Comprehensive tests

**Gaps**:
- ❌ Missing narrowband frequency-domain methods
- ❌ Missing 3D beamforming support
- ❌ Missing GPU implementations
- ❌ Limited experimental/neural methods

**Recommendation**: **Canonical location** - consolidate all beamforming here.

---

### 2. `domain/sensor/beamforming/` ❌ PRIMARY DUPLICATION

**Location**: `src/domain/sensor/beamforming/`  
**Status**: 🔴 Architectural layer violation  
**Role**: Should handle sensor geometry only, currently contains full algorithm suite

#### Structure

```
domain/sensor/beamforming/
├── adaptive/
│   ├── mod.rs                  (450 LOC) - Main adaptive beamforming module
│   ├── adaptive.rs             (280 LOC) - Adaptive beamformer core
│   ├── algorithms/
│   │   ├── mod.rs              (120 LOC)
│   │   ├── covariance_taper.rs (180 LOC)
│   │   ├── delay_and_sum.rs    (150 LOC) ← DUPLICATE
│   │   ├── eigenspace_mv.rs    (220 LOC) ← DUPLICATE (ESMV)
│   │   ├── mod.rs              (80 LOC)
│   │   ├── music.rs            (240 LOC) ← DUPLICATE
│   │   ├── mvdr.rs             (200 LOC) ← DUPLICATE
│   │   ├── robust_capon.rs     (180 LOC)
│   │   ├── source_estimation.rs(150 LOC)
│   │   └── utils.rs            (100 LOC)
│   ├── algorithms_old.rs       (DEPRECATED)
│   ├── array_geometry.rs       (200 LOC)
│   ├── beamformer.rs           (300 LOC)
│   ├── conventional.rs         (180 LOC)
│   ├── matrix_utils.rs         (150 LOC)
│   ├── opast.rs                (250 LOC) - Online subspace tracking
│   ├── past.rs                 (220 LOC) - Projection approximation subspace tracking
│   ├── source_estimation.rs    (180 LOC)
│   ├── steering.rs             (200 LOC)
│   ├── subspace.rs             (280 LOC)
│   ├── tapering.rs             (150 LOC)
│   └── weights.rs              (180 LOC)
├── experimental/
│   ├── mod.rs                  (50 LOC)
│   └── neural.rs               (800 LOC) - Neural beamforming
├── narrowband/
│   ├── mod.rs                  (300 LOC)
│   ├── capon.rs                (350 LOC)
│   ├── snapshots/
│   │   ├── mod.rs              (200 LOC)
│   │   └── windowed/
│   │       └── mod.rs          (180 LOC)
│   └── steering_narrowband.rs  (150 LOC)
├── time_domain/
│   ├── mod.rs                  (200 LOC)
│   ├── das/
│   │   └── mod.rs              (400 LOC) ← DUPLICATE (DAS)
│   └── delay_reference.rs      (120 LOC) ← DUPLICATE
├── shaders/
│   └── mod.rs                  (GPU implementations, feature-gated)
├── ai_integration.rs           (600 LOC) - AI/PINN integration
├── beamforming_3d.rs           (500 LOC) - 3D beamforming
├── config.rs                   (250 LOC)
├── covariance.rs               (400 LOC)
├── mod.rs                      (180 LOC)
├── processor.rs                (350 LOC)
└── steering.rs                 (200 LOC)
```

**Total**: 49 files, ~8,000 LOC

#### Algorithms Implemented

1. **Time-Domain** (`time_domain/`)
   - Delay-and-sum (DAS) - DUPLICATE of `analysis/`
   - Delay reference policy - DUPLICATE
   - Time-domain steering

2. **Adaptive** (`adaptive/`)
   - MVDR (Minimum Variance Distortionless Response) - DUPLICATE
   - Robust Capon with diagonal loading - DUPLICATE
   - Eigenspace MVDR (ESMV) - DUPLICATE
   - MUSIC (Multiple Signal Classification) - DUPLICATE
   - PAST/OPAST (subspace tracking)
   - Source estimation

3. **Narrowband** (`narrowband/`)
   - Capon spatial spectrum
   - STFT-based snapshot extraction
   - Narrowband steering vectors

4. **Experimental** (`experimental/`)
   - Neural beamforming (PINN)
   - AI-enhanced beamforming
   - Distributed processing

5. **3D Beamforming** (`beamforming_3d.rs`)
   - Volumetric beamforming
   - 3D apodization
   - Metrics and processors

6. **Infrastructure**
   - Covariance estimation (`covariance.rs`)
   - Steering vector calculation (`steering.rs`)
   - Processor pipelines (`processor.rs`)
   - Configuration (`config.rs`)

#### Assessment

**Strengths**:
- ✅ Comprehensive algorithm suite
- ✅ Production-tested and mature
- ✅ Advanced features (GPU, neural, 3D)
- ✅ Well-documented with literature references

**Problems**:
- ❌ **Layer violation**: Analysis algorithms in domain layer
- ❌ **Duplicates** `analysis/signal_processing/beamforming/` implementations
- ❌ Mixes sensor geometry with signal processing
- ❌ Hard to maintain consistency across duplicates
- ❌ Confuses downstream users (which to use?)

**Recommendation**: **Migrate algorithms to `analysis/`**, keep only sensor geometry here.

---

### 3. `domain/source/transducers/phased_array/beamforming.rs` ⚠️ TRANSMIT

**Location**: `src/domain/source/transducers/phased_array/beamforming.rs`  
**Status**: 🟡 Specialized use case, partial duplication  
**Role**: Transmit beamforming for phased array focusing/steering

#### Structure

```rust
// Single file: beamforming.rs (~150 LOC)

pub enum BeamformingMode {
    Focus { target: (f64, f64, f64) },      // Focus at point
    Steer { theta: f64, phi: f64 },         // Steer to angle
    Custom { delays: Vec<f64> },            // Custom phase
    PlaneWave { direction: (f64, f64, f64) }, // Plane wave
}

pub struct BeamformingCalculator {
    sound_speed: f64,
    frequency: f64,
}

impl BeamformingCalculator {
    pub fn calculate_focus_delays(&self, positions: &[(f64, f64, f64)], target: (f64, f64, f64)) -> Vec<f64>;
    pub fn calculate_steering_delays(&self, positions: &[(f64, f64, f64)], theta: f64, phi: f64) -> Vec<f64>;
    pub fn calculate_plane_wave_delays(&self, positions: &[(f64, f64, f64)], direction: (f64, f64, f64)) -> Vec<f64>;
    pub fn calculate_beam_width(&self, aperture: f64) -> f64;
    pub fn calculate_focal_zone(&self, aperture: f64, focal_distance: f64) -> f64;
}
```

#### Algorithms Implemented

1. **Transmit Focusing**: Calculate delays to focus at a point
2. **Transmit Steering**: Calculate delays to steer beam to angle
3. **Plane Wave**: Calculate delays for plane wave transmission
4. **Beam Properties**: Beam width, focal zone calculations

#### Assessment

**Strengths**:
- ✅ Specialized for transmit (different use case than receive beamforming)
- ✅ Simple, focused implementation
- ✅ Appropriate location (part of source transducer)

**Problems**:
- ⚠️ **Duplicates delay calculation logic** from beamforming modules
- ⚠️ Beam width/focal zone calculations could be shared utilities

**Recommendation**: **Keep with refactoring** - Extract delay calculation utilities to shared module, maintain transmit-specific logic here.

---

### 4. `core/utils/sparse_matrix/beamforming.rs` ❌ INAPPROPRIATE LOCATION

**Location**: `src/core/utils/sparse_matrix/beamforming.rs`  
**Status**: 🔴 Architectural layer violation  
**Role**: Sparse matrix utilities for beamforming (domain logic in core)

#### Structure

```rust
// Single file: beamforming.rs (~120 LOC)

pub struct BeamformingMatrix {
    steering_matrix: CompressedSparseRowMatrix,
    num_elements: usize,
    num_directions: usize,
}

impl BeamformingMatrix {
    pub fn create(num_elements: usize, num_directions: usize) -> Self;
    
    pub fn build_delay_sum_matrix(
        &mut self,
        element_positions: ArrayView2<f64>,
        directions: ArrayView2<f64>,
        sound_speed: f64,
        frequency: f64,
    ) -> KwaversResult<()>;
    
    pub fn apply_weights(&self, data: &Array1<f64>) -> KwaversResult<Array1<f64>>;
    
    pub fn steering_matrix(&self) -> &CompressedSparseRowMatrix;
}
```

#### Algorithms Implemented

1. **Steering Matrix Construction**: Build sparse steering matrix from geometry
2. **Delay-Sum Matrix**: Delay-and-sum via sparse matrix multiplication
3. **Weight Application**: Apply beamforming weights efficiently

#### Assessment

**Strengths**:
- ✅ Efficient sparse matrix representation
- ✅ Useful for large-scale arrays

**Problems**:
- ❌ **Wrong layer**: Domain/analysis logic in `core/utils`
- ❌ **Naming collision**: "Beamforming" in utility module confusing
- ❌ Duplicates steering vector logic from beamforming modules
- ❌ Tight coupling to beamforming concepts

**Recommendation**: **Remove or refactor** - Move sparse matrix utilities to `analysis/beamforming/utils/sparse.rs` or make generic (remove beamforming-specific naming).

---

### 5. `domain/sensor/passive_acoustic_mapping/beamforming_config.rs` ✅ CORRECT PATTERN

**Location**: `src/domain/sensor/passive_acoustic_mapping/beamforming_config.rs`  
**Status**: 🟢 Correct architectural pattern  
**Role**: PAM-specific policy wrapper over shared beamforming algorithms

#### Structure

```rust
// Single file: beamforming_config.rs (~160 LOC)

pub enum PamBeamformingMethod {
    DelayAndSum,
    CaponDiagonalLoading { diagonal_loading: f64 },
    MUSIC { num_sources: usize },
    EigenspaceMinVariance { signal_subspace_dimension: usize },
    TimeExposureAcoustics, // PAM-specific post-processing
}

pub enum ApodizationType {
    Rectangular,
    Hamming,
    Hann,
    Tukey { alpha: f64 },
}

pub struct PamBeamformingConfig {
    pub core: BeamformingCoreConfig,        // Shared config
    pub method: PamBeamformingMethod,       // Algorithm selection
    pub frequency_range: (f64, f64),        // PAM-specific
    pub spatial_resolution: f64,            // PAM-specific
    pub apodization: ApodizationType,       // PAM-specific
    pub focal_point: [f64; 3],              // PAM-specific
}

impl PamBeamformingConfig {
    pub fn validate(&self) -> KwaversResult<()>;
    // No algorithm implementations, just policy
}
```

#### Assessment

**Strengths**:
- ✅ **Correct pattern**: Policy over shared algorithms
- ✅ Delegates to `domain::sensor::beamforming::BeamformingCoreConfig`
- ✅ PAM-specific concerns clearly separated
- ✅ No algorithm duplication

**Problems**:
- ⚠️ Currently depends on `domain::sensor::beamforming` (which will be deprecated)
- ⚠️ Needs update to reference `analysis::beamforming` after consolidation

**Recommendation**: **Keep as-is**, update dependency after consolidation to `analysis/beamforming/`.

---

## Duplication Matrix

| Algorithm | `analysis/` | `domain/sensor/` | `phased_array/` | `sparse_matrix/` |
|-----------|-------------|------------------|-----------------|------------------|
| **Delay-and-Sum (DAS)** | ✅ Implemented | ❌ DUPLICATE | ⚠️ Transmit variant | ⚠️ Matrix form |
| **MVDR/Capon** | ✅ Implemented | ❌ DUPLICATE | - | - |
| **MUSIC** | ✅ Implemented | ❌ DUPLICATE | - | - |
| **ESMV** | ✅ Implemented | ❌ DUPLICATE | - | - |
| **Robust Capon** | ❌ Missing | ✅ Unique | - | - |
| **Narrowband Capon** | ❌ Missing | ✅ Unique | - | - |
| **3D Beamforming** | ❌ Missing | ✅ Unique | - | - |
| **Neural/PINN** | ❌ Missing | ✅ Unique | - | - |
| **PAST/OPAST** | ❌ Missing | ✅ Unique | - | - |
| **Covariance Estimation** | ⚠️ Partial | ✅ Complete | - | - |
| **Steering Vectors** | ⚠️ Basic | ✅ Complete | ⚠️ Transmit | ⚠️ Sparse |
| **Delay Calculation** | ✅ Implemented | ❌ DUPLICATE | ⚠️ Transmit | - |

**Key**:
- ✅ Implemented / Canonical
- ❌ Duplicate (should be removed)
- ⚠️ Partial overlap (needs refactoring)
- `-` Not applicable

---

## Consolidation Strategy

### Goal

Establish `analysis/signal_processing/beamforming/` as the **Single Source of Truth** for all beamforming algorithms, with clear architectural layering:

```
Domain Layer (Geometry)
  ├── Sensor positions
  ├── Element spacing
  └── Array configuration
       ↓
Analysis Layer (Algorithms) ← CANONICAL
  ├── Time-domain beamforming
  ├── Frequency-domain beamforming
  ├── Adaptive beamforming
  └── Experimental methods
       ↓
Application Layer (Policy)
  ├── PAM beamforming config
  ├── Imaging presets
  └── Clinical workflows
```

### Phase 1: Audit and Planning ✅ (Current)

**Deliverables**:
- [x] Comprehensive inventory of all beamforming implementations
- [x] Duplication matrix identifying overlaps
- [x] Consolidation strategy document (this file)
- [x] Risk assessment

**Duration**: 4-6 hours  
**Status**: ✅ COMPLETE

---

### Phase 2: Infrastructure Setup (4-6 hours)

**Objective**: Prepare `analysis/beamforming/` to receive migrated algorithms

**Tasks**:
1. **Extend trait hierarchy**
   - Define `Beamformer` base trait
   - Define `TimeDomainBeamformer`, `FrequencyDomainBeamformer` sub-traits
   - Define `AdaptiveBeamformer` trait (already exists, review)

2. **Add missing modules**
   - `analysis/beamforming/narrowband/` for frequency-domain methods
   - `analysis/beamforming/experimental/` for neural/PINN methods
   - `analysis/beamforming/utils/` for shared utilities
   - `analysis/beamforming/covariance/` for covariance estimation

3. **Create migration utilities**
   - Compatibility layer for transitioning code
   - Type aliases for smooth migration
   - Re-export deprecated types with warnings

**Deliverables**:
- Extended trait definitions
- Module structure for all algorithm categories
- Migration utility module

---

### Phase 3: Algorithm Migration (12-16 hours)

**Objective**: Move unique algorithms from `domain/sensor/beamforming/` to `analysis/beamforming/`

#### 3.1 Narrowband Methods (4 hours)

**Source**: `domain/sensor/beamforming/narrowband/`  
**Target**: `analysis/beamforming/narrowband/`

**Algorithms**:
- Capon spatial spectrum
- Narrowband steering vectors
- STFT snapshot extraction
- Windowing functions

**Strategy**:
1. Copy narrowband module to `analysis/`
2. Update imports to use analysis layer
3. Add tests
4. Deprecate `domain/sensor/beamforming/narrowband/`

#### 3.2 Advanced Adaptive Methods (4 hours)

**Source**: `domain/sensor/beamforming/adaptive/`  
**Target**: `analysis/beamforming/adaptive/` (extend existing)

**Algorithms** (not yet in `analysis/`):
- Robust Capon
- PAST/OPAST subspace tracking
- Source estimation
- Covariance tapering

**Strategy**:
1. Identify unique algorithms (not duplicates)
2. Migrate to `analysis/beamforming/adaptive/`
3. Unify with existing MVDR/MUSIC/ESMV implementations
4. Add comprehensive tests

#### 3.3 3D Beamforming (3 hours)

**Source**: `domain/sensor/beamforming/beamforming_3d.rs`  
**Target**: `analysis/beamforming/volumetric/` (new module)

**Algorithms**:
- 3D delay-and-sum
- Volumetric apodization
- 3D beam metrics

**Strategy**:
1. Create `analysis/beamforming/volumetric/` module
2. Migrate 3D beamforming logic
3. Separate geometry (stay in domain) from processing (analysis)

#### 3.4 Experimental Methods (3 hours)

**Source**: `domain/sensor/beamforming/experimental/`  
**Target**: `analysis/beamforming/experimental/`

**Algorithms**:
- Neural beamforming (PINN)
- AI-enhanced beamforming
- Distributed processing

**Strategy**:
1. Move experimental module wholesale
2. Keep feature gates (`feature = "experimental_neural"`)
3. Update imports and dependencies

#### 3.5 Covariance Infrastructure (2 hours)

**Source**: `domain/sensor/beamforming/covariance.rs`  
**Target**: `analysis/beamforming/covariance/`

**Components**:
- Covariance estimator
- Spatial smoothing
- Post-processing

**Strategy**:
1. Create `analysis/beamforming/covariance/` module
2. Migrate covariance estimation logic
3. Keep as shared utility for adaptive methods

---

### Phase 4: Refactor Transmit Beamforming (2-3 hours)

**Objective**: Extract shared delay calculation utilities, keep transmit-specific logic

**Tasks**:
1. **Create shared utilities**
   - `analysis/beamforming/utils/delays.rs`
   - Generic delay calculation functions
   - Beam property calculations (width, focal zone)

2. **Refactor `phased_array/beamforming.rs`**
   - Use shared delay utilities
   - Keep transmit-specific `BeamformingMode` enum
   - Keep `BeamformingCalculator` as wrapper

3. **Update tests**
   - Verify transmit beamforming still works
   - Add tests for shared utilities

**Deliverables**:
- `analysis/beamforming/utils/delays.rs` (shared delay calculations)
- Refactored `domain/source/transducers/phased_array/beamforming.rs`
- Zero breaking changes for phased array users

---

### Phase 5: Handle Sparse Matrix Utilities (2 hours)

**Objective**: Move or make generic `core/utils/sparse_matrix/beamforming.rs`

**Option A: Move to Analysis** (Recommended)
- Move to `analysis/beamforming/utils/sparse.rs`
- Rename to clarify it's beamforming-specific
- Update all imports

**Option B: Make Generic**
- Remove beamforming-specific naming
- Make generic sparse matrix utilities
- Create adapter in `analysis/beamforming/`

**Chosen**: Option A (simpler, clearer intent)

**Tasks**:
1. Move file to `analysis/beamforming/utils/sparse.rs`
2. Update imports in consumers
3. Add deprecation notice in old location
4. Update tests

---

### Phase 6: Deprecation and Migration (4-6 hours)

**Objective**: Deprecate `domain/sensor/beamforming/`, provide migration path

**Tasks**:
1. **Add deprecation attributes**
   ```rust
   #[deprecated(
       since = "2.17.0",
       note = "Moved to `analysis::signal_processing::beamforming`. \
               See migration guide at docs/refactor/BEAMFORMING_MIGRATION.md"
   )]
   pub mod beamforming;
   ```

2. **Create migration guide**
   - Before/after examples for each algorithm
   - Import path changes
   - API changes (if any)
   - Timeline for removal (3.0.0)

3. **Update internal consumers**
   - PAM: Update to use `analysis/beamforming/`
   - Imaging: Update imports
   - Clinical: Update imports
   - Tests: Update imports

4. **Add compatibility re-exports**
   ```rust
   // In domain/sensor/mod.rs
   #[deprecated(since = "2.17.0")]
   pub use crate::analysis::signal_processing::beamforming as beamforming;
   ```

**Deliverables**:
- Deprecation attributes on all `domain/sensor/beamforming` items
- Migration guide (comprehensive)
- Updated consumers (internal)
- Compatibility layer for external users

---

### Phase 7: Testing and Validation (4-6 hours)

**Objective**: Ensure zero regressions, all tests pass

**Tasks**:
1. **Run full test suite**
   - `cargo test --all-features`
   - Verify all beamforming tests pass

2. **Add integration tests**
   - Test each migrated algorithm
   - Test backward compatibility layer
   - Test import paths work

3. **Performance benchmarks**
   - Verify zero performance regression
   - Compare old vs new implementations

4. **Architecture validation**
   - Run `cargo xtask arch` (if available)
   - Verify layer violations cleared
   - Check dependency graph

**Deliverables**:
- 100% test pass rate
- Performance benchmark results
- Architecture validation report

---

## Risk Assessment

### High Risks 🔴

1. **Breaking Changes for External Users**
   - **Risk**: Users depend on `domain/sensor/beamforming` API
   - **Impact**: Build failures, migration burden
   - **Mitigation**: 
     - Long deprecation period (2.17.0 → 3.0.0)
     - Comprehensive migration guide
     - Re-export compatibility layer
     - Clear communication in release notes

2. **Feature Parity Loss**
   - **Risk**: Miss unique algorithms during migration
   - **Impact**: Features lost, users blocked
   - **Mitigation**:
     - Comprehensive audit (this document)
     - Algorithm-by-algorithm migration checklist
     - Test coverage for all migrated algorithms

3. **Test Coverage Gaps**
   - **Risk**: Migrated code lacks tests, regressions introduced
   - **Impact**: Bugs in production
   - **Mitigation**:
     - Maintain existing test coverage
     - Add new tests for edge cases
     - Run benchmarks to catch performance regressions

### Medium Risks 🟡

4. **GPU/Experimental Feature Compatibility**
   - **Risk**: Feature-gated code may break during migration
   - **Impact**: GPU users blocked
   - **Mitigation**:
     - Test with all feature combinations
     - Keep feature gates consistent
     - Document feature requirements

5. **Covariance Estimation Coupling**
   - **Risk**: Covariance module tightly coupled to domain layer
   - **Impact**: Difficult to migrate cleanly
   - **Mitigation**:
     - Extract interfaces first
     - Migrate incrementally
     - Keep backward compatibility during transition

6. **Documentation Drift**
   - **Risk**: Docs not updated during migration
   - **Impact**: User confusion
   - **Mitigation**:
     - Update docs alongside code
     - Add migration guide
     - Update examples in README

### Low Risks 🟢

7. **Phased Array Transmit Beamforming**
   - **Risk**: Refactoring breaks transmit logic
   - **Impact**: Source generation broken
   - **Mitigation**:
     - Minimal changes (extract utilities only)
     - Comprehensive tests
     - Keep transmit-specific logic isolated

8. **PAM Configuration Layer**
   - **Risk**: PAM config breaks during migration
   - **Impact**: PAM workflows broken
   - **Mitigation**:
     - Simple dependency update (one import path change)
     - Existing tests should catch issues

---

## Success Criteria

### Must Have ✅

- [ ] All unique algorithms migrated to `analysis/beamforming/`
- [ ] Zero breaking changes (backward compatibility maintained)
- [ ] All tests passing (100% pass rate)
- [ ] `domain/sensor/beamforming/` deprecated with warnings
- [ ] Migration guide created with examples
- [ ] Architecture layer violations eliminated
- [ ] Zero performance regression

### Nice to Have 📋

- [ ] GPU implementations tested across all features
- [ ] Benchmarks showing performance improvements
- [ ] Architecture checker passes (zero violations)
- [ ] Property-based tests added
- [ ] SIMD optimizations verified

---

## Effort Estimation

| Phase | Description | Estimated Hours | Risk | Dependencies |
|-------|-------------|----------------|------|--------------|
| 1 | Audit (this doc) | 4-6 | Low | None |
| 2 | Infrastructure setup | 4-6 | Low | Phase 1 |
| 3 | Algorithm migration | 12-16 | High | Phase 2 |
| 4 | Transmit refactor | 2-3 | Medium | Phase 3 |
| 5 | Sparse matrix | 2 | Low | Phase 3 |
| 6 | Deprecation | 4-6 | Medium | Phase 3-5 |
| 7 | Testing | 4-6 | Medium | Phase 6 |
| **Total** | **32-45 hours** | **~5-6 days** | - | - |

**Buffer**: Add 20% for unexpected issues → **38-54 hours** (6-7 days)

**Compared to Original Estimate**: 28-36 hours → **Revised: 38-54 hours** (more complex than anticipated)

---

## Implementation Checklist

### Phase 1: Audit ✅
- [x] Inventory all beamforming locations
- [x] Identify duplicates vs. unique algorithms
- [x] Create duplication matrix
- [x] Risk assessment
- [x] Effort estimation
- [x] Document audit (this file)

### Phase 2: Infrastructure
- [ ] Define `Beamformer` base trait
- [ ] Create `analysis/beamforming/narrowband/` module
- [ ] Create `analysis/beamforming/experimental/` module
- [ ] Create `analysis/beamforming/utils/` module
- [ ] Create `analysis/beamforming/covariance/` module
- [ ] Create migration utility stubs

### Phase 3: Migration
- [ ] Migrate narrowband Capon
- [ ] Migrate STFT snapshot extraction
- [ ] Migrate Robust Capon
- [ ] Migrate PAST/OPAST
- [ ] Migrate 3D beamforming
- [ ] Migrate neural/PINN beamforming
- [ ] Migrate covariance estimation
- [ ] Add tests for all migrated algorithms

### Phase 4: Refactor
- [ ] Extract delay utilities to `utils/delays.rs`
- [ ] Refactor `phased_array/beamforming.rs`
- [ ] Update phased array tests

### Phase 5: Sparse Matrix
- [ ] Move `sparse_matrix/beamforming.rs` to `analysis/utils/sparse.rs`
- [ ] Update all imports
- [ ] Deprecate old location

### Phase 6: Deprecation
- [ ] Add deprecation attributes to `domain/sensor/beamforming/`
- [ ] Create migration guide
- [ ] Add compatibility re-exports
- [ ] Update PAM config to use new paths
- [ ] Update internal consumers

### Phase 7: Testing
- [ ] Run full test suite
- [ ] Add integration tests
- [ ] Run benchmarks
- [ ] Architecture validation
- [ ] Update documentation

---

## Timeline

**Sprint Duration**: 5-7 days (full-time equivalent)

| Day | Phase | Activities | Deliverable |
|-----|-------|------------|-------------|
| 1 | Audit ✅ | Inventory, analysis, planning | This document |
| 2 | Infrastructure | Traits, modules, stubs | Ready for migration |
| 3-4 | Migration | Narrowband, adaptive, 3D, experimental | Algorithms migrated |
| 5 | Refactor | Transmit, sparse matrix | Utilities extracted |
| 6 | Deprecation | Deprecate old, migration guide | Users can migrate |
| 7 | Testing | Tests, benchmarks, validation | Production-ready |

---

## Next Steps

1. **Review this audit** with stakeholders
2. **Approve consolidation strategy**
3. **Begin Phase 2** (Infrastructure Setup)
4. **Execute migration** following phases 3-7
5. **Document progress** in sprint summary

---

**Audit Date**: 2026-01-15  
**Prepared By**: Elite Mathematically-Verified Systems Architect  
**Status**: AUDIT COMPLETE - Ready for Phase 2 execution  
**Approval**: Pending stakeholder review