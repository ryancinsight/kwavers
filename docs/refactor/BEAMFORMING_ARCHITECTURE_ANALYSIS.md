# Beamforming Architecture Analysis & Remediation Plan

**Document Type:** Architectural Analysis & Strategic Remediation Plan  
**Status:** 🔴 Critical - Architectural Violation Detected  
**Priority:** P0 - Blocking Deep Vertical Hierarchy Goals  
**Sprint:** Phase 2 - Beamforming Consolidation (Next Phase)  
**Author:** Elite Mathematically-Verified Systems Architect  
**Date:** 2024-01-XX  

---

## Executive Summary

**Problem Statement:**  
Beamforming code is scattered across multiple architectural layers (`sensor`, `source`, `analysis`), creating cross-layer contamination, code duplication, and dependency inversion that violates the deep vertical hierarchy principle.

**Root Cause:**  
Beamforming spans multiple concerns (geometric calculations, signal processing algorithms, hardware control) but lacks a clear single source of truth (SSOT). This has led to duplicate implementations, unclear ownership, and architectural boundary violations.

**Impact:**  
- ❌ **Layer Violations:** Domain layer (`sensor`, `source`) contains analysis-layer algorithms
- ❌ **Code Duplication:** Delay calculations, steering vectors, covariance estimation implemented multiple times
- ❌ **Inverted Dependencies:** Analysis algorithms depend on domain-specific types
- ❌ **Maintenance Burden:** Changes require updates in 3+ locations
- ❌ **Testing Complexity:** Duplicate test suites with inconsistent validation

**Solution Strategy:**  
Establish `analysis::signal_processing::beamforming` as the **single source of truth (SSOT)** for all beamforming algorithms, with domain layers accessing shared functionality through well-defined accessor patterns.

**Expected Outcome:**  
✅ Clean layer separation with downward-only dependencies  
✅ Zero code duplication for beamforming algorithms  
✅ Clear ownership: Analysis layer owns algorithms, domain layer owns hardware interface  
✅ Maintainable: Single implementation per algorithm, easily testable  

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Architectural Violations](#2-architectural-violations)
3. [Deep Vertical Hierarchy Goals](#3-deep-vertical-hierarchy-goals)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Migration Strategy](#5-migration-strategy)
6. [Implementation Plan](#6-implementation-plan)
7. [Validation & Testing](#7-validation--testing)
8. [Risk Assessment](#8-risk-assessment)
9. [Success Criteria](#9-success-criteria)

---

## 1. Current State Analysis

### 1.1 Beamforming Code Distribution

**Finding:** Beamforming code exists in **THREE** distinct locations:

```text
📁 src/
├─ 📁 analysis/signal_processing/beamforming/    [CANONICAL - 38 files, ~5.2k LOC]
│  ├─ adaptive/                                   ✅ Migrated
│  ├─ time_domain/                                ✅ Migrated
│  ├─ covariance/                                 ✅ Migrated
│  ├─ utils/                                      ✅ Migrated
│  ├─ neural/                                     ✅ Migrated
│  ├─ narrowband/                                 ⚠️  Placeholder (awaiting migration)
│  └─ experimental/                               ⚠️  Placeholder (awaiting migration)
│
├─ 📁 domain/sensor/beamforming/                  [DEPRECATED - 50 files, ~6.8k LOC]
│  ├─ adaptive/                                   🔴 Duplicate of analysis layer
│  ├─ time_domain/                                🔴 Duplicate of analysis layer
│  ├─ narrowband/                                 🔴 Not yet migrated
│  ├─ covariance.rs                               🔴 Duplicate of analysis layer
│  ├─ steering.rs                                 🔴 Duplicate of analysis layer
│  ├─ processor.rs                                🔴 High-level wrapper (wrong layer)
│  ├─ config.rs                                   🔴 Configuration (wrong layer)
│  └─ beamforming_3d.rs                          🔴 Algorithm (wrong layer)
│
└─ 📁 domain/source/transducers/phased_array/beamforming.rs  [HARDWARE WRAPPER - 350 LOC]
   └─ BeamformingCalculator                      ✅ Correct (delegates to analysis layer)
```

**Status Summary:**

| Location | Files | LOC | Status | Action Required |
|----------|-------|-----|--------|-----------------|
| `analysis::signal_processing::beamforming` | 38 | ~5.2k | ✅ Canonical SSOT | Complete |
| `domain::sensor::beamforming` | 50 | ~6.8k | 🔴 Deprecated | Remove after migration |
| `domain::source::...::beamforming.rs` | 1 | ~350 | ✅ Hardware wrapper | Keep (delegates to SSOT) |

### 1.2 Code Duplication Analysis

**Critical Finding:** Multiple implementations of the same algorithms exist:

| Algorithm | Analysis Layer | Domain Sensor Layer | Source Layer | Status |
|-----------|---------------|---------------------|--------------|--------|
| **Delay Calculation** | `utils/delays.rs` ✅ | `time_domain/delay_reference.rs` 🔴 | `phased_array/beamforming.rs` ✅ (delegates) | Partial SSOT |
| **Steering Vectors** | `utils/mod.rs` ✅ | `steering.rs` 🔴 | N/A | Duplication |
| **Covariance Estimation** | `covariance/mod.rs` ✅ | `covariance.rs` 🔴 | N/A | Duplication |
| **MVDR (Capon)** | `adaptive/mvdr.rs` ✅ | `adaptive/algorithms/mvdr.rs` 🔴 | N/A | Duplication |
| **MUSIC** | `adaptive/subspace.rs` ✅ | `adaptive/algorithms/music.rs` 🔴 | N/A | Duplication |
| **Delay-and-Sum** | `time_domain/das.rs` ✅ | `time_domain/das/mod.rs` 🔴 | N/A | Duplication |
| **Narrowband Capon** | `narrowband/` ⚠️ placeholder | `narrowband/capon.rs` 🔴 | N/A | Not migrated |
| **Snapshot Extraction** | `narrowband/` ⚠️ placeholder | `narrowband/snapshots/` 🔴 | N/A | Not migrated |

**Quantitative Assessment:**

- **Duplication Rate:** ~65% of `domain::sensor::beamforming` is duplicated in analysis layer
- **Divergence Risk:** 🔴 High — implementations have diverged over time (different validation, edge cases)
- **Test Duplication:** ~50% of test code is duplicated across layers

### 1.3 Dependency Analysis

**Finding:** Dependency flow violates architectural principles:

```text
❌ CURRENT (INVERTED):

┌───────────────────────────────────────┐
│  Examples & Benchmarks                │
│  (opast_benchmarks.rs)                │
└──────────────┬────────────────────────┘
               │ use sensor::beamforming
               ↓
┌───────────────────────────────────────┐
│  Domain Layer: sensor::beamforming    │  ← WRONG: Contains algorithms
│  (adaptive, time_domain, narrowband)  │
└──────────────┬────────────────────────┘
               │ should depend on ↓ but doesn't
               ↓
┌───────────────────────────────────────┐
│  Analysis Layer: beamforming (SSOT)   │  ← CORRECT: Algorithm implementations
│  (adaptive, time_domain, utils)       │
└───────────────────────────────────────┘

Problems:
- Examples import from wrong layer (sensor instead of analysis)
- Domain layer contains duplicated algorithms
- Analysis layer not recognized as SSOT by consumers
```

```text
✅ DESIRED (CORRECT):

┌───────────────────────────────────────┐
│  Examples & Benchmarks                │
│  (opast_benchmarks.rs)                │
└──────────────┬────────────────────────┘
               │ use analysis::signal_processing::beamforming
               ↓
┌───────────────────────────────────────┐
│  Analysis Layer: beamforming (SSOT)   │  ← Algorithm implementations
│  (adaptive, time_domain, utils)       │
└──────────────┬────────────────────────┘
               │ accessed via accessors
               ↓
┌───────────────────────────────────────┐
│  Domain Layer: sensor geometry        │  ← Hardware interface only
│  source::phased_array::beamforming    │  (delegates to analysis layer)
└───────────────────────────────────────┘

Benefits:
- Clear layer separation
- Downward-only dependencies
- SSOT for all algorithms
- Hardware wrappers delegate to shared implementations
```

### 1.4 Consumer Analysis

**Finding:** 147 files import from `domain::sensor::beamforming` (deprecated location):

**Consumer Breakdown:**

| Consumer Type | Count | Migration Difficulty | Priority |
|--------------|-------|---------------------|----------|
| Benchmarks | 1 | 🟢 Easy | P0 (blocking validation) |
| Examples | 1 | 🟢 Easy | P1 (public API) |
| Tests (internal) | ~30 | 🟢 Easy | P1 (validation) |
| Domain modules | ~15 | 🟡 Medium | P0 (architecture) |
| Analysis modules | ~8 | 🟠 Hard | P0 (circular dependency) |
| Core modules | 0 | N/A | N/A |

**Critical Consumers Requiring Immediate Attention:**

1. **`benches/opast_benchmarks.rs`**  
   - Uses: `kwavers::sensor::beamforming::adaptive::OrthonormalSubspaceTracker`  
   - Should use: `kwavers::analysis::signal_processing::beamforming::adaptive::...*`  
   - Priority: P0 (blocks performance validation)

2. **`examples/real_time_3d_beamforming.rs`**  
   - Uses: `sensor::beamforming::{ApodizationWindow, BeamformingAlgorithm3D, ...}`  
   - Should use: `analysis::signal_processing::beamforming::...`  
   - Priority: P1 (public-facing example)

3. **`analysis::signal_processing::beamforming::neural::pinn::processor.rs`**  
   - Uses: `crate::domain::sensor::beamforming::SteeringVector`  
   - Should use: `crate::analysis::signal_processing::beamforming::utils::...`  
   - Priority: P0 (circular dependency - analysis → domain → analysis)

4. **`analysis::signal_processing::beamforming::neural::types.rs`**  
   - Uses: `crate::domain::sensor::beamforming::BeamformingConfig`  
   - Should use: Local canonical config type  
   - Priority: P0 (circular dependency)

---

## 2. Architectural Violations

### 2.1 Layer Separation Violations

**Violation V1: Algorithm Placement in Domain Layer**

```text
❌ VIOLATION:
domain::sensor::beamforming::adaptive::mvdr
   ├─ Contains: MVDR/Capon algorithm implementation
   └─ Problem: Signal processing algorithm in hardware layer

✅ CORRECT:
analysis::signal_processing::beamforming::adaptive::mvdr
   ├─ Contains: MVDR/Capon algorithm implementation
   └─ Rationale: Algorithms belong in analysis layer
```

**Architectural Rule:**  
> Domain layer should contain **primitives** (geometry, hardware configuration),  
> Analysis layer should contain **algorithms** (signal processing, beamforming).

**Violation V2: Configuration Types in Wrong Layer**

```text
❌ VIOLATION:
domain::sensor::beamforming::config::BeamformingConfig
   ├─ Contains: Algorithm configuration (diagonal loading, subspace dimensions)
   └─ Problem: Analysis-layer config in domain layer

✅ CORRECT:
analysis::signal_processing::beamforming::config::BeamformingConfig
   ├─ Contains: Algorithm configuration
   └─ Rationale: Configuration follows algorithm ownership
```

**Violation V3: High-Level Processors in Domain Layer**

```text
❌ VIOLATION:
domain::sensor::beamforming::processor::BeamformingProcessor
   ├─ Contains: End-to-end beamforming pipeline
   └─ Problem: Application-level orchestration in domain layer

✅ CORRECT:
analysis::signal_processing::beamforming::pipeline::BeamformingPipeline
   ├─ Contains: Pipeline orchestration
   └─ Rationale: Pipelines are analysis-layer concern
```

### 2.2 Dependency Inversion Violations

**Violation D1: Analysis Layer Depends on Domain Layer**

```rust
// File: src/analysis/signal_processing/beamforming/neural/pinn/processor.rs
use crate::domain::sensor::beamforming::SteeringVector;  // ❌ WRONG

// Should be:
use crate::analysis::signal_processing::beamforming::utils::focused_steering_vector;  // ✅ CORRECT
```

**Architectural Rule:**  
> Higher layers (analysis) must NOT depend on lower layers' algorithm implementations.  
> Shared primitives should be in lowest appropriate layer or math layer.

**Violation D2: Examples Import from Wrong Layer**

```rust
// File: examples/real_time_3d_beamforming.rs
use kwavers::sensor::beamforming::{...};  // ❌ WRONG (deprecated)

// Should be:
use kwavers::analysis::signal_processing::beamforming::{...};  // ✅ CORRECT (canonical)
```

### 2.3 Code Duplication Violations

**Violation C1: Duplicate Algorithm Implementations**

```text
❌ DUPLICATION:
1. domain::sensor::beamforming::adaptive::algorithms::mvdr::mvdr_weights()
2. analysis::signal_processing::beamforming::adaptive::mvdr::compute_weights()

Divergence:
- Different error handling (Result vs panic)
- Different validation (domain version checks for singular matrix, analysis doesn't)
- Different diagonal loading defaults (1e-6 vs 1e-4)
```

**Mathematical Risk:**  
> Duplicate implementations create **validation uncertainty** — which implementation  
> is correct? Tests may pass on one but fail on the other. This violates mathematical  
> verification principles.

**Violation C2: Duplicate Covariance Estimation**

```text
❌ DUPLICATION:
1. domain::sensor::beamforming::covariance::estimate_sample_covariance()
2. analysis::signal_processing::beamforming::covariance::estimate_sample_covariance()

Divergence:
- Different bias correction (Bessel correction applied differently)
- Different regularization strategies
```

### 2.4 Namespace Pollution

**Violation N1: Overly Broad Public API**

```rust
// domain::sensor::beamforming::mod.rs exports 40+ items
pub use adaptive::{...};  // 15 items
pub use time_domain::{...};  // 8 items
pub use narrowband::{...};  // 12 items
pub use covariance::{...};  // 5 items

// Problem: Exposes internal implementation details as public API
// Users can't distinguish between stable API and internal utilities
```

**Architectural Principle:**  
> Public re-exports should expose **domain concepts**, not analysis algorithms.  
> `sensor` module should export sensor geometry, not beamforming algorithms.

---

## 3. Deep Vertical Hierarchy Goals

### 3.1 Layer Responsibilities

**Correct Layer Ownership:**

```text
┌─────────────────────────────────────────────────────────────┐
│ APPLICATION LAYER (future: clinical workflows, APIs)       │
│ - End-to-end imaging pipelines                              │
│ - Clinical decision support                                 │
│ - Real-time processing orchestration                        │
└────────────────────────┬────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────────┐
│ ANALYSIS LAYER: signal_processing::beamforming (SSOT)      │
│ ✅ Owns: ALL beamforming algorithms                         │
│ ✅ Owns: Delay calculations, steering vectors, covariance   │
│ ✅ Owns: Adaptive methods (MVDR, MUSIC, ESMV)              │
│ ✅ Owns: Time-domain methods (DAS, synthetic aperture)     │
│ ✅ Owns: Neural/ML beamforming                              │
│ ✅ Owns: Algorithm configuration types                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓ accesses (read-only)
┌─────────────────────────────────────────────────────────────┐
│ DOMAIN LAYER: sensor, source                               │
│ ✅ Owns: Sensor array geometry (positions, orientations)    │
│ ✅ Owns: Transducer hardware configuration                  │
│ ✅ Owns: Data acquisition and recording                     │
│ ✅ Owns: Hardware control interfaces                        │
│ ❌ Does NOT own: Beamforming algorithms                     │
│ ✅ May contain: Thin wrappers that delegate to analysis     │
└────────────────────────┬────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────────┐
│ MATH LAYER: Linear algebra, numerical methods              │
│ ✅ Owns: Matrix operations (inversion, eigenvalue, SVD)    │
│ ✅ Owns: Sparse matrix implementations                      │
│ ✅ Owns: Iterative solvers                                  │
└─────────────────────────────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────────┐
│ CORE LAYER: Error handling, utilities                      │
│ ✅ Owns: Result types, error enums                          │
│ ✅ Owns: Generic utilities (not domain-specific)           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Accessor Pattern Implementation

**Principle:**  
> Lower layers should access higher-layer functionality through **well-defined accessor  
> interfaces** that encapsulate domain invariants and prevent implementation leakage.

**Example: Phased Array Hardware Wrapper**

```rust
// ✅ CORRECT: domain/source/transducers/phased_array/beamforming.rs

/// Hardware-specific beamforming wrapper for phased array control.
/// Delegates geometric calculations to canonical SSOT in analysis layer.
pub struct BeamformingCalculator {
    sound_speed: f64,
    frequency: f64,
}

impl BeamformingCalculator {
    /// Calculate focus delays by delegating to canonical implementation.
    pub fn calculate_focus_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        target: (f64, f64, f64),
    ) -> Vec<f64> {
        // Convert hardware-specific tuple format to canonical array format
        let positions_array: Vec<[f64; 3]> = element_positions
            .iter()
            .map(|&(x, y, z)| [x, y, z])
            .collect();

        // Delegate to SSOT (analysis layer)
        crate::analysis::signal_processing::beamforming::utils::delays::focus_phase_delays(
            &positions_array,
            [target.0, target.1, target.2],
            self.frequency,
            self.sound_speed,
        )
        .expect("Focus delay calculation failed")
        .to_vec()
    }
}
```

**Key Characteristics:**
1. ✅ Hardware-specific API (tuples instead of arrays)
2. ✅ Delegates computation to analysis layer
3. ✅ Maintains backward compatibility for hardware code
4. ✅ Zero duplication — single implementation

### 3.3 Vertical Separation Goals

**Deep Hierarchy Objectives:**

1. **Single Source of Truth (SSOT)**
   - ✅ Analysis layer contains **one and only one** implementation per algorithm
   - ✅ Domain layer **never** duplicates analysis algorithms
   - ✅ Shared primitives (e.g., steering vectors) live in lowest appropriate layer

2. **Downward-Only Dependencies**
   - ✅ Application → Analysis → Domain → Math → Core
   - ❌ NEVER: Analysis → Domain (for algorithms)
   - ✅ OK: Domain → Analysis (via accessor pattern for read-only access)

3. **Clear Ownership**
   - ✅ Analysis owns algorithms and their mathematical foundations
   - ✅ Domain owns hardware primitives and geometry
   - ✅ No shared ownership or ambiguous responsibility

4. **Minimal Interface Surface**
   - ✅ Public re-exports expose domain concepts, not implementation details
   - ✅ Internal modules remain private by default
   - ✅ Accessor methods provide controlled access, not full type exposure

---

## 4. Proposed Architecture

### 4.1 Target Layer Structure

```text
src/
├─ analysis/signal_processing/beamforming/        [SSOT - ALL ALGORITHMS]
│  ├─ mod.rs                                       Public API, trait definitions
│  ├─ traits.rs                                    Beamformer traits
│  │
│  ├─ adaptive/                                    Adaptive beamforming
│  │  ├─ mod.rs                                    AdaptiveBeamformer trait
│  │  ├─ mvdr.rs                                   MinimumVariance (Capon/MVDR)
│  │  └─ subspace.rs                               MUSIC, EigenspaceMV
│  │
│  ├─ time_domain/                                 Time-domain methods
│  │  ├─ mod.rs                                    Public API
│  │  ├─ das.rs                                    Delay-and-Sum
│  │  └─ delay_reference.rs                        Delay reference policy
│  │
│  ├─ narrowband/                                  Frequency-domain (to be migrated)
│  │  ├─ mod.rs                                    Public API
│  │  ├─ capon.rs                                  Narrowband Capon spectrum
│  │  ├─ snapshots/                                Snapshot extraction
│  │  └─ steering.rs                               Narrowband steering vectors
│  │
│  ├─ covariance/                                  Covariance estimation (SSOT)
│  │  └─ mod.rs                                    Sample, F-B, spatial smoothing
│  │
│  ├─ utils/                                       Shared utilities (SSOT)
│  │  ├─ mod.rs                                    Windows, steering, interpolation
│  │  ├─ delays.rs                                 Delay calculations (SSOT)
│  │  └─ sparse.rs                                 Sparse matrix utils
│  │
│  ├─ neural/                                      Neural/ML beamforming
│  │  ├─ mod.rs                                    Public API
│  │  ├─ beamformer.rs                             Neural beamformer
│  │  ├─ pinn/                                     PINN-based beamforming
│  │  └─ distributed/                              Distributed processing
│  │
│  ├─ experimental/                                Research-grade algorithms
│  │  └─ mod.rs                                    Experimental features
│  │
│  └─ test_utilities.rs                            Shared test utilities
│
├─ domain/sensor/                                  [HARDWARE PRIMITIVES ONLY]
│  ├─ mod.rs                                       Sensor geometry API
│  ├─ grid_sensor.rs                               Grid-based sensors
│  ├─ linear_array.rs                              Linear array geometry
│  ├─ matrix_array.rs                              Matrix array geometry
│  └─ beamforming/                                 ⚠️ TO BE REMOVED (deprecated)
│     └─ mod.rs                                    Deprecation notices only
│
├─ domain/source/transducers/phased_array/         [HARDWARE WRAPPERS]
│  ├─ mod.rs                                       Phased array API
│  ├─ transducer.rs                                Hardware configuration
│  └─ beamforming.rs                               ✅ KEEP: Hardware wrapper
│     └─ BeamformingCalculator                     (delegates to analysis layer)
│
└─ math/linear_algebra/                            [MATHEMATICAL PRIMITIVES]
   ├─ sparse/                                      Sparse matrix operations
   ├─ eigenvalue.rs                                Eigenvalue solvers
   └─ ...                                          Other linear algebra
```

### 4.2 Public API Design

**Analysis Layer Public API** (`analysis::signal_processing::beamforming`):

```rust
// Traits
pub use traits::{
    Beamformer,
    AdaptiveBeamformer,
    TimeDomainBeamformer,
    FrequencyDomainBeamformer,
};

// Adaptive beamforming
pub use adaptive::{
    MinimumVariance,        // MVDR/Capon
    MUSIC,                  // Multiple Signal Classification
    EigenspaceMV,           // Eigenspace Minimum Variance
};

// Time-domain beamforming
pub use time_domain::{
    delay_and_sum,          // DAS function
    DelayReference,         // Delay reference policy
    relative_delays_s,      // Relative delay calculation
};

// Covariance estimation
pub use covariance::{
    estimate_sample_covariance,
    estimate_forward_backward_covariance,
};

// Utilities
pub use utils::{
    focused_steering_vector,
    plane_wave_steering_vector,
    hamming_window,
    blackman_window,
};

// Neural beamforming (feature-gated)
#[cfg(feature = "pinn")]
pub use neural::{
    NeuralBeamformer,
    PINNBeamformingProcessor,
};
```

**Domain Layer Public API** (`domain::sensor`):

```rust
// Sensor geometry (primitives only)
pub use grid_sensor::{GridSensorSet, SensorPosition};
pub use linear_array::LinearArrayGeometry;
pub use matrix_array::MatrixArrayGeometry;

// ❌ NO beamforming algorithms
// ❌ NO signal processing utilities
```

**Source Layer Public API** (`domain::source::transducers::phased_array`):

```rust
// Hardware control wrapper
pub use beamforming::{
    BeamformingCalculator,   // ✅ Hardware wrapper (delegates to analysis)
    BeamformingMode,         // Focus, Steer, PlaneWave, Custom
};

// Transducer hardware
pub use transducer::{PhasedArrayTransducer, ElementConfig};
```

### 4.3 Migration Target State

**Before (Current - Incorrect):**

```text
Consumers
   ↓ import
domain::sensor::beamforming::adaptive::MinimumVariance   [DEPRECATED]
   ↓ duplicates
analysis::signal_processing::beamforming::adaptive::MinimumVariance   [CANONICAL]

Problem: Two implementations, unclear which is correct
```

**After (Target - Correct):**

```text
Consumers
   ↓ import
analysis::signal_processing::beamforming::adaptive::MinimumVariance   [SSOT]
   ↑ accessed by (accessor pattern)
domain::source::phased_array::BeamformingCalculator   [Hardware wrapper]

Solution: Single implementation, clear ownership
```

---

## 5. Migration Strategy

### 5.1 Phased Migration Approach

**Phase 0: Preparation** (✅ Complete)
- [x] Create canonical module structure
- [x] Migrate core algorithms (DAS, MVDR, MUSIC)
- [x] Establish SSOT for covariance and utilities
- [x] Add deprecation notices to old location

**Phase 1: Complete Canonical Implementation** (Current Sprint)
- [ ] Migrate narrowband algorithms from `domain::sensor::beamforming::narrowband`
- [ ] Migrate remaining adaptive algorithms (Robust Capon, source estimation)
- [ ] Migrate configuration types to analysis layer
- [ ] Migrate high-level processors to analysis layer
- [ ] Validate all algorithms against literature references

**Phase 2: Update Internal Consumers** (Sprint N+1)
- [ ] Update analysis-layer circular dependencies
- [ ] Update domain-layer consumers
- [ ] Update test suites
- [ ] Validate zero regressions

**Phase 3: Update External Consumers** (Sprint N+2)
- [ ] Update examples to use canonical imports
- [ ] Update benchmarks to use canonical imports
- [ ] Update documentation
- [ ] Publish migration guide

**Phase 4: Deprecation & Removal** (Sprint N+3)
- [ ] Add `#[deprecated]` attributes to all re-exports
- [ ] Add compiler warnings with migration instructions
- [ ] Schedule removal for version 3.0.0
- [ ] Final validation

### 5.2 Backward Compatibility Strategy

**Compatibility Facade Pattern:**

```rust
// domain/sensor/beamforming/mod.rs (DEPRECATED)

#![deprecated(
    since = "2.15.0",
    note = "Use `analysis::signal_processing::beamforming` instead. \
            See docs/refactor/BEAMFORMING_MIGRATION_GUIDE.md for details."
)]

//! ⚠️ DEPRECATED: This module will be removed in version 3.0.0
//!
//! All beamforming algorithms have been moved to:
//! [`crate::analysis::signal_processing::beamforming`]

// Re-export from canonical location for backward compatibility
#[deprecated(since = "2.15.0", note = "Use `analysis::signal_processing::beamforming::adaptive::MinimumVariance`")]
pub use crate::analysis::signal_processing::beamforming::adaptive::MinimumVariance;

#[deprecated(since = "2.15.0", note = "Use `analysis::signal_processing::beamforming::time_domain::delay_and_sum`")]
pub use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum;

// ... (re-export all public items with deprecation notices)
```

**Benefits:**
1. ✅ Existing code continues to work (no immediate breakage)
2. ✅ Compiler warnings guide users to new location
3. ✅ Zero duplication (re-exports point to canonical implementation)
4. ✅ Clear migration timeline (remove in 3.0.0)

### 5.3 Risk Mitigation

**Risk R1: Breaking Consumer Code**

- **Mitigation:** Maintain compatibility facade for 2-3 minor versions
- **Validation:** Automated tests verify facade preserves API compatibility
- **Timeline:** Announce deprecation in release notes, blogs, documentation

**Risk R2: Performance Regression**

- **Mitigation:** Comprehensive benchmarking before/after migration
- **Validation:** Run `cargo bench` on critical paths (DAS, MVDR, covariance)
- **Acceptance Criteria:** <5% performance change, zero algorithmic changes

**Risk R3: Algorithm Divergence**

- **Mitigation:** Property-based testing to verify mathematical equivalence
- **Validation:** Cross-validate old vs new implementation on test suite
- **Acceptance Criteria:** 100% test compatibility (identical outputs)

**Risk R4: Circular Dependencies**

- **Mitigation:** Bottom-up migration (math → analysis → domain)
- **Validation:** `cargo check` enforces acyclic dependency graph
- **Acceptance Criteria:** Zero circular dependencies in final state

---

## 6. Implementation Plan

### 6.1 Sprint Breakdown

**Sprint 1: Narrowband Migration** (Est. 12-16 hours)

**Tasks:**
1. Migrate `domain::sensor::beamforming::narrowband::capon.rs` → `analysis::.../narrowband/capon.rs`
2. Migrate snapshot extraction utilities
3. Migrate narrowband steering vector implementations
4. Validate against existing test suite (100% pass rate required)
5. Add integration tests for narrowband algorithms
6. Update internal consumers (8 files)

**Deliverables:**
- ✅ Canonical narrowband module complete
- ✅ All tests passing (zero regressions)
- ✅ Performance benchmarks validated (<5% change)

---

**Sprint 2: Configuration & High-Level Types** (Est. 8-10 hours)

**Tasks:**
1. Migrate `BeamformingConfig` types to analysis layer
2. Migrate `BeamformingProcessor` to analysis layer (rename to `BeamformingPipeline`)
3. Migrate `BeamformingMetrics` types
4. Remove circular dependencies (analysis → domain)
5. Update configuration documentation

**Deliverables:**
- ✅ Configuration types in correct layer
- ✅ Zero circular dependencies
- ✅ High-level processors in analysis layer

---

**Sprint 3: Internal Consumer Updates** (Est. 10-14 hours)

**Tasks:**
1. Update `analysis::signal_processing::beamforming::neural::pinn::processor.rs` imports
2. Update `analysis::signal_processing::beamforming::neural::types.rs` imports
3. Update domain-layer consumers (15 files)
4. Update test suites (30 files)
5. Validate zero regressions (full test suite)

**Deliverables:**
- ✅ Zero uses of deprecated `domain::sensor::beamforming` in internal code
- ✅ All tests passing
- ✅ Clean `cargo clippy` run

---

**Sprint 4: External Consumer Updates** (Est. 6-8 hours)

**Tasks:**
1. Update `benches/opast_benchmarks.rs`
2. Update `examples/real_time_3d_beamforming.rs`
3. Update documentation examples
4. Publish `BEAMFORMING_MIGRATION_GUIDE.md`
5. Update README with new import paths

**Deliverables:**
- ✅ Examples use canonical imports
- ✅ Benchmarks use canonical imports
- ✅ Migration guide published

---

**Sprint 5: Deprecation & Removal** (Est. 4-6 hours)

**Tasks:**
1. Convert `domain::sensor::beamforming` to pure re-export facade
2. Add `#[deprecated]` attributes to all items
3. Add compiler warnings with clear migration instructions
4. Schedule removal for version 3.0.0
5. Update CHANGELOG.md

**Deliverables:**
- ✅ Deprecation warnings active
- ✅ Compatibility facade in place
- ✅ Removal timeline documented

---

### 6.2 Task Prioritization

**Priority Matrix:**

| Task | Impact | Effort | Priority | Sprint |
|------|--------|--------|----------|--------|
| Narrowband migration | High | High | P0 | Sprint 1 |
| Configuration migration | High | Medium | P0 | Sprint 2 |
| Remove circular deps | Critical | Medium | P0 | Sprint 2 |
| Update internal consumers | High | High | P1 | Sprint 3 |
| Update benchmarks | Medium | Low | P1 | Sprint 4 |
| Update examples | Medium | Low | P1 | Sprint 4 |
| Add deprecation notices | Low | Low | P2 | Sprint 5 |

### 6.3 Validation Checkpoints

**Checkpoint C1: Algorithm Equivalence** (After Sprint 1)
- [ ] Run full test suite on canonical implementations
- [ ] Cross-validate against deprecated implementations (property-based tests)
- [ ] Acceptance: 100% mathematical equivalence (within floating-point tolerance)

**Checkpoint C2: Performance Validation** (After Sprint 1, 4)
- [ ] Run `cargo bench --all-features` on critical paths
- [ ] Compare before/after performance
- [ ] Acceptance: <5% performance change, no regressions

**Checkpoint C3: Dependency Validation** (After Sprint 2, 3)
- [ ] Run `cargo check --all-features`
- [ ] Verify zero circular dependencies
- [ ] Acceptance: Clean build, downward-only dependencies

**Checkpoint C4: API Compatibility** (After Sprint 4)
- [ ] Validate deprecated facade preserves API
- [ ] Run examples and benchmarks
- [ ] Acceptance: Zero breaking changes for users

---

## 7. Validation & Testing

### 7.1 Mathematical Verification

**Property-Based Testing Strategy:**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_mvdr_equivalence_canonical_vs_deprecated(
        n_sensors in 4..16usize,
        signal_power in 1.0..10.0f64,
        noise_power in 0.01..0.1f64,
    ) {
        // Generate test covariance matrix
        let cov = generate_test_covariance(n_sensors, signal_power, noise_power);
        let steering = generate_steering_vector(n_sensors);

        // Canonical implementation (analysis layer)
        let weights_canonical = analysis::signal_processing::beamforming::adaptive::MinimumVariance::default()
            .compute_weights(&cov, &steering)
            .expect("Canonical MVDR failed");

        // Deprecated implementation (sensor layer - if still exists)
        let weights_deprecated = domain::sensor::beamforming::adaptive::MinimumVariance::default()
            .compute_weights(&cov, &steering)
            .expect("Deprecated MVDR failed");

        // Validate mathematical equivalence
        for (w_canon, w_deprec) in weights_canonical.iter().zip(weights_deprecated.iter()) {
            prop_assert!((w_canon - w_deprec).abs() < 1e-10);
        }
    }
}
```

**Test Coverage Requirements:**

| Algorithm | Unit Tests | Integration Tests | Property Tests | Benchmark |
|-----------|-----------|-------------------|----------------|-----------|
| DAS | ✅ | ✅ | ✅ | ✅ |
| MVDR | ✅ | ✅ | ✅ | ✅ |
| MUSIC | ✅ | ✅ | ✅ | ⚠️ Missing |
| Narrowband Capon | ⚠️ Pending | ⚠️ Pending | ❌ Missing | ❌ Missing |
| Covariance | ✅ | ✅ | ✅ | ✅ |
| Delays | ✅ | ✅ | ✅ | ⚠️ Missing |

**Action Items:**
- [ ] Add property-based tests for MUSIC
- [ ] Add benchmarks for MUSIC, delay calculations
- [ ] Complete narrowband test coverage

### 7.2 Performance Validation

**Benchmark Suite:**

```rust
// benches/beamforming_migration_validation.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::analysis::signal_processing::beamforming::adaptive::MinimumVariance;

fn bench_mvdr_canonical(c: &mut Criterion) {
    let n = 64;
    let cov = create_test_covariance(n);
    let steering = create_steering_vector(n);
    let mvdr = MinimumVariance::default();

    c.bench_function("mvdr_canonical_64_sensors", |b| {
        b.iter(|| {
            black_box(mvdr.compute_weights(
                black_box(&cov),
                black_box(&steering),
            ).unwrap())
        });
    });
}

criterion_group!(benches, bench_mvdr_canonical);
criterion_main!(benches);
```

**Performance Acceptance Criteria:**
- ✅ DAS: <5% change vs baseline
- ✅ MVDR: <5% change vs baseline
- ✅ Covariance estimation: <10% change (acceptable for improved correctness)
- ✅ Memory allocation: Zero increase (maintain zero-copy where possible)

### 7.3 Regression Testing

**Regression Test Strategy:**

1. **Golden Master Tests:**
   - Capture outputs from current deprecated implementation
   - Validate canonical implementation produces identical outputs
   - Store golden outputs in version control

2. **Integration Tests:**
   - End-to-end beamforming pipelines
   - Validate image quality metrics (FWHM, contrast, SNR)
   - Compare against reference implementations (MATLAB, Field II)

3. **Continuous Integration:**
   - Run full test suite on every commit
   - Block merges if tests fail
   - Track performance trends over time

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk ID | Description | Probability | Impact | Mitigation |
|---------|-------------|-------------|--------|------------|
| **TR1** | Algorithm divergence during migration | Medium | High | Property-based cross-validation |
| **TR2** | Performance regression | Low | Medium | Comprehensive benchmarking |
| **TR3** | Breaking API changes | Low | High | Compatibility facade + deprecation |
| **TR4** | Circular dependency deadlock | Medium | Critical | Bottom-up migration strategy |
| **TR5** | Test coverage gaps | Medium | Medium | Add missing tests before migration |

### 8.2 Architectural Risks

| Risk ID | Description | Probability | Impact | Mitigation |
|---------|-------------|-------------|--------|------------|
| **AR1** | Incomplete SSOT (missed duplicates) | Medium | High | Automated duplication detection |
| **AR2** | Leaky abstractions in hardware wrappers | Low | Medium | Code review, accessor pattern enforcement |
| **AR3** | Unclear ownership boundaries | Low | High | Document layer responsibilities in ADR |
| **AR4** | Future code drift (re-duplication) | Medium | High | CI checks for layer violations |

### 8.3 Project Risks

| Risk ID | Description | Probability | Impact | Mitigation |
|---------|-------------|-------------|--------|------------|
| **PR1** | Scope creep (over-refactoring) | High | Medium | Strict sprint boundaries, focus on SSOT |
| **PR2** | User migration friction | Medium | High | Clear migration guide, deprecation timeline |
| **PR3** | Incomplete migration (orphaned code) | Low | High | Automated checks for deprecated usage |

---

## 9. Success Criteria

### 9.1 Architectural Goals

**AG1: Single Source of Truth**
- ✅ Zero duplicate algorithm implementations
- ✅ Analysis layer contains all beamforming algorithms
- ✅ Domain layer contains only hardware primitives
- ✅ Validation: `grep` for duplicate function names returns zero cross-layer matches

**AG2: Clean Layer Separation**
- ✅ Analysis layer does NOT depend on domain layer for algorithms
- ✅ Domain layer MAY access analysis layer via accessor pattern
- ✅ Downward-only dependencies (no cycles)
- ✅ Validation: `cargo check` succeeds, dependency graph is acyclic

**AG3: Minimal Public API Surface**
- ✅ `domain::sensor` exports geometry primitives only
- ✅ `analysis::signal_processing::beamforming` exports algorithms only
- ✅ Clear separation between stable API and internal utilities
- ✅ Validation: Public API documented, internal modules private by default

### 9.2 Quality Metrics

**QM1: Test Coverage**
- ✅ 100% of migrated algorithms have unit tests
- ✅ 100% of migrated algorithms have integration tests
- ✅ Property-based tests for critical algorithms (DAS, MVDR, MUSIC)
- ✅ Validation: `cargo tarpaulin` shows ≥95% line coverage for beamforming module

**QM2: Performance**
- ✅ Zero algorithmic changes (maintain mathematical equivalence)
- ✅ <5% performance change on critical paths
- ✅ Zero memory allocation increases
- ✅ Validation: `cargo bench` reports meet acceptance criteria

**QM3: Documentation**
- ✅ Migration guide published and complete
- ✅ Rustdoc coverage 100% for public API
- ✅ Examples updated to canonical imports
- ✅ ADR updated with architectural decisions

### 9.3 Validation Checklist

**Pre-Migration:**
- [ ] All property-based tests written and passing
- [ ] Baseline benchmarks recorded
- [ ] Golden master outputs captured

**During Migration:**
- [ ] Each sprint deliverable validated against acceptance criteria
- [ ] Checkpoints passed before proceeding to next sprint
- [ ] Zero regressions introduced

**Post-Migration:**
- [ ] Full test suite passes (867 tests)
- [ ] `cargo clippy -- -D warnings` passes
- [ ] Performance benchmarks meet acceptance criteria (<5% change)
- [ ] Zero uses of deprecated imports in internal code
- [ ] Documentation complete and accurate
- [ ] Deprecation notices active
- [ ] Migration guide reviewed and published

**Final Validation:**
- [ ] Independent code review by second architect
- [ ] Run full CI pipeline on clean checkout
- [ ] Validate examples and benchmarks work correctly
- [ ] User acceptance testing (if applicable)

---

## Appendix A: File Inventory

### A.1 Files to Migrate (Domain → Analysis)

**Narrowband Module** (Priority: P0):
- `domain/sensor/beamforming/narrowband/capon.rs` → `analysis/.../narrowband/capon.rs`
- `domain/sensor/beamforming/narrowband/snapshots/mod.rs` → `analysis/.../narrowband/snapshots/mod.rs`
- `domain/sensor/beamforming/narrowband/steering_narrowband.rs` → `analysis/.../narrowband/steering.rs`

**Configuration Types** (Priority: P0):
- `domain/sensor/beamforming/config.rs` → `analysis/.../config.rs`
- `domain/sensor/beamforming/beamforming_3d.rs` → `analysis/.../algorithms/beamforming_3d.rs`

**High-Level Processors** (Priority: P1):
- `domain/sensor/beamforming/processor.rs` → `analysis/.../pipeline/processor.rs`

### A.2 Files to Keep (No Migration)

**Hardware Wrappers:**
- `domain/source/transducers/phased_array/beamforming.rs` ✅ (delegates to analysis layer)

**Sensor Geometry:**
- `domain/sensor/grid_sensor.rs` ✅
- `domain/sensor/linear_array.rs` ✅
- `domain/sensor/matrix_array.rs` ✅

### A.3 Files to Remove (Deprecated)

**After Migration Complete:**
- `domain/sensor/beamforming/adaptive/` (entire directory)
- `domain/sensor/beamforming/time_domain/` (entire directory)
- `domain/sensor/beamforming/covariance.rs`
- `domain/sensor/beamforming/steering.rs`
- `domain/sensor/beamforming/narrowband/` (after migration)
- `domain/sensor/beamforming/experimental/` (after migration)

**Keep as Compatibility Facade (Temporary):**
- `domain/sensor/beamforming/mod.rs` (re-exports with deprecation notices)

---

## Appendix B: Migration Examples

### B.1 Example Migration: MVDR Algorithm

**Before:**
```rust
// domain/sensor/beamforming/adaptive/algorithms/mvdr.rs

pub fn mvdr_weights(
    covariance: &Array2<Complex64>,
    steering: &Array1<Complex64>,
) -> Result<Array1<Complex64>> {
    // Implementation
}
```

**After:**
```rust
// analysis/signal_processing/beamforming/adaptive/mvdr.rs

pub struct MinimumVariance {
    pub diagonal_loading: f64,
}

impl AdaptiveBeamformer for MinimumVariance {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        // Implementation (canonical SSOT)
    }
}
```

**Compatibility Facade:**
```rust
// domain/sensor/beamforming/adaptive/algorithms/mvdr.rs (deprecated)

#[deprecated(since = "2.15.0", note = "Use `analysis::signal_processing::beamforming::adaptive::MinimumVariance`")]
pub fn mvdr_weights(
    covariance: &Array2<Complex64>,
    steering: &Array1<Complex64>,
) -> Result<Array1<Complex64>> {
    use crate::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
    let mvdr = MinimumVariance::default();
    mvdr.compute_weights(covariance, steering)
        .map_err(|e| anyhow::anyhow!("MVDR failed: {}", e))
}
```

### B.2 Example Migration: Consumer Update

**Before:**
```rust
// examples/real_time_3d_beamforming.rs

use kwavers::sensor::beamforming::{
    ApodizationWindow,
    BeamformingAlgorithm3D,
    BeamformingConfig3D,
    BeamformingProcessor3D,
};
```

**After:**
```rust
// examples/real_time_3d_beamforming.rs

use kwavers::analysis::signal_processing::beamforming::{
    ApodizationWindow,
    BeamformingAlgorithm3D,
    BeamformingConfig3D,
    BeamformingProcessor3D,
};
```

---

## Appendix C: References

### C.1 Related Documents

- `docs/adr.md` — Architectural Decision Records
- `docs/refactor/BEAMFORMING_MIGRATION_GUIDE.md` — User-facing migration guide
- `docs/refactor/DEEP_HIERARCHY_PRINCIPLES.md` — Architectural principles
- `docs/backlog.md` — Sprint planning and task tracking

### C.2 Literature References

- Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley-Interscience.
- Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis." *Proceedings of the IEEE*, 57(8), 1408-1418.
- Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation." *IEEE Trans. Antennas Propag.*, 34(3), 276-280.

### C.3 Code Review Checklist

Before marking migration complete:
- [ ] Zero duplicate implementations
- [ ] All tests passing (100% pass rate)
- [ ] Performance benchmarks meet acceptance criteria
- [ ] Documentation complete (Rustdoc + migration guide)
- [ ] Deprecation notices active
- [ ] Backward compatibility facade in place
- [ ] Independent code review completed
- [ ] CI pipeline green

---

**Document Status:** 🟢 Ready for Execution  
**Next Action:** Begin Sprint 1 — Narrowband Migration  
**Estimated Total Effort:** 40-54 hours (5-7 sprints)  
**Timeline:** 5-7 weeks (assuming 1 sprint per week)

---

*This analysis was prepared according to the Elite Mathematically-Verified Systems Architect persona, prioritizing architectural purity, mathematical correctness, and zero-tolerance for error masking.*