# Task 2.1: Beamforming Migration Assessment
## Remove Deprecated Beamforming Module - Pre-Flight Analysis

**Date:** 2024-12-19  
**Task ID:** Phase 1, Sprint 2, Task 2.1  
**Status:** 📋 PLANNING  
**Priority:** P1 (High)  
**Risk Level:** 🔴 HIGH  
**Estimated Effort:** 10-12 hours

---

## Executive Summary

The deprecated `domain/sensor/beamforming/` module contains **37 source files** with extensive functionality that has been **partially migrated** to `analysis/signal_processing/beamforming/` (29 files). However, **critical blockers exist** that prevent immediate removal:

1. **11 active import statements** remain in production code
2. **Core types still consumed**: `SteeringVector`, `BeamformingConfig`, `BeamformingProcessor`
3. **Feature-gated code dependencies**: AI integration, PINN, experimental features
4. **Partial migration status**: Not all functionality has been ported to canonical location

**Recommendation:** **DEFER complete removal** to Phase 2. Execute **incremental migration** of remaining consumers in current sprint.

---

## Current State Analysis

### Module Statistics

| Metric | Deprecated Path | Canonical Path | Status |
|--------|----------------|----------------|--------|
| **Location** | `src/domain/sensor/beamforming/` | `src/analysis/signal_processing/beamforming/` | Parallel |
| **File Count** | 37 files | 29 files | Canonical has less |
| **Deprecation Warnings** | Active | N/A | Warnings present |
| **Active Imports** | 11 found | Migrated consumers | Partial |
| **Core Types** | Still exported | Partially duplicated | Mixed |

### Active Import Analysis

#### Category 1: Within Analysis Module (Safe to Update)
```rust
// analysis/signal_processing/beamforming/neural/pinn/processor.rs
use crate::domain::sensor::beamforming::SteeringVector;

// analysis/signal_processing/beamforming/neural/types.rs
use crate::domain::sensor::beamforming::BeamformingConfig;
```
**Impact:** 🟢 LOW - Same module family, easy to migrate
**Action:** Create canonical types in analysis module or re-export from domain temporarily

#### Category 2: Domain Sensor Modules (Medium Coupling)
```rust
// domain/sensor/localization/beamforming_search/config.rs
use crate::domain::sensor::beamforming::{
    BeamformingCoreConfig, SteeringVectorMethod, TimeDomainDelayReference,
};

// domain/sensor/localization/beamforming_search/mod.rs
use crate::domain::sensor::beamforming::BeamformingProcessor;

// domain/sensor/passive_acoustic_mapping/beamforming_config.rs
use crate::domain::sensor::beamforming::BeamformingCoreConfig;

// domain/sensor/passive_acoustic_mapping/mod.rs
use crate::domain::sensor::beamforming::BeamformingProcessor;
```
**Impact:** 🟡 MEDIUM - Domain modules importing from domain (acceptable temporary state)
**Action:** Evaluate if these should remain in domain or move to analysis

#### Category 3: Clinical Workflows (Cross-Layer)
```rust
// clinical/imaging/workflows.rs
use crate::domain::sensor::beamforming::BeamformingConfig3D;

// infra/api/clinical_handlers.rs
use crate::domain::sensor::beamforming::ai_integration::{
    AIBeamformingConfig, AIEnhancedBeamformingProcessor,
};
```
**Impact:** 🟡 MEDIUM - High-level code depending on deprecated module
**Action:** Update to canonical location after ensuring types exist

#### Category 4: Self-References (Within Deprecated Module)
```rust
// domain/sensor/beamforming/ai_integration.rs
use crate::domain::sensor::beamforming::{BeamformingConfig, BeamformingProcessor};

// domain/sensor/beamforming/beamforming_3d.rs
use crate::domain::sensor::beamforming::config::BeamformingConfig;

// domain/sensor/beamforming/narrowband/capon.rs
use crate::domain::sensor::beamforming::covariance::CovarianceEstimator;
use crate::domain::sensor::beamforming::narrowband::snapshots::...;
use crate::domain::sensor::beamforming::narrowband::steering_narrowband::...;
use crate::domain::sensor::beamforming::{SteeringVector, SteeringVectorMethod};
```
**Impact:** 🔴 HIGH - Internal module cohesion; breaks if removed prematurely
**Action:** Must migrate entire module as atomic unit or maintain temporary exports

---

## Critical Blockers

### Blocker 1: Core Type Definitions Missing in Canonical Location

**Issue:** Key types used by multiple consumers are defined in deprecated module but not fully replicated in canonical location.

**Missing/Incomplete Types:**
- `BeamformingConfig` - Core configuration type
- `BeamformingProcessor` - Main processor interface
- `SteeringVector` - Fundamental beamforming primitive
- `BeamformingConfig3D` - 3D-specific configuration
- `BeamformingCoreConfig` - Low-level configuration

**Evidence:**
```rust
// Deprecated location (still active)
pub use config::{BeamformingConfig, BeamformingCoreConfig};
pub use processor::BeamformingProcessor;
pub use steering::SteeringVector;
pub use beamforming_3d::BeamformingConfig3D;
```

**Impact:** Cannot remove deprecated module until these types are available in canonical location.

**Resolution Required:**
1. Verify canonical module has equivalent types or migrate them
2. Update all consumers to use canonical types
3. Add temporary re-exports in deprecated module pointing to canonical location

---

### Blocker 2: Feature-Gated Dependencies

**Issue:** AI integration and PINN features depend on types only in deprecated module.

**Feature-Gated Imports:**
```rust
#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
pub mod ai_integration;

#[cfg(feature = "experimental_neural")]
pub mod experimental;

pub use ai_integration::{
    AIBeamformingConfig, AIBeamformingResult, AIEnhancedBeamformingProcessor,
    ClinicalDecisionSupport, DiagnosisAlgorithm, FeatureExtractor, RealTimeWorkflow,
};
```

**Impact:** Removing module breaks optional features.

**Resolution Required:**
1. Verify AI integration is migrated to `analysis/signal_processing/beamforming/neural/`
2. Check if `experimental` module is duplicated or only in deprecated location
3. Update feature-gated code to use canonical paths

---

### Blocker 3: Narrowband Algorithms Still in Deprecated Location

**Issue:** Complex narrowband beamforming (Capon, MUSIC, MVDR) appears to remain in deprecated location with incomplete migration.

**Deprecated Exports:**
```rust
pub use narrowband::{
    capon_spatial_spectrum_point,
    capon_spatial_spectrum_point_complex_baseband,
    extract_complex_baseband_snapshots,
    extract_narrowband_snapshots,
    CaponSpectrumConfig,
    NarrowbandSteering,
    NarrowbandSteeringVector,
    // ... many more
};
```

**Status Check Required:**
- Does `analysis/signal_processing/beamforming/narrowband/` have complete implementations?
- Are APIs equivalent or breaking changes required?

---

## Migration Strategy Options

### Option A: Complete Removal (Original Plan) ❌ NOT FEASIBLE NOW

**Pros:**
- Clean break, no technical debt
- Enforces correct architecture immediately

**Cons:**
- 🔴 Blocking issues prevent immediate execution
- 🔴 Requires verifying entire canonical module is feature-complete
- 🔴 Risk of breaking production code
- 🔴 Estimated effort exceeds sprint capacity (15-20 hours realistic)

**Verdict:** **REJECTED** - Too many blockers for current sprint.

---

### Option B: Incremental Migration (Recommended) ✅

**Phase 2.1a: Migrate Analysis Module Self-References (2-3 hours)**
- Update `analysis/signal_processing/beamforming/neural/pinn/processor.rs`
- Update `analysis/signal_processing/beamforming/neural/types.rs`
- Ensure canonical types exist or create them

**Phase 2.1b: Document Remaining Dependencies (1 hour)**
- Create comprehensive list of missing types in canonical module
- Document which deprecated types must be preserved/migrated
- Prioritize by usage frequency

**Phase 2.1c: Temporary Re-Export Strategy (2 hours)**
- Update deprecated `mod.rs` to re-export from canonical location where possible
- Maintain backward compatibility during transition period
- Add clear deprecation path in documentation

**Phase 2.1d: Clinical/Infra Updates (2-3 hours)**
- Update `clinical/imaging/workflows.rs` if types available
- Update `infra/api/clinical_handlers.rs` if types available
- Defer if types not yet migrated

**Total Effort:** 7-9 hours (fits in sprint)

**Pros:**
- ✅ Makes measurable progress
- ✅ Reduces import count from 11 to 4-6
- ✅ Fits within sprint capacity
- ✅ Low risk (incremental changes)

**Cons:**
- ⚠️ Does not eliminate module entirely
- ⚠️ Requires Phase 2 continuation

**Verdict:** **RECOMMENDED** - Pragmatic approach balancing progress and risk.

---

### Option C: Strategic Re-Export (Alternative) 🟡

**Approach:** Keep deprecated module as facade that re-exports from canonical location.

**Implementation:**
```rust
// domain/sensor/beamforming/mod.rs
//! ⚠️ DEPRECATED: This module is a compatibility facade.
//! All functionality has moved to `analysis::signal_processing::beamforming`.

#[deprecated(since = "2.15.0", note = "Use `analysis::signal_processing::beamforming` instead")]
pub use crate::analysis::signal_processing::beamforming::{
    adaptive, narrowband, time_domain, covariance,
    BeamformingConfig, BeamformingProcessor, SteeringVector,
};
```

**Pros:**
- ✅ Maintains backward compatibility indefinitely
- ✅ Zero consumer code changes required
- ✅ Clear deprecation path

**Cons:**
- ⚠️ Module directory still exists (not truly "removed")
- ⚠️ Re-export maintenance burden
- ⚠️ Doesn't enforce migration

**Verdict:** **FALLBACK** - Use if canonical module not feature-complete by Phase 2 end.

---

## Recommended Execution Plan

### Sprint 2 Task 2.1 (Revised Scope)

**Objective:** Reduce deprecated module usage by 50%+ without breaking changes.

#### Step 1: Pre-Flight Verification (1 hour)
```bash
# Verify canonical module feature completeness
find src/analysis/signal_processing/beamforming -name "*.rs" -exec grep "pub struct\|pub fn\|pub enum" {} \; | wc -l

# Check for core types
grep -r "pub struct BeamformingConfig\|pub struct BeamformingProcessor" src/analysis/signal_processing/beamforming/

# Verify narrowband algorithms
ls src/analysis/signal_processing/beamforming/narrowband/
```

**Decision Point:** If core types missing, execute Option C (re-export strategy) instead.

#### Step 2: Migrate Analysis Module Self-References (2-3 hours)

**File 1:** `analysis/signal_processing/beamforming/neural/pinn/processor.rs`
```diff
- use crate::domain::sensor::beamforming::SteeringVector;
+ use crate::analysis::signal_processing::beamforming::steering::SteeringVector;
```

**File 2:** `analysis/signal_processing/beamforming/neural/types.rs`
```diff
- use crate::domain::sensor::beamforming::BeamformingConfig;
+ use crate::analysis::signal_processing::beamforming::config::BeamformingConfig;
```

**Validation:**
```bash
cargo build --all-features
cargo test --lib analysis::signal_processing::beamforming
```

#### Step 3: Update Clinical Workflows (2-3 hours)

**File 3:** `clinical/imaging/workflows.rs`
```diff
- use crate::domain::sensor::beamforming::BeamformingConfig3D;
+ use crate::analysis::signal_processing::beamforming::config::BeamformingConfig3D;
```

**File 4:** `infra/api/clinical_handlers.rs`
```diff
- use crate::domain::sensor::beamforming::ai_integration::{
+ use crate::analysis::signal_processing::beamforming::neural::{
      AIBeamformingConfig, AIEnhancedBeamformingProcessor,
  };
```

**Validation:**
```bash
cargo build --all-features --lib
cargo clippy --all-features -- -W clippy::all
```

#### Step 4: Document Remaining Dependencies (1 hour)

Create `BEAMFORMING_MIGRATION_STATUS.md`:
- List of unmigrated types with usage counts
- Dependencies in domain/sensor/localization
- Dependencies in domain/sensor/passive_acoustic_mapping
- Recommended Phase 2 continuation plan

#### Step 5: Update Deprecation Documentation (1 hour)

Enhance `domain/sensor/beamforming/mod.rs`:
```rust
//! # Migration Status (as of v2.15.0)
//!
//! - ✅ Time-domain algorithms: Fully migrated
//! - ✅ Adaptive algorithms: Fully migrated
//! - ✅ Neural beamforming: Fully migrated
//! - ⚠️ Core config types: Partial (see BEAMFORMING_MIGRATION_STATUS.md)
//! - ⚠️ Domain sensor integrations: Deferred to Phase 2
```

---

## Success Criteria (Revised)

| Criterion | Target | Status |
|-----------|--------|--------|
| Reduce import count | 11 → ≤5 | 🔵 Pending |
| Zero breaking changes | Build passes | 🔵 Pending |
| Analysis module self-sufficient | No domain imports | 🔵 Pending |
| Documentation complete | Migration status doc | 🔵 Pending |
| Tests pass | All beamforming tests | 🔵 Pending |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Core types missing in canonical location | 🔴 HIGH | 🔴 HIGH | Verify first in Step 1; fallback to Option C |
| Breaking API changes required | 🟡 MEDIUM | 🔴 HIGH | Incremental migration; maintain compatibility |
| Feature-gated code breaks | 🟡 MEDIUM | 🟡 MEDIUM | Test with all features enabled |
| Domain sensor modules break | 🟢 LOW | 🟡 MEDIUM | Keep domain→domain imports temporarily valid |

---

## Rollback Plan

If Step 2 or 3 fails:
```bash
# Revert changes
git checkout HEAD -- src/analysis/signal_processing/beamforming/neural/
git checkout HEAD -- src/clinical/imaging/workflows.rs
git checkout HEAD -- src/infra/api/clinical_handlers.rs

# Verify build
cargo build --all-features
```

---

## Phase 2 Continuation Requirements

To **completely remove** `domain/sensor/beamforming/` in Phase 2:

1. **Migrate Core Types:**
   - `BeamformingConfig` → `analysis/.../config.rs`
   - `BeamformingProcessor` → `analysis/.../processor.rs`
   - `SteeringVector` → `analysis/.../steering.rs`

2. **Resolve Domain Dependencies:**
   - Evaluate `domain/sensor/localization` beamforming usage
   - Evaluate `domain/sensor/passive_acoustic_mapping` beamforming usage
   - Decision: Keep in domain OR move to analysis

3. **Feature-Gate Verification:**
   - Test with `--features experimental_neural`
   - Test with `--features pinn`
   - Ensure AI integration fully migrated

4. **Final Removal:**
   - Delete `src/domain/sensor/beamforming/` directory
   - Update all module references
   - Remove deprecation warnings (module gone)

**Estimated Phase 2 Effort:** 8-10 additional hours

---

## Recommendation to Stakeholders

**Proceed with Option B (Incremental Migration)** for Sprint 2 Task 2.1:
- Achievable within sprint capacity (7-9 hours vs 10-12 planned)
- Low risk (backward compatible changes)
- Measurable progress (50%+ import reduction)
- Sets up clean Phase 2 completion

**Defer complete removal** to Phase 2 Sprint 4:
- Requires canonical module feature verification
- Needs domain sensor architecture decision
- Higher complexity than initially estimated

---

**Assessment Status:** ✅ COMPLETE  
**Recommended Action:** Execute Incremental Migration (Option B)  
**Next Step:** Pre-flight verification (Step 1) to confirm canonical module readiness  
**Go/No-Go Decision Point:** After Step 1 verification results

---

**Prepared By:** Kwavers Refactoring Team  
**Date:** 2024-12-19  
**Review Required:** Architecture Committee sign-off on incremental approach