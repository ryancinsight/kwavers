# Kwavers Architecture Remediation Plan

**Prepared:** 2026-01-28  
**Status:** ACTIVE  
**Priority:** CRITICAL

---

## Executive Summary

The kwavers ultrasound/optics simulation library has undergone comprehensive architectural analysis revealing **critical issues** blocking clean compilation and violating proper layering. This document outlines the systematic remediation strategy to restore the codebase to production-ready quality.

### Current State
- **Build Status:** Clean (after E0753 fix)
- **Architectural Violations:** 5 critical + 15 medium issues
- **Dead Code:** 108 files with `#[allow(dead_code)]`
- **Duplicate Functionality:** 20+ implementations across layers
- **Incomplete Work:** 270+ TODO/FIXME comments

### Target State
- **Build Status:** 0 errors, 0 warnings
- **Architecture:** Fully compliant 8-layer hierarchy
- **Code Quality:** No dead code, no deprecations
- **Completion:** All TODO items addressed
- **Deployment Ready:** Production-quality codebase

### Timeline
**Phase 1 (Critical):** 1-2 days  
**Phase 2 (High Priority):** 3-5 days  
**Phase 3 (Medium Priority):** 1-2 weeks  
**Phase 4 (Ongoing):** 2-4 weeks  
**Total:** 3-5 weeks

---

## Phase 1: Critical Build Fixes (IMMEDIATE)

### ✅ Task 1.1: Fix E0753 Doc Comment Errors
**Status:** COMPLETED  
**Date:** 2026-01-28  
**Files Modified:** 1
- `src/analysis/mod.rs` - Converted inner doc comments to regular comments

**Verification:**
```bash
cargo check ✅ Clean
cargo build --lib ✅ Clean
```

### Task 1.2: Consolidate Duplicate Module Definitions
**Status:** IN PROGRESS  
**Effort:** 2-3 hours  
**Files Affected:** 6

#### Identified Duplicates

**1. Imaging Modules (CRITICAL)**

Location 1: `src/domain/imaging/` (Domain Layer 3)
```
ultrasound/
  mod.rs
  beamformer.rs
  config.rs
photoacoustic/
  mod.rs
  config.rs
ceus/
  mod.rs
elastography/
  mod.rs
hifu/
  mod.rs
```

Location 2: `src/clinical/imaging/` (Clinical Layer 7)
```
ultrasound/
  workflows/
    mod.rs
photoacoustic/
  workflows/
    mod.rs
ceus/
  workflows/
    mod.rs
elastography/
  workflows/
    mod.rs
hifu/
  workflows/
    mod.rs
```

**Issue:** Both layers define same modality types with partial duplication

**Solution:**
1. Keep canonical definitions in `domain/imaging/` (Layer 3 - domain model)
2. Move workflow-specific code to `clinical/imaging/` (Layer 7 - application)
3. Create clear interfaces in `domain/` that `clinical/` extends
4. Remove duplicate type definitions

**Action Items:**
- [ ] Audit `domain/imaging/` for minimal domain types
- [ ] Audit `clinical/imaging/` for workflow implementations
- [ ] Create separation: `domain/imaging/` = types, `clinical/imaging/` = workflows
- [ ] Update imports in all dependent files
- [ ] Run tests to verify

**Implementation Strategy:**

Step 1: Unify Ultrasound Module
```rust
// src/domain/imaging/ultrasound/mod.rs (CANONICAL - MINIMAL)
pub struct UltrasoundConfig {
    pub frequency: f64,
    pub aperture_size: f64,
    pub num_elements: usize,
}

// src/clinical/imaging/ultrasound/workflows/mod.rs (WORKFLOWS)
pub struct UltrasoundWorkflow {
    config: domain::imaging::ultrasound::UltrasoundConfig,
    // Clinical-specific fields
}
impl UltrasoundWorkflow {
    pub fn execute(&self) { /* workflow logic */ }
}
```

Step 2: Apply to all modalities
- Photoacoustic
- CEUS (Contrast-Enhanced Ultrasound)
- Elastography
- HIFU (High-Intensity Focused Ultrasound)

### Task 1.3: Fix Duplicate Imports
**Status:** PENDING  
**Effort:** 1 hour  
**Files Affected:** 1

**File:** `src/clinical/imaging/workflows/mod.rs`

**Issues:**
```rust
// Line 5
use crate::core::error::KwaversResult;

// Line 248 (duplicate)
use crate::core::error::KwaversResult;
```

**Solution:**
- [ ] Remove duplicate import on line 248
- [ ] Consolidate all imports at top of file
- [ ] Run `cargo clippy` to verify

---

## Phase 2: High-Priority Architectural Fixes (3-5 days)

### Task 2.1: Fix Solver→Analysis Reverse Dependency
**Status:** PENDING  
**Severity:** HIGH  
**Effort:** 3-4 hours  
**Files Affected:** 2

#### Problem
**File:** `src/solver/inverse/pinn/ml/beamforming_provider.rs`

```rust
// Lines 1-20 (VIOLATION)
use crate::analysis::signal_processing::beamforming::neural::pinn_interface::{
    ModelInfo, PinnBeamformingConfig, PinnBeamformingProvider,
    TrainingMetrics, UncertaintyConfig,
};
```

**Issue:**
- Layer 4 (Solver) imports from Layer 6 (Analysis)
- Violates unidirectional dependency rule
- Creates circular dependency risk

**Correct Architecture:**
```
Layer 8 (Infrastructure)
    ↓
Layer 7 (Clinical)
    ↓
Layer 6 (Analysis)  ← Should import from Solver/Physics
    ↓
Layer 5 (Simulation)
    ↓
Layer 4 (Solver)    ← Should NOT import from Analysis
    ↓
Layer 3 (Domain)
    ↓
Layer 2 (Physics)
    ↓
Layer 1 (Math)
    ↓
Layer 0 (Core)
```

#### Solution: Extract Solver-Agnostic Interface

**Step 1: Create abstract interface in Solver layer**
```rust
// src/solver/inverse/pinn/interface.rs (NEW)
pub trait PinnBeamformingModel: Send + Sync {
    fn infer(&self, input: &Array3<f64>) -> KwaversResult<Array3<f64>>;
    fn get_metadata(&self) -> ModelMetadata;
}

pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub input_shape: (usize, usize, usize),
}
```

**Step 2: Move Analysis-specific code to Analysis layer**
```rust
// src/analysis/signal_processing/beamforming/neural/pinn_interface.rs
use crate::solver::inverse::pinn::interface::PinnBeamformingModel;

pub struct PinnBeamformingProvider {
    model: Box<dyn PinnBeamformingModel>,
}
```

**Step 3: Update imports**
```rust
// src/solver/inverse/pinn/ml/beamforming_provider.rs
use crate::solver::inverse::pinn::interface::{
    ModelMetadata, PinnBeamformingModel,
};
// ✅ NO imports from analysis/
```

**Implementation Checklist:**
- [ ] Create `src/solver/inverse/pinn/interface.rs`
- [ ] Define abstract `PinnBeamformingModel` trait
- [ ] Move analysis-specific types to analysis layer
- [ ] Update all imports
- [ ] Run tests
- [ ] Verify no reverse dependencies

### Task 2.2: Move Localization from Domain to Analysis
**Status:** PENDING  
**Severity:** HIGH  
**Effort:** 4-5 hours  
**Files Affected:** 10+

#### Problem
**Files in wrong layer:**
- `src/domain/sensor/localization/mod.rs` ← Should be in analysis
- `src/domain/sensor/localization/multilateration.rs`
- `src/domain/sensor/localization/music.rs`
- `src/domain/sensor/localization/array/mod.rs`

**Issue:**
- Localization is post-processing/analysis, not domain model
- Domain should define **what** (sensor positions), not **how** (localization algorithms)
- Analysis should define algorithms for extracting positions

**Correct Structure:**
```
Domain Layer (3):
  sensor/
    array/mod.rs              ← Sensor positions, types
    
Analysis Layer (6):
  signal_processing/
    localization/
      multilateration.rs      ← Algorithms
      music.rs
      mod.rs
```

#### Solution: Move and Update References

**Step 1: Create analysis/signal_processing/localization/**
```rust
// src/analysis/signal_processing/localization/mod.rs
pub mod multilateration;
pub mod music;
pub mod position;  // Type for position (minimal)
```

**Step 2: Remove from domain (with deprecation)**
```rust
// src/domain/sensor/localization/mod.rs
#![deprecated(
    since = "3.2.0",
    note = "Use analysis::signal_processing::localization instead. Moved to Analysis layer for proper architectural separation."
)]
pub use crate::analysis::signal_processing::localization::*;
```

**Step 3: Update all imports**
Files importing from `domain::sensor::localization`:
- `src/analysis/signal_processing/localization/beamforming_search.rs`
- `src/analysis/signal_processing/localization/music.rs`
- `src/clinical/imaging/workflows/neural/ai_beamforming_processor.rs`
- `src/domain/sensor/beamforming/sensor_beamformer.rs`

**Implementation Checklist:**
- [ ] Create target structure in analysis layer
- [ ] Copy localization implementations
- [ ] Update all imports in dependent files
- [ ] Add deprecation notices in domain
- [ ] Run full test suite
- [ ] Verify no domain→analysis dependencies remain

---

## Phase 3: Code Quality & Dead Code Cleanup (1-2 weeks)

### Task 3.1: Audit and Remove Dead Code
**Status:** PENDING  
**Effort:** 15-20 hours  
**Files Affected:** 108

#### Files with `#[allow(dead_code)]` (by layer):

**Analysis Layer (52 files):**
- `src/analysis/signal_processing/beamforming/` - 8 files
- `src/analysis/visualization/` - 12 files
- `src/analysis/ml/` - 10 files
- `src/analysis/performance/` - 8 files
- `src/analysis/imaging/` - 10 files
- Other: 4 files

**Solver Layer (31 files):**
- `src/solver/forward/fdtd/` - 8 files
- `src/solver/forward/pstd/` - 6 files
- `src/solver/inverse/pinn/` - 10 files
- GPU solvers: 7 files

**Physics Layer (15 files):**
- `src/physics/acoustics/` - 6 files
- `src/physics/thermal/` - 4 files
- `src/physics/optics/` - 3 files
- `src/physics/chemistry/` - 2 files

#### Strategy:

**For each file:**
1. Remove `#[allow(dead_code)]` pragma
2. Compile and identify actually unused functions
3. For truly unused code:
   - Check git history for reason
   - Add `#[deprecated]` with explanation if it's experimental
   - Remove if genuinely unused
4. For legitimately unused (test utilities, variants):
   - Add comment explaining why it's kept
   - Keep the `#[allow(dead_code)]`

**Example:**
```rust
// BEFORE
#[allow(dead_code)]
pub fn experimental_feature() { }

// AFTER - Truly unused
// (removed entirely)

// AFTER - Experimental variant
#[deprecated(since = "3.2.0", note = "Experimental feature")]
#[allow(dead_code)]
pub fn experimental_feature() {
    // Kept for future enhancement in ML-based beamforming
}
```

**Implementation Checklist:**
- [ ] Create spreadsheet tracking dead code audit
- [ ] Process files by priority (analysis → solver → physics)
- [ ] Document rationale for keeping code
- [ ] Remove truly dead code
- [ ] Add deprecation notices for experimental code
- [ ] Run clippy to verify

### Task 3.2: Address 270+ TODO/FIXME Comments
**Status:** PENDING  
**Effort:** 40-60 hours (varies by complexity)

#### Categorization by Priority:

**P0 - CRITICAL (must fix for v3.2):** ~20 items
**P1 - HIGH (should fix for v3.2):** ~50 items
**P2 - MEDIUM (nice to have for v3.2):** ~100 items
**P3 - LOW (future enhancement):** ~100 items

**High-Impact P0 Items:**

1. **API Implementation** (25+ TODOs)
   - Location: `src/infra/api/router.rs`
   - Issue: OAuth 2.0, rate limiting, versioning
   - Effort: 15-20 hours
   - Impact: Production readiness

2. **PINN Training** (8 TODOs)
   - Location: `src/solver/inverse/pinn/ml/beamforming_provider.rs`
   - Issue: Actual PINN inference not implemented
   - Effort: 10-15 hours
   - Impact: ML beamforming not functional

3. **GPU Pipeline** (12 TODOs)
   - Location: `src/solver/backend/gpu/pipeline.rs`
   - Issue: Incomplete WGSL compute shaders
   - Effort: 8-12 hours
   - Impact: GPU acceleration incomplete

**Action Plan:**

```bash
# Find all TODOs
grep -r "TODO\|FIXME" src --include="*.rs" | wc -l

# Group by severity
grep -r "TODO_AUDIT: P0" src --include="*.rs" | wc -l  # ~20
grep -r "TODO_AUDIT: P1" src --include="*.rs" | wc -l  # ~50
grep -r "TODO_AUDIT: P2" src --include="*.rs" | wc -l  # ~100
```

**Implementation Strategy:**
1. Prioritize P0 + P1 items (critical path)
2. Create issues for P2 + P3 items in CI/CD
3. Document all incomplete features
4. Add feature flags for incomplete work
5. Update roadmap with completion targets

---

## Phase 4: Deduplication & Consolidation (2-4 weeks)

### Task 4.1: Consolidate Beamforming Implementations
**Status:** PENDING**  
**Effort:** 20-30 hours  
**Files:** 20+ locations

#### Current Distribution:

| Location | Type | Status |
|----------|------|--------|
| `analysis/signal_processing/beamforming/neural/` | Neural (ML) | Canonical |
| `analysis/signal_processing/beamforming/three_dimensional/` | 3D DAS | Variant |
| `analysis/signal_processing/beamforming/adaptive/` | Adaptive | Variant |
| `analysis/signal_processing/beamforming/time_domain/` | Time-domain | Variant |
| `domain/sensor/beamforming/` | Legacy domain | OBSOLETE |
| `clinical/imaging/workflows/neural/` | Clinical app | Consumer |

#### Strategy: Canonical + Variants Pattern

**Canonical (authoritative):**
```rust
// src/analysis/signal_processing/beamforming/beamformer.rs
pub trait Beamformer {
    fn beamform(&self, data: &Array3<f64>) -> KwaversResult<Array3<f64>>;
}
```

**Variants (specialized implementations):**
```rust
// src/analysis/signal_processing/beamforming/variants/
pub mod neural;        // ML-based
pub mod three_d;       // 3D DAS
pub mod adaptive;      // Adaptive nulling
pub mod time_domain;   // Fast time-domain
pub mod standard;      // Classical Bartlett
```

**Action Items:**
- [ ] Define common `Beamformer` trait
- [ ] Move all to `analysis/signal_processing/beamforming/`
- [ ] Organize as canonical + variants
- [ ] Document variant selection criteria
- [ ] Remove duplicates from domain layer
- [ ] Update all imports

### Task 4.2: Consolidate FDTD Solver Variants
**Status:** PENDING**  
**Effort:** 15-20 hours  
**Files:** 8+ locations

#### Current FDTD Implementations:

```
Canonical:
  src/solver/forward/fdtd/solver.rs

Variants:
  src/solver/forward/fdtd/electromagnetic.rs        ← EM variant
  src/solver/forward/fdtd/plugin.rs                 ← Plugin wrapper
  src/gpu/fdtd.rs                                   ← GPU variant
  src/solver/forward/hybrid/fdtd_fem_coupling.rs   ← Coupled variant
  src/solver/inverse/pinn/ml/fdtd_reference.rs     ← Reference model
  src/math/simd.rs                                 ← SIMD accelerated
  src/solver/backend/gpu/pipeline.rs               ← GPU pipeline
```

#### Solution: Canonical Solver with Backend Selection

```rust
// src/solver/forward/fdtd/solver.rs (CANONICAL)
pub struct FDTDSolver {
    backend: BackendType,
    config: FDTDConfig,
}

// src/solver/forward/fdtd/backend.rs (NEW)
pub enum BackendType {
    CPU,
    CPUWithSIMD,
    GPU,
}

impl FDTDSolver {
    pub fn execute(&self) -> KwaversResult<SimulationResult> {
        match self.backend {
            BackendType::CPU => self.solve_cpu(),
            BackendType::CPUWithSIMD => self.solve_simd(),
            BackendType::GPU => self.solve_gpu(),
        }
    }
}
```

**Action Items:**
- [ ] Define backend selection interface
- [ ] Consolidate into single solver
- [ ] Create clear variant documentation
- [ ] Remove scattered implementations
- [ ] Add backend auto-selection logic

---

## Phase 5: Comprehensive Testing & Validation (Parallel)

### Task 5.1: Build Clean Verification
**Status:** PENDING  
**Requirement:** 0 errors, 0 warnings

```bash
cargo check                # ✅ Must pass
cargo build --lib         # ✅ Must pass with 0 warnings
cargo build --release     # ✅ Must pass
cargo clippy --lib        # ✅ Must pass (0 warnings)
cargo test --lib          # ✅ All tests must pass
```

### Task 5.2: Architecture Verification
**Status:** PENDING  

**Checklist:**
- [ ] No Layer 3 (Domain) → Layer 6 (Analysis) dependencies
- [ ] No Layer 4 (Solver) → Layer 6+ (Analysis+) dependencies
- [ ] No circular imports at module level
- [ ] All deprecated types have migration path
- [ ] Clear documentation for all APIs

### Task 5.3: Code Coverage
**Status:** PENDING  

**Target:** >80% coverage on critical paths
- Physics layer: 90%+
- Solver layer: 85%+
- Analysis layer: 80%+
- Clinical layer: 70%+

---

## Implementation Timeline

### Week 1 (Days 1-5)
- **Day 1-2:** Phase 1 critical fixes + Phase 2.1 (reverse dependency)
- **Day 3-4:** Phase 2.2 (move localization)
- **Day 5:** Verification & testing

### Week 2 (Days 6-10)
- **Days 6-7:** Phase 3.1 dead code audit
- **Days 8-10:** Phase 3.2 high-priority TODOs

### Week 3-4 (Days 11-20)
- **Days 11-15:** Phase 4.1 beamforming consolidation
- **Days 16-20:** Phase 4.2 FDTD consolidation

### Week 5+ (Days 21+)
- Phase 3.2 remaining TODOs
- Phase 5 comprehensive testing
- Documentation updates

---

## Success Criteria

### Build Quality
✅ `cargo build --release` produces 0 errors  
✅ `cargo clippy --lib` produces 0 warnings  
✅ All 1,670+ tests passing  
✅ No `#[allow(dead_code)]` without documentation  

### Architecture
✅ No circular dependencies  
✅ Strict unidirectional layer dependencies  
✅ All cross-layer violations eliminated  
✅ Clear canonical patterns for variants  

### Code Quality
✅ All TODO_AUDIT items categorized  
✅ P0/P1 TODOs completed  
✅ Dead code removed or documented  
✅ Documentation complete  

### Deployment Readiness
✅ Production-quality build  
✅ Clean architecture  
✅ Comprehensive tests  
✅ Ready for external review  

---

## Risk Mitigation

### High-Risk Tasks
1. **Moving localization types**
   - Risk: Breaking existing code
   - Mitigation: Add deprecation period, clear migration guide
   
2. **Consolidating solvers**
   - Risk: Performance regression
   - Mitigation: Benchmark before/after, compare results

3. **Removing dead code**
   - Risk: Removing needed code
   - Mitigation: Check git history, add tests first

### Testing Strategy
- Unit tests for all changes
- Integration tests for workflow changes
- Regression tests before/after consolidation
- Example scripts to verify API compatibility

---

## References

### Architecture References
- K-Wave (GitHub: ucl-bug/k-wave) - MATLAB reference
- jWave (GitHub: ucl-bug/jwave) - Java implementation
- mSOUND (GitHub: m-SOUND/mSOUND) - MATLAB simulations
- Kranion (GitHub: jws2f/Kranion) - MRI-guided ultrasound

### Standards
- Clean Code principles (Martin, 2008)
- Domain-Driven Design (Evans, 2003)
- Layered Architecture best practices
- Rust API Guidelines (API-WG)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-28 | System | Initial remediation plan |

**Status:** ACTIVE - Ready for implementation  
**Next Review:** 2026-02-04 (end of Week 1)
