# Immediate Next Steps: Beamforming Architecture Remediation

**Status:** 🟢 Ready to Execute  
**Phase:** Sprint 1 Preparation  
**Priority:** P0 — Critical Architectural Issue  
**Estimated Effort:** 2-3 hours (preparation phase)  
**Owner:** Elite Mathematically-Verified Systems Architect  
**Date:** 2024-01-XX  

---

## Executive Summary

**Current State:**  
✅ Build is green (`cargo check --all-features` passes)  
✅ Architectural analysis complete (`BEAMFORMING_ARCHITECTURE_ANALYSIS.md`)  
✅ Sprint 1 plan documented (`SPRINT_1_NARROWBAND_MIGRATION.md`)  
✅ Canonical location established (`analysis::signal_processing::beamforming`)  

**Problem:**  
Beamforming code duplicated across 3 layers (sensor, source, analysis), violating deep vertical hierarchy goals.

**Immediate Goal:**  
Begin Sprint 1 — Narrowband Migration to establish complete SSOT in analysis layer.

---

## Immediate Action Plan (Next 2-3 Hours)

### Action 1: Source Inventory (45 min)

**Objective:** Document current narrowband implementation in domain layer

**Steps:**
1. Read and analyze `domain/sensor/beamforming/narrowband/mod.rs`
2. Inventory all public exports and functions
3. Document dependencies (what does it import?)
4. Identify consumer usage patterns
5. Create `NARROWBAND_SOURCE_INVENTORY.md`

**Commands:**
```bash
# Find all narrowband files
find src/domain/sensor/beamforming/narrowband -name "*.rs"

# Count lines of code
find src/domain/sensor/beamforming/narrowband -name "*.rs" -exec wc -l {} +

# Find all imports of narrowband module
grep -r "use.*sensor.*beamforming.*narrowband" src/ --include="*.rs"

# Find all consumers (direct usage)
grep -r "domain::sensor::beamforming::narrowband" src/ --include="*.rs"
```

**Deliverable:**
- `docs/refactor/NARROWBAND_SOURCE_INVENTORY.md` with:
  - Complete list of public functions and types
  - Dependency graph
  - Consumer list (prioritized)
  - Migration complexity assessment

---

### Action 2: Dependency Graph Analysis (30 min)

**Objective:** Determine migration order to avoid circular dependencies

**Steps:**
1. Map imports within narrowband module
2. Identify external dependencies (covariance, steering, utils)
3. Check if external dependencies already migrated
4. Determine bottom-up migration order
5. Document any circular dependency risks

**Analysis Questions:**
- Does narrowband depend on adaptive? (if yes, wait for adaptive completion)
- Does narrowband depend on time_domain? (if yes, already migrated ✅)
- Does narrowband depend on covariance? (if yes, already migrated ✅)
- Does narrowband depend on domain-specific types? (if yes, need to break coupling)

**Deliverable:**
- Dependency graph diagram (ASCII or Mermaid)
- Migration order plan with rationale
- Risk assessment for circular dependencies

---

### Action 3: Consumer Impact Analysis (30 min)

**Objective:** Identify all code that will break if we migrate narrowband

**Steps:**
1. Find all internal consumers (analysis, domain layers)
2. Find all external consumers (examples, benchmarks, tests)
3. Classify by migration priority (P0, P1, P2)
4. Estimate update effort for each consumer
5. Create consumer update plan

**Consumer Classification:**
- **P0 (Critical):** Analysis layer consumers (circular dependency risk)
- **P1 (High):** Domain layer consumers (architectural violation)
- **P2 (Medium):** Examples, benchmarks (public-facing)
- **P3 (Low):** Internal tests (easy to update)

**Deliverable:**
- Consumer list with:
  - File path
  - Usage pattern (what functions/types are used?)
  - Update priority
  - Estimated effort
  - Update strategy

---

### Action 4: Pre-Migration Validation (15 min)

**Objective:** Establish baseline metrics before migration

**Steps:**
1. Run full test suite and record pass rate
2. Run benchmarks (if any exist for narrowband)
3. Document current warnings/errors
4. Create git branch for migration work

**Commands:**
```bash
# Record test baseline
cargo test --all-features 2>&1 | tee baseline_tests.log

# Record benchmark baseline (if exists)
cargo bench --bench narrowband_beamforming 2>&1 | tee baseline_bench.log

# Record warnings
cargo clippy --all-features 2>&1 | tee baseline_clippy.log

# Create migration branch
git checkout -b refactor/narrowband-migration-sprint1
git add docs/refactor/BEAMFORMING_ARCHITECTURE_ANALYSIS.md
git add docs/refactor/SPRINT_1_NARROWBAND_MIGRATION.md
git add docs/refactor/IMMEDIATE_NEXT_STEPS.md
git commit -m "docs: Add beamforming architecture analysis and Sprint 1 plan"
```

**Deliverable:**
- Baseline metrics documented
- Clean git branch for migration work
- Rollback plan if migration fails

---

## Decision Points

### Decision D1: Migrate Now or Wait?

**Question:** Should we proceed with narrowband migration immediately, or complete other tasks first?

**Analysis:**
- ✅ **Proceed Now:** Narrowband is independent (no circular deps detected)
- ✅ **Proceed Now:** Canonical infrastructure ready (covariance, utils migrated)
- ✅ **Proceed Now:** No blocking dependencies identified
- ❌ **Wait:** Only if dependency analysis reveals circular dependencies

**Recommendation:** ✅ Proceed with narrowband migration (Sprint 1)

---

### Decision D2: Migrate Everything or Incremental?

**Question:** Should we migrate all narrowband algorithms at once, or one-by-one?

**Options:**

**Option A: Atomic Migration (All at Once)**
- Pros: Single commit, clear boundary, no partial state
- Cons: Higher risk, harder to rollback if issues found
- Effort: 6-8 hours continuous work

**Option B: Incremental Migration (File-by-File)**
- Pros: Lower risk, easier to validate, can pause if issues found
- Cons: Multiple commits, temporary inconsistent state
- Effort: 6-8 hours spread over multiple sessions

**Recommendation:** ✅ **Option B (Incremental)** — Safer for complex migration

**Migration Order:**
1. `capon.rs` (core algorithm, most critical)
2. `steering_narrowband.rs` (dependency for capon)
3. `snapshots/` (utility module)
4. Update module structure and re-exports

---

### Decision D3: Keep Deprecated Code or Remove Immediately?

**Question:** After migration, should we keep deprecated implementation or remove it?

**Options:**

**Option A: Keep as Compatibility Facade**
- Pros: Zero breaking changes for users, smooth transition
- Cons: Maintains duplication temporarily (1-2 minor versions)
- Timeline: Remove in version 3.0.0

**Option B: Remove Immediately**
- Pros: Zero duplication, clean codebase
- Cons: Breaking change for external users
- Timeline: Requires major version bump

**Recommendation:** ✅ **Option A (Compatibility Facade)** — User-friendly migration

**Implementation:**
```rust
// domain/sensor/beamforming/narrowband/mod.rs (after migration)

#![deprecated(since = "2.15.0", note = "Use analysis::signal_processing::beamforming::narrowband")]

// Re-export from canonical location
pub use crate::analysis::signal_processing::beamforming::narrowband::*;
```

---

## Risk Assessment

### Risk R1: Hidden Circular Dependencies

**Probability:** Low  
**Impact:** Critical (blocks compilation)

**Mitigation:**
- Dependency graph analysis (Action 2) will reveal cycles
- If found, break cycle by moving shared code to lower layer
- Bottom-up migration order prevents cycles

**Contingency Plan:**
- If circular dependency found during migration:
  1. Identify shared code causing cycle
  2. Move shared code to `utils` or `math` layer
  3. Update both modules to use shared implementation
  4. Resume migration

---

### Risk R2: Algorithm Behavioral Differences

**Probability:** Medium  
**Impact:** High (mathematical incorrectness)

**Mitigation:**
- Property-based tests validate mathematical equivalence
- Cross-validate against deprecated implementation
- Use existing test suite as oracle (both should pass)

**Contingency Plan:**
- If divergence found:
  1. Document difference
  2. Determine which is correct (literature validation)
  3. Fix incorrect implementation
  4. Add regression test
  5. Document change in CHANGELOG

---

### Risk R3: Performance Regression

**Probability:** Low  
**Impact:** Medium (affects real-time imaging)

**Mitigation:**
- Benchmark before/after migration
- Profile if regression detected
- Acceptance criteria: <5% change

**Contingency Plan:**
- If >5% regression:
  1. Profile with `cargo flamegraph`
  2. Optimize hot paths
  3. Document trade-off if correctness requires performance sacrifice
  4. Update ADR with decision rationale

---

## Success Criteria (Phase Completion)

### Preparation Phase (This Document)
- [ ] Source inventory complete (`NARROWBAND_SOURCE_INVENTORY.md`)
- [ ] Dependency graph documented
- [ ] Consumer impact analysis complete
- [ ] Baseline metrics recorded
- [ ] Git branch created
- [ ] Migration order determined
- [ ] Risk mitigation strategies defined

### Sprint 1 Completion (Next Phase)
- [ ] All narrowband algorithms migrated to canonical location
- [ ] 100% test pass rate maintained (867/867 tests)
- [ ] Performance benchmarks meet acceptance criteria (<5% change)
- [ ] Zero uses of `domain::sensor::beamforming::narrowband` in internal code
- [ ] Compatibility facade in place with deprecation notices
- [ ] Documentation updated (ADR, migration guide, README)

---

## Timeline

### Today (Preparation Phase)
- **Next 2-3 hours:** Complete Actions 1-4
- **End of day:** Ready to begin Sprint 1 migration

### This Week (Sprint 1 Execution)
- **Day 1:** Preparation phase (this document)
- **Day 2:** Migrate `capon.rs`
- **Day 3:** Migrate `steering_narrowband.rs` and `snapshots/`
- **Day 4:** Test migration and validation
- **Day 5:** Consumer updates and performance validation
- **Day 6:** Cleanup, documentation, code review

---

## Next Actions (Prioritized)

### 🔴 Critical (Do First)
1. **Action 1:** Source Inventory — Understand what we're migrating
2. **Action 2:** Dependency Analysis — Prevent circular dependencies
3. **Action 3:** Consumer Analysis — Identify breaking changes

### 🟡 Important (Do Next)
4. **Action 4:** Pre-Migration Validation — Establish baseline
5. **Decision D1-D3:** Make architectural decisions
6. **Create Branch:** Begin migration work

### 🟢 Next Phase (After Preparation)
7. Begin Sprint 1 execution (`SPRINT_1_NARROWBAND_MIGRATION.md`)
8. Migrate algorithms one-by-one (incremental approach)
9. Validate each step before proceeding

---

## Commands Quick Reference

```bash
# === Analysis Commands ===

# Find narrowband files
find src/domain/sensor/beamforming/narrowband -name "*.rs"

# Find consumers
grep -r "domain::sensor::beamforming::narrowband" src/ --include="*.rs"

# Count LOC
tokei src/domain/sensor/beamforming/narrowband/

# === Validation Commands ===

# Run tests
cargo test --all-features

# Run clippy
cargo clippy --all-features -- -D warnings

# Build check
cargo check --all-features

# Run benchmarks
cargo bench --bench narrowband_beamforming

# === Git Commands ===

# Create migration branch
git checkout -b refactor/narrowband-migration-sprint1

# Commit progress
git add docs/refactor/*.md
git commit -m "docs: Add Sprint 1 migration plan and analysis"

# Push branch
git push -u origin refactor/narrowband-migration-sprint1
```

---

## Appendix: File Paths

### Source Files (Domain Layer - TO BE MIGRATED)
```
src/domain/sensor/beamforming/narrowband/
├── mod.rs                          Public API
├── capon.rs                        Narrowband Capon algorithm
├── steering_narrowband.rs          Narrowband steering vectors
└── snapshots/
    ├── mod.rs                      Snapshot extraction
    └── windowed/
        └── mod.rs                  Windowed snapshot utilities
```

### Target Files (Analysis Layer - CANONICAL SSOT)
```
src/analysis/signal_processing/beamforming/narrowband/
├── mod.rs                          Public API (complete implementation)
├── capon.rs                        Narrowband Capon (to be created)
├── steering.rs                     Narrowband steering (to be created)
└── snapshots/
    ├── mod.rs                      Snapshot extraction (to be created)
    ├── extraction.rs               Core extraction logic (to be created)
    └── windowing.rs                Window functions (to be created)
```

### Documentation Files
```
docs/refactor/
├── BEAMFORMING_ARCHITECTURE_ANALYSIS.md      ✅ Complete
├── SPRINT_1_NARROWBAND_MIGRATION.md          ✅ Complete
├── IMMEDIATE_NEXT_STEPS.md                   ✅ This document
├── NARROWBAND_SOURCE_INVENTORY.md            ⚠️  To be created (Action 1)
└── BEAMFORMING_MIGRATION_GUIDE.md            ⚠️  To be updated after Sprint 1
```

---

## Conclusion

**Current Status:** 🟢 Ready to proceed with preparation phase

**Immediate Action:** Begin Action 1 (Source Inventory)

**Estimated Time to Sprint 1 Start:** 2-3 hours

**Confidence Level:** High — Infrastructure is in place, plan is clear, risks are identified and mitigated.

**Blocking Issues:** None identified

**Go/No-Go Decision:** ✅ **GO** — Proceed with preparation phase immediately

---

**Next Step:** Execute Action 1 — Create `NARROWBAND_SOURCE_INVENTORY.md`

---

*This plan adheres to Elite Mathematically-Verified Systems Architect principles: careful planning before execution, risk mitigation strategies defined, success criteria clear, zero tolerance for architectural violations.*