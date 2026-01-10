# Refactoring Index — kwavers
**Master Navigation for Deep Vertical Hierarchy Refactoring**

**Date:** 2025-01-12  
**Status:** 🔴 READY FOR EXECUTION  
**Version:** 2.15.0 → 3.0.0 (Breaking Changes)

---

## Document Hierarchy

This refactoring effort is documented across multiple files. Use this index to navigate:

### 📋 Primary Documents

1. **DEEP_VERTICAL_HIERARCHY_AUDIT.md** (1,102 lines)
   - **Purpose:** Comprehensive architectural analysis
   - **Audience:** Technical leads, architects
   - **Contains:** 
     - Complete file inventory (947 files)
     - Cross-contamination analysis
     - Dependency flow violations
     - Target architecture
     - 50+ files >500 lines analysis
   - **When to read:** Before starting, for complete context

2. **REFACTORING_QUICK_START.md** (493 lines)
   - **Purpose:** Quick reference and immediate action guide
   - **Audience:** Engineers executing refactoring
   - **Contains:**
     - Top 5 critical issues
     - File move quick reference
     - Target architecture summary
     - Common patterns
     - Emergency procedures
   - **When to read:** Daily, for quick decisions

3. **REFACTORING_EXECUTION_CHECKLIST.md** (1,865 lines)
   - **Purpose:** Step-by-step tactical execution
   - **Audience:** Engineers performing actual refactoring
   - **Contains:**
     - 10 sprints, 4 phases, 6 weeks
     - Atomic tasks with verification steps
     - Bash commands for each operation
     - Test requirements per step
   - **When to read:** During execution, task by task

4. **gap_audit.md** (existing)
   - **Purpose:** Mathematical and physics validation
   - **Audience:** Scientists, validation engineers
   - **Contains:**
     - Literature references
     - Physics correctness proofs
     - Test coverage analysis
   - **When to read:** For validation concerns

5. **THIS FILE (REFACTORING_INDEX.md)**
   - **Purpose:** Navigation and progress tracking
   - **Audience:** Everyone
   - **Contains:**
     - Document map
     - Progress dashboard
     - Quick links

---

## Quick Navigation

### 🎯 By Role

**Project Manager / Tech Lead:**
- Start: [Executive Summary](#executive-summary)
- Track: [Progress Dashboard](#progress-dashboard)
- Review: [Phase Completion Criteria](#phase-completion-criteria)

**Software Engineer (Executor):**
- Start: `REFACTORING_QUICK_START.md`
- Work from: `REFACTORING_EXECUTION_CHECKLIST.md`
- Reference: [Common Patterns](#common-patterns-reference)

**Reviewer / QA:**
- Check: [Success Criteria](#success-criteria)
- Validate: [Verification Commands](#verification-commands)
- Test: [Test Requirements](#test-requirements)

**Architect:**
- Review: `DEEP_VERTICAL_HIERARCHY_AUDIT.md`
- Consult: [Target Architecture](#target-architecture-summary)
- Validate: [Dependency Flow](#dependency-flow-rules)

### 🔍 By Topic

| Topic | Document | Section |
|-------|----------|---------|
| **Beamforming Duplication** | DEEP_VERTICAL_HIERARCHY_AUDIT.md | Lines 290-450 |
| **Physics-Solver Separation** | DEEP_VERTICAL_HIERARCHY_AUDIT.md | Lines 451-550 |
| **Grid Operations** | DEEP_VERTICAL_HIERARCHY_AUDIT.md | Lines 551-650 |
| **Clinical Workflows** | DEEP_VERTICAL_HIERARCHY_AUDIT.md | Lines 651-750 |
| **File Splitting Strategy** | DEEP_VERTICAL_HIERARCHY_AUDIT.md | Lines 751-850 |
| **Module Depth Issues** | DEEP_VERTICAL_HIERARCHY_AUDIT.md | Lines 851-900 |
| **Quick File Moves** | REFACTORING_QUICK_START.md | Lines 50-110 |
| **Emergency Procedures** | REFACTORING_QUICK_START.md | Lines 380-450 |
| **Sprint 1A Tasks** | REFACTORING_EXECUTION_CHECKLIST.md | Lines 50-350 |
| **Sprint 1B Tasks** | REFACTORING_EXECUTION_CHECKLIST.md | Lines 351-650 |

---

## Executive Summary

### Problem Statement

**Current State:**
- 947 Rust source files
- 50+ files violating GRASP principle (>500 lines)
- Largest file: 3,115 lines (neural.rs)
- 200+ cross-layer violations
- 15+ duplicate implementations
- 8-level module nesting

**Target State:**
- ~780 well-organized files
- 100% GRASP compliant (<500 lines/file)
- Zero layer violations (strict hierarchy)
- Zero duplication (SSOT enforced)
- Maximum 4-5 level nesting

### Critical Issues (Priority P0)

1. **Beamforming Duplication** (🔴 CRITICAL)
   - 38 files in `domain/sensor/beamforming/`
   - 15 files in `analysis/signal_processing/beamforming/`
   - ~40-50% code duplication (~6,000-7,500 lines)
   - **Impact:** Maintenance nightmare, inconsistent APIs
   - **Solution:** Consolidate in `analysis/`, delete `domain/sensor/beamforming/`

2. **Physics-Solver Coupling** (🔴 CRITICAL)
   - Physics equations in `solver/forward/acoustic/`, `solver/forward/nonlinear/`
   - Tight coupling between "what to solve" and "how to solve"
   - **Impact:** Cannot swap solvers, hard to test
   - **Solution:** Move physics to `physics/`, keep only numerics in `solver/`

3. **Grid Operations Scattered** (🔴 CRITICAL)
   - Operators in 5+ locations
   - Inconsistent implementations
   - **Impact:** Bug risks, performance issues
   - **Solution:** Consolidate in `math/numerics/differentiation/`

4. **Clinical Workflows Misplaced** (🔴 CRITICAL)
   - 76 files in `physics/` that belong in `clinical/`
   - Application logic mixed with physics
   - **Impact:** Confused responsibilities
   - **Solution:** Move to `clinical/imaging/` and `clinical/therapy/`

5. **Massive Files** (🔴 CRITICAL)
   - 50+ files >500 lines
   - Largest: 3,115 lines (6.2x over limit)
   - **Impact:** GRASP violation, hard to maintain
   - **Solution:** Split to <500 lines each

---

## Progress Dashboard

### Overall Progress

```
Phase 1: Critical Duplication      [░░░░░░░░░░] 0% (0/3 sprints)
Phase 2: Clinical Consolidation    [░░░░░░░░░░] 0% (0/2 sprints)
Phase 3: Dead Code Removal          [░░░░░░░░░░] 0% (0/2 sprints)
Phase 4: Validation & Docs          [░░░░░░░░░░] 0% (0/2 sprints)

Total Progress:                     [░░░░░░░░░░] 0% (0/10 sprints)
```

### Sprint Status

| Sprint | Name | Duration | Status | Tests | Files | Notes |
|--------|------|----------|--------|-------|-------|-------|
| **1A** | Beamforming Consolidation | Days 1-3 | ⬜ Not Started | - | - | - |
| **1B** | Grid Operations | Days 4-6 | ⬜ Not Started | - | - | - |
| **1C** | Physics-Solver | Days 7-10 | ⬜ Not Started | - | - | - |
| **2A** | Clinical Migration | Days 11-14 | ⬜ Not Started | - | - | - |
| **2B** | File Splitting | Days 15-20 | ⬜ Not Started | - | - | - |
| **3A** | Dead Code | Days 21-23 | ⬜ Not Started | - | - | - |
| **3B** | Dependencies | Days 24-25 | ⬜ Not Started | - | - | - |
| **4A** | Testing | Days 26-28 | ⬜ Not Started | - | - | - |
| **4B** | Documentation | Days 29-30 | ⬜ Not Started | - | - | - |

**Legend:** ⬜ Not Started | 🔵 In Progress | ✅ Complete | ❌ Failed

---

## Phase Completion Criteria

### Phase 1: Critical Duplication (Week 1-2)

**Sprint 1A: Beamforming Consolidation**
- [ ] 38 files moved from `domain/sensor/beamforming/` → `analysis/signal_processing/beamforming/`
- [ ] 3 large files split (3,115, 1,260, 1,148 lines → <500 each)
- [ ] All beamforming algorithms consolidated (SSOT)
- [ ] 150+ import statements updated
- [ ] 867/867 tests passing
- [ ] Zero performance regression

**Sprint 1B: Grid Operations Consolidation**
- [ ] 5 files moved from `domain/grid/operators/` → `math/numerics/differentiation/`
- [ ] Stencil definitions consolidated
- [ ] Spectral operators unified
- [ ] Interpolation utilities moved
- [ ] 100+ import statements updated
- [ ] 867/867 tests passing

**Sprint 1C: Physics-Solver Separation**
- [ ] Physics models moved to `physics/acoustics/models/`
- [ ] Solver layer cleaned (only numerical methods)
- [ ] Physics-solver bridge created
- [ ] Zero circular dependencies
- [ ] 867/867 tests passing

### Phase 2: Clinical Consolidation (Week 3-4)

**Sprint 2A: Clinical Workflows Migration**
- [ ] 76 files moved to `clinical/imaging/` and `clinical/therapy/`
- [ ] 23 large workflow files split
- [ ] Physics layer cleaned (no application logic)
- [ ] 150+ import statements updated
- [ ] 867/867 tests passing

**Sprint 2B: Massive File Decomposition**
- [ ] All 50+ files >500 lines split
- [ ] 100% GRASP compliance achieved
- [ ] Git history preserved
- [ ] 867/867 tests passing

### Phase 3: Dead Code Removal (Week 5)

**Sprint 3A: File Cleanup**
- [ ] Deprecated code deleted
- [ ] Build artifacts removed
- [ ] Documentation consolidated
- [ ] .gitignore updated

**Sprint 3B: Dependency Audit**
- [ ] Unused dependencies removed
- [ ] Feature flags optimized
- [ ] Dependency rationale documented

### Phase 4: Validation & Documentation (Week 6)

**Sprint 4A: Comprehensive Testing**
- [ ] All 867 tests passing
- [ ] Property-based tests added
- [ ] Performance benchmarks validated
- [ ] Zero clippy warnings

**Sprint 4B: Documentation Update**
- [ ] README.md updated
- [ ] ADR updated
- [ ] API documentation complete
- [ ] Migration guide created

---

## Target Architecture Summary

```
src/
├── core/         ✅ Foundation (constants, errors, types)
├── infra/        ✅ Infrastructure (API, I/O, cloud)
├── domain/       ✅ Domain Primitives (grid, medium, sensor, source)
│   └── sensor/   [NO beamforming - hardware abstractions ONLY]
├── math/         ✅ Mathematical Primitives
│   └── numerics/
│       ├── differentiation/  [ALL operators HERE - SSOT]
│       └── interpolation/    [Grid interpolation - SSOT]
├── physics/      ✅ Physics Models (NO applications)
│   └── acoustics/
│       ├── models/     [Wave equations]
│       └── mechanics/  [Cavitation, streaming]
├── solver/       ✅ Numerical Methods (NO physics)
│   └── numerical_methods/  [FDTD, PSTD, DG]
├── analysis/     ✅ Analysis & Signal Processing
│   └── signal_processing/
│       └── beamforming/  [ALL beamforming HERE - SSOT]
├── simulation/   ✅ Orchestration
├── clinical/     ✅ Application Workflows
│   ├── imaging/
│   └── therapy/
└── gpu/          ✅ GPU Acceleration
```

---

## Dependency Flow Rules

```
         clinical/     [Applications]
              ↓
         simulation/   [Orchestration]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  solver/           analysis/
    ↓                   ↓
    └─────────┬─────────┘
              ↓
          physics/      [Models]
              ↓
          domain/       [Primitives]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  math/              infra/
    ↓                   ↓
    └─────────┬─────────┘
              ↓
           core/        [Foundation]

✅ Downward dependencies allowed
❌ Upward dependencies forbidden
❌ Circular dependencies forbidden
```

---

## Success Criteria

### Mandatory (Must Pass)

```bash
# 1. File size compliance
find src -name "*.rs" -exec wc -l {} \; | awk '$1 > 500 {print}'
# Expected: No output

# 2. Test suite
cargo test --all-features
# Expected: test result: ok. 867 passed; 0 failed

# 3. Build time
time cargo build --release
# Expected: <30s (SRS NFR-002 compliant)

# 4. Code quality
cargo clippy -- -D warnings
# Expected: 0 warnings emitted

# 5. SSOT verification
grep -r "fn delay_and_sum" src/ | wc -l
# Expected: 1 (single source of truth)
```

### Performance Targets

- **Test execution:** <30s total (SRS NFR-002)
- **Build time:** <30s release build
- **Benchmark regression:** <5% acceptable
- **Memory usage:** No leaks detected

---

## Verification Commands

### File Size Check
```bash
# List all violations
find src -name "*.rs" -exec wc -l {} \; | \
  awk '$1 > 500 {print "❌ " $2 " (" $1 " lines)"}'

# Count violations
find src -name "*.rs" -exec wc -l {} \; | \
  awk '$1 > 500 {count++} END {print "Total violations: " (count ? count : 0)}'
```

### Duplication Check
```bash
# Check for duplicate implementations
# Example: delay_and_sum should only exist once
grep -r "fn delay_and_sum" src/ | wc -l
grep -r "fn calculate_delays" src/ | wc -l
grep -r "fn gradient_x" src/ | wc -l

# Expected: 1 for each (SSOT)
```

### Layer Violation Check
```bash
# Domain should NOT import solver
grep -r "use crate::solver" src/domain/ | wc -l
# Expected: 0

# Physics should NOT import clinical
grep -r "use crate::clinical" src/physics/ | wc -l
# Expected: 0

# Math should NOT import domain
grep -r "use crate::domain" src/math/ | wc -l
# Expected: 0 or only grid for geometry
```

### Test Status
```bash
# Full test suite
cargo test --all-features 2>&1 | tee test_results.log

# Parse results
grep "test result:" test_results.log
# Expected: test result: ok. 867 passed; 0 failed; 0 ignored

# Ignored tests
cargo test --all-features -- --ignored
```

### Code Quality
```bash
# Clippy warnings
cargo clippy --all-features -- -D warnings 2>&1 | tee clippy_results.log

# Count warnings
grep "warning:" clippy_results.log | wc -l
# Expected: 0

# Format check
cargo fmt --check
```

---

## Common Patterns Reference

### Pattern 1: Moving a File
```bash
git mv src/old/path/file.rs src/new/path/file.rs
# Update old/path/mod.rs: remove "pub mod file;"
# Update new/path/mod.rs: add "pub mod file;"
cargo check && cargo test --lib new::path::file
```

### Pattern 2: Splitting a Large File
```bash
# Create structure
mkdir -p src/module/submodules/
# Extract code to multiple files (<500 lines each)
# Update parent mod.rs
echo "pub mod submodules;" >> src/module/mod.rs
# Verify
wc -l src/module/submodules/*.rs
cargo test --lib module::submodules
```

### Pattern 3: Updating Imports
```bash
# Find affected files
grep -r "old::path" src/
# Replace (carefully!)
find src -name "*.rs" -exec sed -i 's/old::path/new::path/g' {} +
# Verify
cargo check && cargo test
```

---

## Test Requirements

### Per Sprint
- **Unit tests:** All module tests passing
- **Integration tests:** Infrastructure, integration, simple_integration
- **Fast tests:** CFL stability, energy conservation
- **Full suite:** 867/867 tests passing

### Full Validation (Phase 4)
- **Literature validation:** k-Wave comparison
- **Physics validation:** Rigorous physics tests
- **Performance benchmarks:** All benchmarks passing
- **Property-based tests:** Proptest suites passing

---

## Risk Mitigation

### Backup Strategy
```bash
# Before starting
git tag refactor-baseline-$(date +%Y%m%d)
git push origin --tags

# Per sprint
git tag sprint-X-complete
git push origin --tags
```

### Rollback Procedure
```bash
# If major issues
git reset --hard sprint-$(($X-1))-complete
# Or
git reset --hard refactor-baseline-YYYYMMDD
```

### Test Checkpoint
```bash
# After EVERY change
cargo test --all-features
# If fails, revert immediately
git reset --hard HEAD~1
```

---

## Communication Plan

### Daily Updates (Standup)
- Current sprint/task
- Completed items
- Blockers
- Next steps

### Sprint Reviews (Every 3-5 Days)
- Demo working code
- Metrics (tests, files, performance)
- Blockers and risks

### Phase Reviews (Every 2 Weeks)
- Comprehensive demo
- Architecture validation
- Performance analysis
- Documentation updates

---

## Emergency Contacts

**Technical Lead:** Elite Mathematically-Verified Systems Architect  
**Primary Docs:**
- Comprehensive: `DEEP_VERTICAL_HIERARCHY_AUDIT.md`
- Quick Reference: `REFACTORING_QUICK_START.md`
- Detailed Steps: `REFACTORING_EXECUTION_CHECKLIST.md`

**Validation:** `gap_audit.md` (mathematical correctness)

---

## Next Steps

### Immediate (Next 5 Minutes)
1. Read `REFACTORING_QUICK_START.md`
2. Create backup: `git tag refactor-baseline-$(date +%Y%m%d)`
3. Create branch: `git checkout -b refactor/deep-vertical-hierarchy`
4. Record baseline metrics
5. Start Sprint 1A

### Sprint 1A (Days 1-3)
1. Review `REFACTORING_EXECUTION_CHECKLIST.md` Sprint 1A
2. Create `analysis/signal_processing/beamforming/` structure
3. Split large files (neural.rs, beamforming_3d.rs, ai_integration.rs)
4. Migrate beamforming algorithms
5. Update imports
6. Verify tests (867/867 passing)

### Sprint 1B (Days 4-6)
1. Create `math/numerics/differentiation/` structure
2. Migrate grid operators
3. Consolidate stencils
4. Update solver imports
5. Verify tests

### Continue per checklist...

---

## Status Updates

**Last Updated:** 2025-01-12  
**Current Phase:** Pre-refactoring (Planning Complete)  
**Next Milestone:** Sprint 1A Kickoff  
**Overall Progress:** 0% (0/10 sprints complete)

---

**🚀 Ready to begin? Start with `REFACTORING_QUICK_START.md`!**