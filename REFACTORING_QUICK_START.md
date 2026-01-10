# Refactoring Quick Start Guide — kwavers
**Immediate Action Plan for Deep Vertical Hierarchy Refactoring**

**Date:** 2025-01-12  
**Status:** 🔴 READY TO EXECUTE  
**Priority:** P0 - CRITICAL ARCHITECTURAL FIXES

---

## Executive Summary

**Problem:** 947 files with significant cross-contamination, layer violations, and 50+ files >500 lines

**Solution:** 6-week refactoring in 4 phases, 10 sprints

**Goal:** Clean architecture with zero duplication, strict layering, GRASP compliance

---

## Critical Issues (Top 5)

### 1. 🔴 Beamforming Duplication
- **Problem:** 38 files in `domain/sensor/beamforming/` + 15 files in `analysis/signal_processing/beamforming/`
- **Impact:** ~6,000 lines duplicated, layer violation
- **Fix:** Consolidate ALL in `analysis/signal_processing/beamforming/`, delete `domain/sensor/beamforming/`

### 2. 🔴 Physics-Solver Coupling
- **Problem:** Physics equations in `solver/forward/acoustic/`, `solver/forward/nonlinear/`
- **Impact:** Tight coupling, unclear separation of concerns
- **Fix:** Move physics to `physics/acoustics/models/`, keep only numerical methods in `solver/`

### 3. 🔴 Grid Operations Scattered
- **Problem:** Operators in 5+ locations (domain/grid/operators/, solver/*/numerics/, math/numerics/)
- **Impact:** Redundant implementations, inconsistent APIs
- **Fix:** Consolidate in `math/numerics/differentiation/`

### 4. 🔴 Clinical Workflows Misplaced
- **Problem:** Clinical logic in `physics/acoustics/imaging/`, `physics/acoustics/therapy/`
- **Impact:** 76 files in wrong layer
- **Fix:** Move to `clinical/imaging/` and `clinical/therapy/`

### 5. 🔴 Massive Files
- **Problem:** 50+ files >500 lines (largest: 3,115 lines)
- **Impact:** GRASP violation, maintainability issues
- **Fix:** Split all files to <500 lines

---

## Quick Reference: File Moves

### Phase 1: Critical Duplication (Week 1-2)

```bash
# Sprint 1A: Beamforming (Days 1-3)
domain/sensor/beamforming/*           → analysis/signal_processing/beamforming/
domain/sensor/beamforming/experimental/neural.rs (3,115 lines) → 7 modules (<500 each)

# Sprint 1B: Grid Operations (Days 4-6)
domain/grid/operators/*               → math/numerics/differentiation/finite_difference/
solver/forward/*/numerics/operators/  → math/numerics/differentiation/spectral/
domain/medium/heterogeneous/interpolation/ → math/numerics/interpolation/

# Sprint 1C: Physics-Solver (Days 7-10)
solver/forward/acoustic/              → physics/acoustics/models/linear/
solver/forward/elastic/               → physics/elasticity/models/
solver/forward/nonlinear/kuznetsov/   → physics/acoustics/models/nonlinear/kuznetsov/
solver/forward/nonlinear/kzk/         → physics/acoustics/models/nonlinear/kzk/
solver/forward/nonlinear/westervelt/  → physics/acoustics/models/nonlinear/westervelt/
```

### Phase 2: Clinical Consolidation (Week 3-4)

```bash
# Sprint 2A: Clinical Workflows (Days 11-14)
physics/acoustics/imaging/modalities/elastography/ → clinical/imaging/elastography/
physics/acoustics/imaging/modalities/ceus/         → clinical/imaging/contrast_enhanced/
physics/acoustics/imaging/modalities/ultrasound/   → clinical/imaging/ultrasound/
physics/acoustics/imaging/fusion.rs                → clinical/imaging/fusion/
physics/acoustics/imaging/registration/            → clinical/imaging/fusion/registration/
physics/acoustics/therapy/*                        → clinical/therapy/
physics/acoustics/transcranial/                    → clinical/therapy/transcranial/
simulation/modalities/photoacoustic.rs             → clinical/imaging/photoacoustic/

# Sprint 2B: File Splitting (Days 15-20)
# Split ALL files >500 lines (50+ files)
```

### Phase 3: Cleanup (Week 5)

```bash
# Sprint 3A: Dead Code (Days 21-23)
DELETE: domain/sensor/beamforming/shaders/
DELETE: physics/acoustics/skull/legacy/
DELETE: solver/utilities/validation/kwave/
DELETE: errors.txt, build logs
ARCHIVE: 15+ redundant markdown docs → docs/archive/

# Sprint 3B: Dependencies (Days 24-25)
cargo udeps → remove unused
Optimize feature flags
Document dependency rationale
```

### Phase 4: Validation (Week 6)

```bash
# Sprint 4A: Testing (Days 26-28)
cargo test --all-features  # 867/867 passing
cargo bench                # No regression >5%
cargo clippy -- -D warnings # Zero warnings

# Sprint 4B: Documentation (Days 29-30)
Update README.md, ADR, API docs
Generate architecture diagrams
Create migration guide
```

---

## Target Architecture

```
src/
├── core/         [Foundation: errors, constants, types]
├── infra/        [Infrastructure: API, I/O, cloud]
├── domain/       [Domain primitives: grid, medium, sensor, source]
│   ├── boundary/
│   ├── field/
│   ├── grid/     [NO operators - just structure]
│   ├── medium/   [Property interfaces ONLY]
│   ├── sensor/   [Hardware abstractions ONLY - NO beamforming]
│   ├── signal/   [Signal definitions ONLY]
│   └── source/
├── math/         [Mathematical primitives]
│   ├── numerics/
│   │   ├── differentiation/  [ALL operators HERE (SSOT)]
│   │   ├── interpolation/    [Grid interpolation (SSOT)]
│   │   └── integration/
│   ├── linear_algebra/
│   ├── geometry/
│   ├── fft/
│   └── ml/
├── physics/      [Physics models ONLY - NO applications]
│   ├── acoustics/
│   │   ├── models/     [Wave equations]
│   │   ├── mechanics/  [Cavitation, streaming]
│   │   └── analytical/
│   ├── elasticity/models/
│   ├── thermal/models/
│   ├── optics/models/
│   └── coupling/
├── solver/       [Numerical methods ONLY - NO physics]
│   ├── numerical_methods/
│   │   ├── fdtd/
│   │   ├── pstd/
│   │   ├── dg/
│   │   └── hybrid/
│   ├── time_integration/
│   ├── inverse/
│   └── plugin_system/
├── analysis/     [Signal processing & analysis]
│   ├── signal_processing/
│   │   └── beamforming/  [ALL beamforming HERE (SSOT)]
│   ├── performance/
│   ├── testing/
│   ├── validation/
│   └── visualization/
├── simulation/   [Orchestration]
│   ├── configuration/
│   ├── builder/
│   └── orchestrator/
├── clinical/     [Application workflows]
│   ├── imaging/
│   │   ├── ultrasound/
│   │   ├── elastography/
│   │   ├── photoacoustic/
│   │   ├── contrast_enhanced/
│   │   └── fusion/
│   ├── therapy/
│   │   ├── hifu/
│   │   ├── lithotripsy/
│   │   ├── transcranial/
│   │   └── cavitation_control/
│   └── protocols/
└── gpu/          [GPU acceleration]
```

---

## Dependency Flow (Strict)

```
         clinical/     ← Application Layer
              ↓
         simulation/   ← Orchestration
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  solver/           analysis/     ← Analysis
    ↓                   ↓
    └─────────┬─────────┘
              ↓
          physics/              ← Physics Models
              ↓
          domain/               ← Domain Primitives
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  math/              infra/     ← Foundation
    ↓                   ↓
    └─────────┬─────────┘
              ↓
           core/                ← Core

RULES:
✅ Lower layers can be used by upper layers
❌ Upper layers CANNOT be used by lower layers
❌ NO circular dependencies
✅ Cross-cutting concerns via dependency injection
```

---

## Success Criteria

### Mandatory (Must Pass)

- [ ] **Zero files >500 lines** (GRASP compliance)
- [ ] **Zero layer violations** (strict hierarchy)
- [ ] **Zero duplicate implementations** (SSOT enforced)
- [ ] **867/867 tests passing** (no regressions)
- [ ] **Zero clippy warnings** (code quality)
- [ ] **Build time <30s** (SRS NFR-002)

### Verification Commands

```bash
# File size compliance
find src -name "*.rs" -exec wc -l {} \; | awk '$1 > 500 {print "❌ " $2 " (" $1 " lines)"}'
# Expected: No output

# Test suite
cargo test --all-features
# Expected: test result: ok. 867 passed; 0 failed

# Build time
time cargo build --release
# Expected: <30s

# Code quality
cargo clippy -- -D warnings
# Expected: 0 warnings emitted

# Duplication check
grep -r "fn delay_and_sum" src/ | wc -l
# Expected: 1 (SSOT - Single Source of Truth)
```

---

## Common Patterns

### Pattern 1: Moving Files

```bash
# Always use git mv to preserve history
git mv src/old/path/file.rs src/new/path/file.rs

# Update module declarations
# old/path/mod.rs: remove "pub mod file;"
# new/path/mod.rs: add "pub mod file;"

# Test immediately
cargo check
cargo test --package kwavers --lib new::path::file
```

### Pattern 2: Splitting Large Files

```bash
# 1. Create target structure
mkdir -p src/module/submodules/
touch src/module/submodules/{part1,part2,part3}.rs
touch src/module/submodules/mod.rs

# 2. Extract code (use your editor)
# - Preserve documentation
# - Keep related functionality together
# - Aim for <500 lines per file

# 3. Update parent module
echo "pub mod submodules;" >> src/module/mod.rs

# 4. Verify line counts
wc -l src/module/submodules/*.rs

# 5. Test
cargo test --package kwavers --lib module::submodules
```

### Pattern 3: Updating Imports

```bash
# Find affected files
grep -r "old::module::path" src/

# Automated replacement (use with caution)
find src -name "*.rs" -exec sed -i 's/old::module::path/new::module::path/g' {} +

# Always verify
cargo check
cargo test
```

---

## Risk Mitigation

### Backup Strategy
```bash
# Before starting
git tag refactor-baseline-$(date +%Y%m%d)
git push origin --tags

# Create feature branch
git checkout -b refactor/deep-vertical-hierarchy
git push -u origin refactor/deep-vertical-hierarchy

# Commit atomically (after each logical step)
git add <files>
git commit -m "Descriptive message"
git push
```

### Rollback Plan
```bash
# If something breaks
git reset --hard <last-good-commit>
git push --force

# If major issues
git checkout main
git branch -D refactor/deep-vertical-hierarchy
# Start over from tagged baseline
```

### Testing Checkpoint
```bash
# After EVERY sprint
cargo test --all-features 2>&1 | tee sprint_X_tests.log
cargo bench 2>&1 | tee sprint_X_bench.log
git tag sprint-X-complete
git push origin --tags

# If tests fail
git reset --hard sprint-$(($X-1))-complete
# Debug and retry
```

---

## Communication

### Stakeholder Updates

**Daily Standups:**
- Sprint X, Day Y: <current task>
- Completed: <list>
- Blockers: <list>
- Next: <task>

**Sprint Reviews:**
- After each sprint (every 3-5 days)
- Demonstrate working code
- Show metrics (tests, benchmarks, file counts)

**Phase Reviews:**
- After each phase (every 2 weeks)
- Comprehensive demonstration
- Performance analysis
- Updated documentation

---

## Getting Started (Next 5 Minutes)

```bash
# 1. Create backup
git tag refactor-baseline-$(date +%Y%m%d)
git push origin --tags

# 2. Create branch
git checkout -b refactor/deep-vertical-hierarchy

# 3. Record baseline
cargo test --all-features 2>&1 | tee baseline_tests.log
cargo bench 2>&1 | tee baseline_benchmarks.log
time cargo build --release 2>&1 | tee baseline_build.log

# 4. Start Sprint 1A
mkdir -p src/analysis/signal_processing/beamforming/neural
cd src/analysis/signal_processing/beamforming/neural

# 5. Read detailed checklist
cat REFACTORING_EXECUTION_CHECKLIST.md | less

# 6. Execute first task (Step 1A.2)
# Split domain/sensor/beamforming/experimental/neural.rs
```

---

## Key Contacts

**Technical Lead:** Elite Mathematically-Verified Systems Architect  
**Documentation:** `DEEP_VERTICAL_HIERARCHY_AUDIT.md` (comprehensive analysis)  
**Checklist:** `REFACTORING_EXECUTION_CHECKLIST.md` (detailed steps)  
**Validation:** `gap_audit.md` (mathematical correctness)

---

## Emergency Procedures

### Build Breaks
```bash
# 1. Don't panic
# 2. Check cargo check output
cargo check 2>&1 | tee error.log

# 3. Fix imports first (most common issue)
# 4. Run targeted tests
cargo test --package kwavers --lib <module>

# 5. If stuck, revert last commit
git reset --hard HEAD~1

# 6. Re-attempt with smaller steps
```

### Test Failures
```bash
# 1. Identify failing test
cargo test --all-features 2>&1 | grep FAILED

# 2. Run specific test with output
cargo test <test_name> -- --nocapture

# 3. Check if test assumption changed
# 4. Update test OR fix code
# 5. Verify no regressions

# 6. If many tests fail, likely import issue
grep -r "old::path" tests/
```

### Performance Regression
```bash
# 1. Compare benchmarks
diff baseline_benchmarks.log sprint_X_bench.log

# 2. Profile specific benchmark
cargo bench --bench <name> -- --profile-time=5

# 3. Common causes:
#    - Removed #[inline] annotations
#    - Added unnecessary abstractions
#    - Changed data structures

# 4. Fix and re-benchmark
# 5. Target: <5% regression acceptable
```

---

## Final Checklist Before Merge

- [ ] All 867 tests passing
- [ ] Zero clippy warnings
- [ ] Build time <30s
- [ ] All files <500 lines
- [ ] Zero duplication (SSOT verified)
- [ ] Documentation updated
- [ ] Migration guide created
- [ ] ADR updated with decisions
- [ ] Performance benchmarks acceptable (<5% regression)
- [ ] Code review completed
- [ ] Stakeholders approved

---

**Ready to begin? Start with Sprint 1A, Step 1A.1!**

See `REFACTORING_EXECUTION_CHECKLIST.md` for detailed instructions.