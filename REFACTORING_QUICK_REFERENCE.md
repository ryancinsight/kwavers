# Kwavers Refactoring Quick Reference Card

**Version:** 1.0  
**Date:** 2025-01-12  
**Status:** 🔴 ACTIVE REFACTORING

---

## 🎯 One-Page Overview

### The Problem
- **928 Rust files** with cross-contamination
- **37 files >1000 lines** (max 3,115 lines)
- **21+ layer violations** across architecture
- **Duplicate logic** in grid ops, numerics, physics models

### The Solution
**8-week phased refactoring** OR **2-week critical path**

---

## 📊 Current Violations (Actual Data)

| Layer | Upward Imports | Target | Status |
|-------|---------------|--------|--------|
| Core | 6 | 0 | 🔴 |
| Math | 11 | 0 | 🔴 |
| Domain | 5 | 0 | 🔴 |
| Physics | 11 | 0 | 🔴 |
| Solver | 0 | 0 | ✅ |

**Total:** 33 violations to fix

---

## 🗺️ Target Architecture (Strict Layering)

```
Layer 8: analysis/     (Cross-cutting: validation, testing, signal processing)
Layer 7: clinical/     (Applications: imaging, therapy, workflows)
Layer 6: simulation/   (Orchestration: builder, config, core loop)
Layer 5: solver/       (Numerical methods: FDTD, PSTD, DG, hybrid)
Layer 4: physics/      (Physical models: acoustics, thermal, optics)
Layer 3: domain/       (Primitives: grid, medium, boundary, field)
Layer 2: math/         (Numerics: operators, FFT, linear algebra)
Layer 1: core/         (Foundation: errors, constants, time)
Layer 0: infra/        (Infrastructure: I/O, API, cloud)

gpu/                   (Cross-cutting: GPU acceleration)
```

**Rule:** Only downward dependencies allowed (↓)

---

## 🚀 8-Week Plan (Full Refactoring)

| Week | Focus | Deliverable |
|------|-------|-------------|
| **1** | Foundation & Math | `math/numerics/operators/` established |
| **2** | Domain Purification | Signal processing moved to analysis |
| **3** | Physics Models | Nonlinear models moved from solver |
| **4** | Solver Cleanup | Pure numerical methods only |
| **5** | Clinical Apps | All imaging/therapy consolidated |
| **6** | File Size | All files <500 lines |
| **7** | Testing | Full suite passing, no regressions |
| **8** | Documentation | Docs updated, migration guide |

---

## ⚡ 2-Week Critical Path (Minimum)

### Week 1: Stop the Bleeding
- [ ] Delete dead code (algorithms_old.rs)
- [ ] Fix core layer violations
- [ ] Create `math/numerics/operators/` skeleton
- [ ] Document remaining violations

### Week 2: Quick Wins
- [ ] Move beamforming → analysis
- [ ] Move imaging → clinical
- [ ] Update critical imports
- [ ] Full test verification

---

## 📋 Priority Moves (By Severity)

### P0: Critical (Week 1-2)
```
domain/sensor/beamforming/*           → analysis/signal_processing/beamforming/
domain/sensor/localization/*          → analysis/signal_processing/localization/
domain/imaging/*                      → clinical/imaging/
math/ml/pinn/physics/*                → physics/ml_integration/ OR delete
```

### P1: High (Week 3-4)
```
solver/forward/nonlinear/kuznetsov/*  → physics/acoustics/models/kuznetsov/
solver/forward/nonlinear/westervelt/* → physics/acoustics/models/westervelt/
solver/forward/nonlinear/kzk/*        → physics/acoustics/models/kzk/
physics/acoustics/imaging/*           → clinical/imaging/
physics/acoustics/therapy/*           → clinical/therapy/
```

### P2: Medium (Week 5-6)
```
simulation/modalities/*               → clinical/imaging/
physics/acoustics/validation/*        → analysis/validation/physics/
solver/utilities/validation/*         → analysis/validation/numerics/
```

---

## 🔍 Verification Commands

### Check Layer Violations
```bash
# Core (should be 0)
grep -r "use crate::" src/core/ --include="*.rs" | grep -v "use crate::core::" | wc -l

# Math (should be 0 from upper layers)
grep -r "use crate::" src/math/ --include="*.rs" | grep -E "(domain|physics|solver)" | wc -l

# Domain (should be 0 from upper layers)
grep -r "use crate::" src/domain/ --include="*.rs" | grep -E "(physics|solver|clinical)" | wc -l

# Physics (should be 0 from solver/clinical)
grep -r "use crate::" src/physics/ --include="*.rs" | grep -E "(solver|clinical)" | wc -l
```

### Find Large Files
```bash
# Files >500 lines
find src -name "*.rs" -exec wc -l {} + | awk '$1 > 500' | wc -l

# Files >1000 lines
find src -name "*.rs" -exec wc -l {} + | awk '$1 > 1000' | sort -rn
```

---

## 🛠️ Daily Workflow

### Morning Standup
1. Review yesterday's progress in `REFACTOR_PHASE_1_CHECKLIST.md`
2. Update task status (🔴 → 🟡 → 🟢)
3. Run verification commands
4. Identify blockers

### During Work
1. Make changes following layer rules
2. Run `cargo check` frequently
3. Run affected tests: `cargo test <module>`
4. Update checklist as you go

### Evening Wrap-up
1. Run full test suite: `cargo test --all-features`
2. Commit with descriptive message
3. Update progress in tracking docs
4. Note any deviations or discoveries

---

## 🚨 Red Flags (Stop & Review)

- ⛔ Test suite failing >10% of tests
- ⛔ Performance regression >10%
- ⛔ Compilation errors lasting >1 hour
- ⛔ Upward dependency introduced
- ⛔ Circular dependency created
- ⛔ File grows >500 lines without splitting

**If you hit a red flag:** Pause, document issue, review with team

---

## ✅ Success Metrics (Track Weekly)

```
┌─────────────────────────────────────────┐
│ REFACTORING PROGRESS DASHBOARD          │
├─────────────────────────────────────────┤
│ Layer Violations:                       │
│   Core:    6 → ? → 0  [██░░░░░░░░] %   │
│   Math:   11 → ? → 0  [██░░░░░░░░] %   │
│   Domain:  5 → ? → 0  [██░░░░░░░░] %   │
│   Physics: 11 → ? → 0  [██░░░░░░░░] %   │
│                                          │
│ File Size Compliance:                   │
│   >1000:  37 → ? → 0  [░░░░░░░░░░] %   │
│   >500:  120 → ? → 0  [░░░░░░░░░░] %   │
│                                          │
│ Test Status:                            │
│   Passing: ???/??? [██████████] 100%    │
│   Perf:    +/-??%  (target: ±5%)       │
└─────────────────────────────────────────┘
```

Update this weekly by running verification commands.

---

## 📚 Key Documents

| Document | Purpose | When to Use |
|----------|---------|-------------|
| `REFACTORING_EXECUTIVE_SUMMARY.md` | High-level overview | Start here, share with stakeholders |
| `ARCHITECTURE_REFACTORING_AUDIT.md` | Complete analysis | Deep dive into issues |
| `DEPENDENCY_ANALYSIS.md` | Violation details | Understanding current state |
| `REFACTOR_PHASE_1_CHECKLIST.md` | Day-by-day tasks | Daily execution |
| `REFACTORING_QUICK_REFERENCE.md` | This doc | Quick lookup during work |

---

## 🎨 Code Patterns

### Before (Wrong)
```rust
// domain/sensor/beamforming/capon.rs
// Signal processing algorithm in domain layer
impl BeamformingProcessor {
    fn capon_beamformer(&self, data: &Array2<f64>) -> Array1<f64> {
        // Complex signal processing logic
    }
}
```

### After (Correct)
```rust
// analysis/signal_processing/beamforming/capon.rs
// Signal processing in analysis layer
impl CaponBeamformer {
    fn process(&self, data: &Array2<f64>) -> Array1<f64> {
        // Same logic, correct location
    }
}

// domain/sensor/geometry.rs
// Domain layer only defines sensor positions
pub struct SensorArray {
    positions: Vec<Point3D>,
    // No processing logic
}
```

---

## 🔄 Git Workflow

### Branch Strategy
```bash
# Main refactoring branch
git checkout -b refactor/architecture-cleanup

# Phase branches
git checkout -b refactor/phase1-foundation
git checkout -b refactor/phase2-domain
# etc.
```

### Commit Message Format
```
<type>(<scope>): <description>

refactor(math): consolidate differential operators
refactor(domain): move beamforming to analysis
fix(core): remove upward dependencies
docs(arch): update ADR with refactoring decisions
```

---

## 💡 Tips & Tricks

### Move Files Safely
```bash
# Use git mv to preserve history
git mv src/old/path/file.rs src/new/path/file.rs

# Update imports in same commit
# Run cargo check immediately
```

### Split Large Files
```bash
# 1. Identify logical components
# 2. Create new files with one component each
# 3. Update mod.rs with re-exports
# 4. Update imports
# 5. Delete original file
# 6. Verify tests pass
```

### Test Incrementally
```bash
# Test affected module immediately
cargo test --lib <module_path>

# Test full suite before committing
cargo test --all-features

# Benchmark if touching hot path
cargo bench --bench <relevant_benchmark>
```

---

## 🆘 Emergency Rollback

```bash
# If things go wrong, rollback to last good state
git log --oneline  # Find last good commit
git revert <commit-hash>
git push origin <branch>

# Or hard reset (if not pushed)
git reset --hard <commit-hash>
```

---

## 📞 Decision Tree

```
Start Refactoring
    ↓
Do you have 8 weeks? 
    Yes → Full refactoring (recommended)
    No → ↓
Do you have 2 weeks?
    Yes → Critical path
    No → ↓
Can you allocate time later?
    Yes → Document violations, schedule later
    No → Incremental improvements (12+ weeks)
```

---

## 🎯 Phase 1 Quick Commands (Week 1)

```bash
# Day 1: Dead code removal
find src -name "*_old.rs" -delete
find src -name "*_backup.rs" -delete
cargo check

# Day 2: Math module structure
mkdir -p src/math/numerics/operators
touch src/math/numerics/operators/{mod,differential,spectral,interpolation}.rs

# Day 3: Differential operators
# Copy from solver/forward/fdtd/numerics/finite_difference.rs
# Adapt to new trait interface

# Day 4: Spectral operators
# Copy from solver/forward/pstd/numerics/operators/spectral.rs
# Adapt to new trait interface

# Day 5: Testing & integration
cargo test --all-features
cargo bench
git commit -m "refactor(math): establish numerics foundation"
```

---

## 🏁 Definition of Done (Per Phase)

- [ ] All planned moves completed
- [ ] Zero new violations introduced
- [ ] All tests passing
- [ ] Performance within ±5%
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Committed and pushed
- [ ] Checklist updated

---

**Remember:** This is a marathon, not a sprint. Take it phase by phase, verify continuously, and maintain quality standards throughout.

**Questions?** Refer to detailed docs or ask the architecture team.

**Status:** 🔴 Ready to begin → 🟡 In progress → 🟢 Phase complete