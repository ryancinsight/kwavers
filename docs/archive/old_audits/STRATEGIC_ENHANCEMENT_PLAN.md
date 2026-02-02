# Kwavers Strategic Enhancement Plan
**Phase: Production Hardening & Research-Driven Enhancement**
**Timeline: Current Sprint**
**Branch: main (all work)**

---

## Executive Vision

Transform kwavers into a **world-class production-grade ultrasound and optics simulation library** by:

1. **Fixing all compilation errors and warnings** (clean build)
2. **Removing all dead code and deprecated patterns** (pristine codebase)
3. **Verifying zero circular dependencies** (clean architecture)
4. **Implementing research-driven enhancements** (state-of-the-art features)
5. **Ensuring proper module hierarchy** (domain/physics/solver/analysis/clinical separation)

---

## Phase 1: CRITICAL FIXES (Fix Now)

### 1.1 Compilation Errors - BLOCKER

**Issue**: 6-7 remaining compilation errors preventing full test suite execution

**Priority**: 🔴 CRITICAL - Block all other work

**Action Items**:
- [ ] Fix ai_integration_test module reference
- [ ] Fix PINN import paths (`kwavers::ml::pinn::*` → `kwavers::solver::inverse::pinn::ml::*`)
- [ ] Fix syntax errors (extra braces, type annotations)
- [ ] Verify all tests compile
- [ ] Run full test suite

**Effort**: 2-4 hours
**Owner**: Primary focus

---

### 1.2 High-Severity Warnings

**Issue**: `#[allow(dead_code)]` markers hiding actual problems

**Priority**: 🟠 HIGH - Address immediately

**Action Items**:
- [ ] Audit every `#[allow(dead_code)]` - is it justified?
- [ ] Remove unjustified allows
- [ ] Document feature-gated code with `#[cfg(...)]`
- [ ] Remove truly dead code

**Effort**: 4-6 hours

---

## Phase 2: ARCHITECTURAL VERIFICATION (Parallel to Phase 1)

### 2.1 Verify No Circular Dependencies

**Command**: `cargo build --all-targets 2>&1 | grep -i circular`

**Expected**: Zero matches

**Action**: If any found:
- [ ] Trace dependency chain
- [ ] Identify which layer is wrong
- [ ] Move code to correct module
- [ ] Verify with rebuild

**Modules to Check**:
- `domain` ← should NOT import from `solver`, `analysis`, `clinical`
- `solver` ← should NOT import from `analysis`, `clinical`
- `physics` ← should NOT import from `domain`, `solver`, `analysis`

---

### 2.2 Verify Separation of Concerns

**Domain Layer Should Contain**:
- ✅ Grid definitions
- ✅ Medium property structs (empty containers, no computations)
- ✅ Sensor array geometry
- ✅ Signal type definitions
- ✅ Boundary condition specs
- ❌ NO algorithms
- ❌ NO solver code
- ❌ NO physics computations

**Physics Layer Should Contain**:
- ✅ Physical models (wave equations, thermal equations)
- ✅ Constitutive relations
- ✅ Material property calculations
- ❌ NO solver implementations
- ❌ NO domain objects
- ❌ NO analysis algorithms

**Solver Layer Should Contain**:
- ✅ Numerical solver implementations (FDTD, PSTD, BEM)
- ✅ Plugin system
- ✅ Time stepping
- ❌ NO analysis algorithms
- ❌ NO clinical workflows
- ❌ NO beamforming code

**Analysis Layer Should Contain**:
- ✅ Signal processing algorithms
- ✅ Beamforming methods
- ✅ Localization algorithms
- ✅ Imaging reconstruction
- ❌ NO domain objects
- ❌ NO clinical workflows
- ❌ NO solver implementations

**Clinical Layer Should Contain**:
- ✅ Therapy workflows
- ✅ Imaging protocols
- ✅ Safety compliance
- ✅ Patient workflows
- ❌ NO core algorithms (use analysis layer)

---

## Phase 3: DEAD CODE ELIMINATION (After Phase 1)

### 3.1 Identify Unused Code

**Commands**:
```bash
# Find unused functions
cargo clippy --all-targets -- -W unused

# Find unused imports
cargo clippy --all-targets -- -W unused-imports

# Find unreachable code
grep -r "unreachable!" src/
```

**Action Items**:
- [ ] List all unused items by file
- [ ] Verify not used elsewhere
- [ ] Remove if safe
- [ ] Comment with explanation if keeping
- [ ] Commit with `refactor: Remove dead code`

---

### 3.2 Remove Obsolete Examples

**Status**: 5 examples removed in previous session

**Verify**:
- [ ] `examples/phase2_*.rs` - all removed? ✅ Yes
- [ ] `examples/clinical_therapy_workflow.rs` - removed? ✅ Yes
- [ ] No other incomplete examples exist?

---

## Phase 4: RESEARCH-DRIVEN ENHANCEMENTS

### 4.1 k-Wave Integration (From k-Wave & k-Wave-Python)

**Current Kwavers**: Basic PSTD implementation

**Target Enhancement**:
```
Add k-space corrected PSTD solver:
- Implement k-space correction operator
- Add proper PML boundary conditions
- Improve grid spacing calculation (points-per-wavelength)
- Add automatic time step validation
```

**Files to Create**:
- [ ] `src/solver/forward/pstd/kspace_correction.rs` - New
- [ ] Enhanced grid calculation in `src/domain/grid/mod.rs`

**Estimated Effort**: 8-12 hours

---

### 4.2 Differentiable Simulations (From j-Wave)

**Current Kwavers**: Fixed solver implementations

**Target Enhancement**:
```
Add autodiff support for inverse problems:
- Mark solver state as differentiable
- Implement adjoint method for gradients
- Enable automatic parameter optimization
```

**Feature Gate**: `#[cfg(feature = "autodiff")]`

**Files to Modify**:
- [ ] `src/solver/forward/pstd/mod.rs` - Add adjoint solver
- [ ] `src/solver/inverse/mod.rs` - Add gradient support
- [ ] `Cargo.toml` - Add `autodiff` feature

**Estimated Effort**: 16-20 hours

---

### 4.3 High-Order FDTD (From Fullwave25)

**Current Kwavers**: 2nd-order FDTD

**Target Enhancement**:
```
Add 4th and 8th-order FDTD:
- Implement staggered grid with multiple ghost points
- Add high-order stencils
- Improve CFL stability analysis
- Provide accuracy-vs-cost comparison
```

**Files to Create**:
- [ ] `src/solver/forward/fdtd/high_order.rs` - New
- [ ] `src/solver/forward/fdtd/stencils.rs` - New

**Estimated Effort**: 12-16 hours

---

### 4.4 Clinical Workflow Integration (From BabelBrain)

**Current Kwavers**: Basic clinical module

**Target Enhancement**:
```
Add complete treatment planning:
- Multi-stage pipeline (prep → acoustic → thermal)
- DICOM import/export
- Safety metric validation (MI, TI, ISPTA)
- Real-time monitoring
```

**Files to Modify**:
- [ ] `src/clinical/therapy/mod.rs` - Add pipeline
- [ ] `src/clinical/safety/mod.rs` - Add validators
- [ ] `src/infra/io/mod.rs` - Add DICOM support

**Estimated Effort**: 20-24 hours

---

### 4.5 Adaptive Beamforming (From DBUA & Sound-Speed-Estimation)

**Current Kwavers**: Basic DAS, MVDR

**Target Enhancement**:
```
Add neural network beamforming:
- Implement learned delay optimization
- Add coherence-based sound speed estimation
- Support uncertainty quantification
- Enable real-time adaptation
```

**Files to Create**:
- [ ] `src/analysis/signal_processing/beamforming/neural.rs` - New
- [ ] `src/analysis/signal_processing/beamforming/coherence.rs` - New

**Estimated Effort**: 16-20 hours

---

## Phase 5: ARCHITECTURE HARDENING

### 5.1 Add Module Documentation

**For each module root** (`mod.rs`), add:
```rust
//! Module purpose
//! 
//! ## Architectural Role
//! - What this layer owns
//! - What it doesn't own
//! - Dependencies
//! 
//! ## Design Patterns
//! - Key abstractions
//! - Plugin architecture
//! 
//! ## Examples
//! ```ignore
//! // Usage example
//! ```
```

**Files to Update**: All 12 top-level modules

**Effort**: 4-6 hours

---

### 5.2 Add SSOT Markers

Create a document showing Single Source of Truth for each concept:

```
Grid Specifications:      src/domain/grid/mod.rs (canonical)
Medium Properties:        src/domain/medium/mod.rs (canonical)
Sensor Arrays:           src/domain/sensor/array.rs (canonical)
Signal Types:            src/domain/signal/mod.rs (canonical)
Physics Constants:       src/physics/constants.rs (canonical)
Beamforming Algorithms:  src/analysis/signal_processing/beamforming/ (canonical)
Solver Interfaces:       src/solver/interface.rs (canonical)
```

**File to Create**: `ARCHITECTURE_SSOT.md`

**Effort**: 2-3 hours

---

## Phase 6: BUILD CONFIGURATION

### 6.1 Fix Cargo.toml Features

**Current**: May have incomplete feature definitions

**Target**:
```toml
[features]
default = ["core"]
core = []           # Mandatory core functionality
physics = []        # All physics modules
solvers = []        # All solver implementations
clinical = []       # Clinical workflows
gpu = ["wgpu"]      # GPU acceleration (requires wgpu)
autodiff = []       # Differentiable computing
visualization = []  # UI/plotting
api = ["axum"]      # REST API
```

**Action**: Review and update `Cargo.toml`

**Effort**: 1-2 hours

---

## Phase 7: TESTING & VALIDATION

### 7.1 Create Baseline Metrics

```bash
# Compilation
cargo build --all-targets 2>&1 | tee baseline_build.log

# Warnings count
cargo clippy --all-features 2>&1 | grep "warning:" | wc -l

# Test results
cargo test --lib 2>&1 | tail -20 | tee baseline_tests.log

# Code quality
cargo fmt --check
```

**Create**: `BASELINE_METRICS.md` with counts

**Effort**: 1 hour

---

### 7.2 Verify Each Phase

**After each phase**:
```bash
cargo build --all-targets
cargo clippy --all-features
cargo test --lib
cargo fmt --check
```

**Commit if all pass**

---

## Phase 8: DOCUMENTATION

### 8.1 Create Architecture Guide

**File**: `ARCHITECTURE_GUIDE.md` (2000+ words)

**Sections**:
1. Layer descriptions with examples
2. Module responsibilities matrix
3. Data flow diagrams
4. Dependency graph
5. SSOT catalog
6. Plugin system guide
7. Adding new features checklist

**Effort**: 4-6 hours

---

### 8.2 Update README

**Current**: Likely basic

**Target**: Include:
- Architecture overview (ASCII diagram)
- Quick start guide
- Module organization
- How to contribute
- Design principles

**Effort**: 2-3 hours

---

## IMPLEMENTATION SEQUENCE

### Week 1: Foundation (Critical Path)
```
Day 1-2: Phase 1 - Fix all compilation errors
Day 2-3: Phase 7.1 - Baseline metrics
Day 3-4: Phase 2 - Verify architecture
Day 4-5: Phase 3 - Remove dead code
```

### Week 2: Enhancement (Parallel)
```
Day 1-2: Phase 4.1 - k-space PSTD (HIGH IMPACT)
Day 2-3: Phase 4.5 - Adaptive beamforming (HIGH IMPACT)
Day 3-4: Phase 4.2 - Autodiff framework (MEDIUM EFFORT)
```

### Week 3: Hardening
```
Day 1-2: Phase 5 - Documentation
Day 2-3: Phase 6 - Build configuration
Day 3-4: Phase 8 - Complete documentation
Day 4-5: Phase 7.2 - Final validation
```

---

## SUCCESS CRITERIA

### Phase 1 (CRITICAL)
- ✅ `cargo build --all-targets` succeeds with zero errors
- ✅ `cargo clippy --all-features` shows no high-severity warnings
- ✅ `cargo test --lib` passes at 95%+ rate

### Phase 2-3
- ✅ No circular dependencies detected
- ✅ Layer separation verified
- ✅ All `#[allow(dead_code)]` justified or removed

### Phase 4-5
- ✅ k-space PSTD working with validation tests
- ✅ Autodiff framework compiles and basic tests pass
- ✅ New algorithms properly located and documented

### Phase 6-8
- ✅ `cargo fmt --check` passes
- ✅ Module docs complete for all layers
- ✅ Architecture guide published
- ✅ Zero warnings in final build

### FINAL STATUS
```
✅ Zero compilation errors
✅ Zero critical warnings  
✅ 99%+ test pass rate
✅ Clean architecture verified
✅ Research-driven features implemented
✅ Production-ready documentation
✅ Approved for 3.0.0 release
```

---

## RESOURCES REQUIRED

**Team**:
- 1 primary engineer: Phases 1-3, 5-8 (2 weeks full-time)
- 1 optional: Phases 4 (10+ hours/week for research features)

**Tools**:
- Rust 1.70+ (for modern features)
- Cargo with clippy enabled
- Git for version control

**External**:
- Reference repositories (already reviewed)
- Academic papers on PSTD, autodiff, high-order FDTD

---

## RISK MITIGATION

**Risk**: Large refactoring breaks functionality

**Mitigation**:
- Work on `main` branch (per your requirements)
- Commit after each minor fix
- Run tests after each phase
- Keep baseline metrics for comparison

**Risk**: Research features take longer than estimated

**Mitigation**:
- Phase 1-3 (foundation) must complete first
- Phase 4 (features) can be prioritized
- Phase 4.1 (k-space) highest priority (directly improves accuracy)

---

## NEXT IMMEDIATE ACTIONS

**Right Now** (Next 30 minutes):
1. Review this plan
2. Identify any modifications needed
3. Confirm Phase 1-3 timeline is acceptable

**Phase 1 Start** (Next 1-2 hours):
```bash
# Start fixing compilation errors
cargo build --all-targets 2>&1 | head -50
# Fix first 2-3 errors
cargo test --lib 2>&1 | grep "^error"
# Continue until clean build
```

---

## COMMITMENT

This plan provides a **clear path to production-grade quality** while implementing **state-of-the-art features** from the latest research. Following this plan will result in:

✅ **Zero dead code, zero warnings, zero deprecated patterns**  
✅ **Clean 9-layer architecture with verified SSOT**  
✅ **Research-driven enhancements (k-space, autodiff, high-order)**  
✅ **Complete documentation and architecture guides**  
✅ **Production-ready for v3.0.0 release**

**Timeline**: 3-4 weeks for complete execution
**Quality Target**: AAA+ (Excellent, publication-ready)
**Complexity**: High (systematic architecture work)
**Value**: Maximum (foundation for years of research)
