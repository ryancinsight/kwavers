# Phase 1 Execution Plan
## Deep Vertical Hierarchy Refactoring - Architectural Corrections

**Date:** 2024-12-19  
**Phase:** 1 (Architectural Layer Corrections)  
**Prerequisites:** Phase 0 Complete ✅  
**Estimated Duration:** 2-3 sprints (30-44 hours)

---

## Executive Summary

Phase 1 focuses on eliminating architectural layer violations and establishing correct deep vertical hierarchy with strict downward-only dependencies. Based on comprehensive codebase audit, the following violations have been identified and prioritized for remediation.

### Current State Assessment

**Build Status:** ✅ GREEN (0 errors, 25 warnings)  
**Architectural Violations:** 4 confirmed  
**Duplicate Modules:** 1 confirmed (beamforming)  
**Misplaced Components:** 2 confirmed (sparse matrices, Grid operators)

---

## Architectural Violations Identified

### ✅ ALREADY FIXED (Pre-Phase 1)
1. **Core → Physics dependency**: ❌ NOT FOUND - core/mod.rs does not import from physics
2. **Core → Math FFT re-export**: ❌ NOT FOUND - core/mod.rs does not re-export FFT

### 🔴 REQUIRES REMEDIATION

#### Violation 1: Sparse Matrices in Core (Should be in Math)
- **Current Location**: `src/core/utils/sparse_matrix/`
- **Correct Location**: `src/math/linear_algebra/sparse/`
- **Impact**: Core layer contains mathematical utilities (layer contamination)
- **Priority**: P1 (High)
- **Estimated Effort**: 6-8 hours

#### Violation 2: Grid Operators in Domain (Potentially Generic)
- **Current Location**: `src/domain/grid/operators/`
- **Analysis Required**: Determine if these are Grid-specific or generic
- **Parallel Location**: `src/math/numerics/operators/` (generic operators exist)
- **Impact**: Potential duplication or misplaced abstraction
- **Priority**: P2 (Medium)
- **Estimated Effort**: 8-10 hours (audit + refactor)

#### Violation 3: Duplicate Beamforming Implementations
- **Deprecated Path**: `src/domain/sensor/beamforming/`
- **Canonical Path**: `src/analysis/signal_processing/beamforming/`
- **Impact**: 76 deprecation warnings, maintenance burden, confusion
- **Priority**: P1 (High)
- **Estimated Effort**: 10-12 hours

#### Violation 4: Solver-Specific Operators (Potential Duplication)
- **Location**: `src/solver/forward/pstd/numerics/operators/`
- **Analysis Required**: Check overlap with `math/numerics/operators/`
- **Impact**: Potential duplication of spectral operators
- **Priority**: P2 (Medium)
- **Estimated Effort**: 4-6 hours (audit + consolidation)

---

## Phase 1 Task Breakdown

### Sprint 1: Core Layer Corrections (12-16 hours)

#### Task 1.1: Move Sparse Matrices to Math
**Priority:** P1  
**Effort:** 6-8 hours  
**Status:** 🔴 Not Started

**Objective**: Relocate sparse matrix implementation from `core/utils/` to `math/linear_algebra/sparse/`

**Steps**:
1. Create target directory structure
   ```bash
   mkdir -p src/math/linear_algebra/sparse
   ```

2. Move files:
   ```
   src/core/utils/sparse_matrix/mod.rs      → src/math/linear_algebra/sparse/mod.rs
   src/core/utils/sparse_matrix/coo.rs      → src/math/linear_algebra/sparse/coo.rs
   src/core/utils/sparse_matrix/csr.rs      → src/math/linear_algebra/sparse/csr.rs
   src/core/utils/sparse_matrix/solver.rs   → src/math/linear_algebra/sparse/solver.rs
   src/core/utils/sparse_matrix/eigenvalue.rs → src/math/linear_algebra/sparse/eigenvalue.rs
   ```

3. Update `src/math/linear_algebra/mod.rs`:
   ```rust
   pub mod sparse;
   pub use sparse::*;
   ```

4. Update `src/core/utils/mod.rs`:
   - Remove `pub mod sparse_matrix;`
   - Remove `pub use sparse_matrix::*;`

5. Update all importers:
   ```bash
   # Find all usages
   grep -r "use crate::core::utils::sparse_matrix" src/
   grep -r "use crate::core::sparse_matrix" src/
   
   # Replace with
   # use crate::math::linear_algebra::sparse
   ```

6. Verify:
   ```bash
   cargo build --all-features
   cargo test --lib
   ```

**Files Affected** (estimated):
- 5 moved files
- 10-15 importer files to update
- 2 module declarations

**Success Criteria**:
- ✅ `core/utils/sparse_matrix/` does not exist
- ✅ `math/linear_algebra/sparse/` contains all sparse matrix code
- ✅ All imports updated
- ✅ Build passes
- ✅ Tests pass

---

#### Task 1.2: Audit Grid Operators vs Math Operators
**Priority:** P2  
**Effort:** 6-8 hours  
**Status:** 🔴 Not Started

**Objective**: Determine correct ownership of differential operators and eliminate duplication

**Analysis Required**:
1. Compare implementations:
   - `domain/grid/operators/gradient.rs` vs `math/numerics/operators/differential.rs`
   - `domain/grid/operators/laplacian.rs` vs spectral operators
   - Check if domain operators are Grid-specific (use Grid API) or generic

2. Decision Tree:
   ```
   IF operators use Grid-specific API (Grid::spacing, Grid::dimension)
      THEN keep in domain/grid/operators/ (correct location)
   ELSE IF operators are generic (work on Array3<f64>)
      THEN move to math/numerics/operators/
   ```

3. Potential Outcomes:
   - **Outcome A**: Domain operators are Grid-specific wrappers → keep in domain, document relationship
   - **Outcome B**: Domain operators duplicate math operators → consolidate to math
   - **Outcome C**: Mixed - split into generic (math) and Grid-specific (domain)

**Steps**:
1. Read and compare implementations:
   ```bash
   # Domain operators
   cat src/domain/grid/operators/gradient.rs
   cat src/domain/grid/operators/laplacian.rs
   cat src/domain/grid/operators/divergence.rs
   cat src/domain/grid/operators/curl.rs
   
   # Math operators
   cat src/math/numerics/operators/differential.rs
   cat src/math/numerics/operators/spectral.rs
   ```

2. Create decision matrix:
   ```
   Operator    | Grid-Specific? | Generic Equivalent? | Action
   ------------|----------------|---------------------|--------
   gradient    | TBD            | TBD                 | TBD
   laplacian   | TBD            | TBD                 | TBD
   divergence  | TBD            | TBD                 | TBD
   curl        | TBD            | TBD                 | TBD
   ```

3. Based on outcome, execute appropriate refactoring (deferred to Sprint 2)

**Deliverable**:
- `OPERATOR_OWNERSHIP_ANALYSIS.md` (decision document)
- Refactoring plan for Sprint 2 (if needed)

**Success Criteria**:
- ✅ Clear ownership documented
- ✅ No unexplained duplication
- ✅ Refactoring plan approved (if needed)

---

### Sprint 2: Duplicate Elimination (10-14 hours)

#### Task 2.1: Remove Deprecated Beamforming Module
**Priority:** P1  
**Effort:** 10-12 hours  
**Status:** 🔴 Not Started

**Objective**: Complete migration from `domain/sensor/beamforming/` to `analysis/signal_processing/beamforming/` and remove deprecated code

**Pre-Flight Checks**:
1. Verify canonical implementation is complete:
   ```bash
   # Check analysis implementation
   ls -R src/analysis/signal_processing/beamforming/
   
   # Count features
   grep -r "pub fn\|pub struct\|pub enum" src/analysis/signal_processing/beamforming/ | wc -l
   grep -r "pub fn\|pub struct\|pub enum" src/domain/sensor/beamforming/ | wc -l
   ```

2. Identify all references to deprecated path:
   ```bash
   grep -r "domain::sensor::beamforming" src/
   grep -r "use crate::domain::sensor::beamforming" src/
   ```

**Steps**:
1. Verify feature parity:
   - Create checklist of all public APIs in deprecated module
   - Verify each exists in canonical module
   - Document any missing features

2. Update all references (estimated 76 locations):
   ```rust
   // OLD
   use crate::domain::sensor::beamforming::...;
   
   // NEW
   use crate::analysis::signal_processing::beamforming::...;
   ```

3. Update module re-exports:
   - Check `src/domain/sensor/mod.rs`
   - Remove beamforming re-export or add deprecation notice

4. Run comprehensive tests:
   ```bash
   cargo test --all-features -- beamforming
   ```

5. Remove deprecated directory:
   ```bash
   git rm -r src/domain/sensor/beamforming/
   ```

6. Update documentation:
   - README.md (if mentions beamforming paths)
   - Architecture docs
   - Module-level rustdoc

**Files Affected** (estimated):
- 76+ files with deprecated imports
- 1 directory removed
- Multiple documentation files

**Success Criteria**:
- ✅ Zero references to `domain::sensor::beamforming`
- ✅ All tests pass
- ✅ Deprecation warnings eliminated (down from 76)
- ✅ Documentation updated

---

#### Task 2.2: Audit PSTD Operators for Duplication
**Priority:** P2  
**Effort:** 4-6 hours  
**Status:** 🔴 Not Started

**Objective**: Verify if `solver/forward/pstd/numerics/operators/` duplicates `math/numerics/operators/`

**Analysis**:
1. Compare spectral operator implementations:
   ```bash
   # PSTD operators
   cat src/solver/forward/pstd/numerics/operators/spectral.rs
   
   # Math operators
   cat src/math/numerics/operators/spectral.rs
   ```

2. Check for PSTD-specific extensions vs generic duplication

3. Decision:
   ```
   IF PSTD operators are PSTD-specific (solver state, domain decomposition)
      THEN keep separate (document relationship)
   ELSE IF PSTD operators duplicate math operators
      THEN consolidate to math, PSTD imports from math
   ```

**Steps**:
1. Read both implementations
2. Create comparison document
3. Based on outcome, execute consolidation (if needed)

**Deliverable**:
- `PSTD_OPERATOR_ANALYSIS.md`
- Consolidation plan (if duplication found)

**Success Criteria**:
- ✅ Clear ownership documented
- ✅ No unexplained duplication

---

### Sprint 3: Validation & Hardening (8-10 hours)

#### Task 3.1: Create Architectural Validation Script
**Priority:** P1  
**Effort:** 4-5 hours  
**Status:** 🔴 Not Started

**Objective**: Automated detection of layer violations

**Script Requirements**:
```python
#!/usr/bin/env python3
# tools/check_layer_violations.py

"""
Architectural layer violation detector.

Enforces dependency flow:
  clinical → analysis → solver → physics → domain → math → core
  
Rules:
  - Lower layers CANNOT import from higher layers
  - Lateral imports allowed within same layer (with caution)
"""

LAYER_HIERARCHY = {
    'core': 0,
    'math': 1,
    'domain': 2,
    'physics': 3,
    'solver': 4,
    'analysis': 5,
    'clinical': 6,
}

def check_imports(file_path):
    """Parse Rust file and check for upward imports."""
    # Implementation...
    
def main():
    violations = []
    for rust_file in find_rust_files('src/'):
        violations.extend(check_imports(rust_file))
    
    if violations:
        print(f"Found {len(violations)} layer violations:")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    else:
        print("✅ No layer violations detected")
        sys.exit(0)
```

**Steps**:
1. Create `tools/check_layer_violations.py`
2. Implement import parser (regex or tree-sitter)
3. Test on known violations
4. Add to CI pipeline (`.github/workflows/`)

**Success Criteria**:
- ✅ Script detects upward imports
- ✅ Passes on current codebase (after fixes)
- ✅ Integrated into CI

---

#### Task 3.2: Update Architecture Documentation
**Priority:** P2  
**Effort:** 4-5 hours  
**Status:** 🔴 Not Started

**Objective**: Sync documentation with post-Phase-1 architecture

**Files to Update**:
1. `README.md`:
   - Module structure diagram
   - Dependency graph
   - Import paths

2. `ARCHITECTURE.md` (create if missing):
   - Layer definitions
   - Dependency rules
   - Module ownership

3. `CONTRIBUTING.md`:
   - Where to add new features
   - Layer guidelines

4. Rustdoc (module-level):
   - Update import examples
   - Correct cross-references

**Steps**:
1. Generate current dependency graph:
   ```bash
   cargo modules generate graph --lib > docs/architecture_phase1.dot
   ```

2. Create/update architecture documents

3. Update README with corrected import examples

4. Regenerate rustdoc and verify links:
   ```bash
   cargo doc --all-features --no-deps
   ```

**Success Criteria**:
- ✅ Documentation matches code
- ✅ Import examples are correct
- ✅ Dependency graph visualized

---

## Testing Strategy

### Per-Task Testing
After each task:
```bash
# 1. Compilation
cargo build --all-features

# 2. Clippy
cargo clippy --all-features -- -D warnings

# 3. Unit tests
cargo test --lib --all-features

# 4. Specific module tests
cargo test --all-features -- <module_name>
```

### End-of-Sprint Testing
```bash
# 1. Full test suite
cargo test --all-features

# 2. Integration tests
cargo test --test '*'

# 3. Benchmarks (if applicable)
cargo bench --no-run

# 4. Documentation build
cargo doc --all-features --no-deps

# 5. Architectural validation
python tools/check_layer_violations.py
```

### Regression Prevention
```bash
# Before any changes
cargo test --all-features > tests_before.log

# After changes
cargo test --all-features > tests_after.log

# Compare
diff tests_before.log tests_after.log
```

---

## Risk Management

### High-Risk Tasks
1. **Beamforming Migration** (Task 2.1)
   - **Risk**: Breaking changes to analysis module
   - **Mitigation**: 
     - Feature-branch development
     - Incremental commit per file
     - Run beamforming tests after each change
   
2. **Sparse Matrix Move** (Task 1.1)
   - **Risk**: Performance-critical code, many dependencies
   - **Mitigation**:
     - Benchmark before/after
     - Verify inlining still works
     - Check generated assembly if needed

### Medium-Risk Tasks
1. **Operator Consolidation** (Tasks 1.2, 2.2)
   - **Risk**: Potential behavior changes if implementations differ
   - **Mitigation**:
     - Thorough comparison before merge
     - Property-based tests to verify equivalence

---

## Success Metrics

### Phase 1 Complete When:
- [ ] All P1 tasks complete
- [ ] All P2 tasks complete or documented/deferred
- [ ] Zero architectural violations detected by validation script
- [ ] Build passes: `cargo build --all-features`
- [ ] Tests pass: `cargo test --all-features`
- [ ] Clippy passes: `cargo clippy --all-features -- -D warnings`
- [ ] Documentation synchronized
- [ ] Deprecation warnings reduced by 76 (beamforming)

### Key Performance Indicators
- **Layer Violations**: 4 → 0
- **Duplicate Modules**: 1 → 0
- **Deprecation Warnings**: 76 → 0 (beamforming-related)
- **Build Time**: Monitor (should not increase significantly)
- **Test Coverage**: Maintain or improve

---

## Task Assignment & Timeline

### Sprint 1 (Week 1)
- **Day 1-2**: Task 1.1 (Sparse Matrices)
- **Day 3-4**: Task 1.2 (Operator Audit)
- **Day 5**: Buffer for issues

### Sprint 2 (Week 2)
- **Day 1-3**: Task 2.1 (Beamforming Migration)
- **Day 4**: Task 2.2 (PSTD Audit)
- **Day 5**: Buffer for issues

### Sprint 3 (Week 3)
- **Day 1-2**: Task 3.1 (Validation Script)
- **Day 2-3**: Task 3.2 (Documentation)
- **Day 4-5**: Final testing, cleanup, review

---

## Rollback Plan

If a task causes critical failures:

1. **Immediate Rollback**:
   ```bash
   git revert <commit-hash>
   cargo build --all-features
   cargo test --all-features
   ```

2. **Document Issue**:
   - What broke
   - Why it broke
   - Alternative approach

3. **Defer or Re-plan**:
   - Move task to Phase 2
   - Break into smaller subtasks
   - Request additional review

---

## Next Steps After Phase 1

### Phase 2: Deep Hierarchy Refinement
1. Split large files (>500 lines)
2. Deepen module hierarchy where it clarifies responsibility
3. Extract more shared components
4. Performance optimization

### Phase 3: Advanced Validation
1. Dependency graph visualization
2. Cyclic dependency detection
3. API stability guarantees
4. Benchmark regression tests

---

## Appendix

### A. Module Layer Classification

```
Layer 0 (Core):
  - core::constants
  - core::error
  - core::time
  - core::log
  - core::utils (non-math)

Layer 1 (Math):
  - math::linear_algebra (including sparse)
  - math::numerics (including operators)
  - math::fft
  - math::ml

Layer 2 (Domain):
  - domain::grid
  - domain::medium
  - domain::boundary
  - domain::source
  - domain::field
  - domain::plugin

Layer 3 (Physics):
  - physics::acoustics
  - physics::optics
  - physics::thermal
  - physics::chemistry

Layer 4 (Solver):
  - solver::forward
  - solver::inverse
  - solver::amr
  - solver::pstd

Layer 5 (Analysis):
  - analysis::signal_processing
  - analysis::statistics
  - analysis::testing

Layer 6 (Clinical):
  - clinical::therapy
  - clinical::imaging
  - clinical::safety
```

### B. Import Pattern Examples

**Correct** (downward):
```rust
// Solver imports from physics (layer 4 → 3)
use crate::physics::acoustics::WaveEquation;

// Physics imports from domain (layer 3 → 2)
use crate::domain::medium::Medium;

// Domain imports from math (layer 2 → 1)
use crate::math::linear_algebra::sparse::CSRMatrix;

// Math imports from core (layer 1 → 0)
use crate::core::error::KwaversResult;
```

**Incorrect** (upward):
```rust
// Core imports from math (layer 0 → 1) ❌
use crate::math::fft::FFT;

// Domain imports from physics (layer 2 → 3) ❌
use crate::physics::acoustics::SPEED_OF_SOUND;
```

### C. Contact Information

**Phase Lead**: Kwavers Refactoring Team  
**Review Board**: Architecture Committee  
**Escalation Path**: Project Lead → Tech Lead → CTO

---

**Document Status**: 📋 ACTIVE  
**Last Updated**: 2024-12-19  
**Next Review**: After Sprint 1 completion