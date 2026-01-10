# Architecture Refactoring Execution Plan
## Kwavers Deep Vertical Hierarchy Enforcement

**Version**: 1.0  
**Date**: 2024  
**Status**: READY FOR EXECUTION  
**Priority**: P0 - Critical Architecture Debt  

---

## Executive Summary

**Problem**: Significant cross-contamination and redundancy across 944 Rust source files violating deep vertical hierarchy principles.

**Impact**: 
- Circular dependencies preventing clean builds
- Duplicated code (~5000 LOC estimated)
- Unclear ownership and maintenance burden
- Slow incremental builds
- Difficult onboarding for new contributors

**Solution**: Phased refactoring over 11 weeks to establish strict layered architecture.

**Success Metrics**:
- Zero circular dependencies
- <500 duplicate LOC
- Build time: clean <60s, incremental <5s
- All modules <500 lines (GRASP compliant)

---

## Phase 0: Preparation (Week 0 - Pre-execution)

### Sprint 0.1: Infrastructure Setup (2 days)

**Objective**: Enable automated architecture validation

#### Tasks

1. **Build Architecture Checker** ✅ COMPLETE
   ```bash
   cd xtask
   cargo build --release
   cargo run -- check-architecture
   ```
   - File: `xtask/src/architecture/dependency_checker.rs` (532 lines)
   - File: `xtask/src/architecture/mod.rs` (204 lines)
   - Integration: `xtask/src/main.rs` updated with `check-architecture` command

2. **Run Baseline Audit**
   ```bash
   cd xtask
   cargo run -- check-architecture --markdown --strict
   ```
   - Output: `ARCHITECTURE_VALIDATION_REPORT.md`
   - Action: Document all current violations (baseline)
   - Metric: Count of violations by category

3. **Add CI Pipeline Check**
   ```yaml
   # .github/workflows/architecture.yml
   name: Architecture Validation
   on: [push, pull_request]
   jobs:
     validate:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Check Architecture
           run: |
             cd xtask
             cargo run -- check-architecture --strict
   ```

4. **Clean Root Directory**
   ```bash
   chmod +x scripts/cleanup_root_docs.sh
   ./scripts/cleanup_root_docs.sh
   ```
   - Moves 19 architecture docs to `docs/architecture/`
   - Moves 9 completed docs to `docs/archive/`
   - Deletes build artifacts: `errors.txt`
   - Creates index files

### Sprint 0.2: Documentation & Planning (1 day)

**Objective**: Ensure team alignment on refactoring approach

#### Deliverables

1. **Architecture Overview** ✅ COMPLETE
   - File: `ARCHITECTURAL_AUDIT_SPRINT_ANALYSIS.md` (998 lines)
   - Sections: Cross-contamination analysis, module dependencies, refactoring priority matrix

2. **Team Review Session**
   - Review: Audit findings with tech lead
   - Decision: Approve phased approach
   - Commitment: Freeze non-critical features during Phases 1-2

3. **Risk Assessment**
   - High-risk areas identified: Medium, Beamforming, Solver interfaces
   - Mitigation: Adapter layers during transition
   - Rollback: Git branches per phase

---

## Phase 1: Critical Path Consolidation (Weeks 1-4)

**Goal**: Eliminate highest-severity cross-contamination

### Sprint 1: Grid Consolidation (Week 1 - 40 hours)

**Priority**: HIGH  
**Affected Files**: 15+  
**Breaking Changes**: YES (import paths)

#### Step 1.1: Create Canonical Grid Types (Day 1-2, 16h)

**Tasks**:
1. Create `domain/grid/topologies.rs`
   ```rust
   pub trait GridTopology {
       fn dims(&self) -> (usize, usize, usize);
       fn linear_index(&self, i: usize, j: usize, k: usize) -> usize;
       fn spatial_indices(&self, idx: usize) -> (usize, usize, usize);
   }
   ```

2. Create `domain/grid/cartesian.rs`
   - Move existing `Grid` implementation
   - Implement `GridTopology` trait
   - Add comprehensive tests

3. Create `domain/grid/cylindrical.rs`
   - Extract from `solver/forward/axisymmetric/coordinates.rs`
   - Implement `GridTopology` trait
   - Migrate tests

4. Create `domain/grid/staggered.rs`
   - Extract from `solver/forward/fdtd/numerics/staggered_grid.rs`
   - Implement `GridTopology` trait
   - Preserve field component enum

**Verification**:
```bash
cargo test --lib domain::grid
cargo doc --no-deps --open --package kwavers --lib domain::grid
```

#### Step 1.2: Update Solver Imports (Day 3-4, 16h)

**Tasks**:
1. Update `solver/forward/fdtd/` to use `domain::grid::StaggeredGrid`
   - Remove `solver/forward/fdtd/numerics/staggered_grid.rs`
   - Update imports in 8+ files
   - Run tests: `cargo test --package kwavers --lib solver::forward::fdtd`

2. Update `solver/forward/axisymmetric/` to use `domain::grid::CylindricalGrid`
   - Remove `solver/forward/axisymmetric/coordinates.rs`
   - Update imports in 4+ files
   - Run tests: `cargo test --package kwavers --lib solver::forward::axisymmetric`

3. Update `math/numerics/operators/` to accept `dyn GridTopology`
   - Generic over grid type
   - No concrete grid dependency

**Verification**:
```bash
cargo test --lib solver::forward
cargo clippy -- -D warnings
```

#### Step 1.3: Deprecation & Migration (Day 5, 8h)

**Tasks**:
1. Add deprecated re-exports for 1 release cycle
   ```rust
   // solver/forward/fdtd/numerics/mod.rs
   #[deprecated(since = "2.15.0", note = "Use domain::grid::StaggeredGrid")]
   pub use crate::domain::grid::StaggeredGrid;
   ```

2. Update all examples in `examples/`
3. Update documentation in `docs/`
4. Create migration guide: `docs/migration/v2.14_to_v2.15.md`

**Deliverables**:
- ✅ All grid operations in `domain/grid/`
- ✅ Zero grid duplication in solvers
- ✅ All tests passing
- ✅ Architecture checker: Grid violations = 0

---

### Sprint 2: Boundary Consolidation (Week 2 - 32 hours)

**Priority**: HIGH  
**Affected Files**: 12+  
**Breaking Changes**: YES (CPML API)

#### Step 2.1: Define Boundary Trait (Day 1, 8h)

**Tasks**:
1. Create `domain/boundary/traits.rs`
   ```rust
   pub trait BoundaryCondition: Send + Sync {
       fn apply(&mut self, field: &mut Field, grid: &dyn GridTopology, dt: f64) -> KwaversResult<()>;
       fn reset(&mut self);
       fn supports_field(&self, field_type: FieldType) -> bool;
   }
   ```

2. Create `domain/boundary/cpml.rs`
   - Consolidate from `domain/boundary/cpml/` directory
   - Implement `BoundaryCondition` trait
   - Migrate memory fields, coefficients

#### Step 2.2: Implement Boundary Types (Day 2-3, 16h)

**Tasks**:
1. `domain/boundary/pml.rs` - Standard PML
2. `domain/boundary/abc.rs` - Absorbing boundary condition
3. Each implements `BoundaryCondition` trait

#### Step 2.3: Remove Solver Implementations (Day 4, 8h)

**Tasks**:
1. Delete `solver/utilities/cpml_integration.rs`
2. Delete `solver/forward/fdtd/numerics/boundary_stencils.rs`
3. Update solvers to use `dyn BoundaryCondition`
   - FDTD: `self.boundary.apply(&mut self.field, &self.grid, dt)?;`
   - PSTD: Similar pattern
   - Hybrid: Delegate to underlying solvers

**Verification**:
```bash
cargo test --lib domain::boundary
cargo test --lib solver::forward::fdtd
cargo test --lib solver::forward::pstd
cargo bench cpml_benchmark
```

**Deliverables**:
- ✅ All boundary conditions in `domain/boundary/`
- ✅ Trait-based boundary application
- ✅ Zero boundary duplication
- ✅ Performance benchmarks: No regression

---

### Sprint 3-4: Medium Consolidation (Week 3-4 - 48 hours)

**Priority**: CRITICAL  
**Affected Files**: 30+  
**Breaking Changes**: YES (Medium traits)

#### Step 3.1: Audit Medium Traits (Week 3, Day 1, 8h)

**Tasks**:
1. List all medium property definitions across codebase
   ```bash
   grep -r "trait.*Medium" src/physics/
   grep -r "struct.*Medium" src/physics/
   grep -r "impl.*Medium" src/physics/
   ```

2. Create trait consolidation matrix:
   - Acoustic properties: Where defined? Where should be?
   - Optical properties: Where defined? Where should be?
   - Thermal properties: Where defined? Where should be?

3. Document breaking changes in migration guide

#### Step 3.2: Consolidate Acoustic Traits (Week 3, Day 2-3, 16h)

**Tasks**:
1. Ensure `domain/medium/heterogeneous/traits/acoustic/` is canonical
   ```rust
   pub trait AcousticMedium: Medium {
       fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64;
       fn density(&self, i: usize, j: usize, k: usize) -> f64;
       fn attenuation(&self, i: usize, j: usize, k: usize) -> f64;
       fn nonlinearity(&self, i: usize, j: usize, k: usize) -> f64;
   }
   ```

2. Update `physics/acoustics/` to import, not define
   - Remove local trait definitions
   - Import from `domain::medium::heterogeneous::traits::acoustic`

3. Run tests: `cargo test --lib physics::acoustics`

#### Step 3.3: Consolidate Optical Traits (Week 3, Day 4-5, 16h)

**Tasks**:
1. Ensure `domain/medium/heterogeneous/traits/optical/` is canonical
   ```rust
   pub trait OpticalMedium: Medium {
       fn refractive_index(&self, i: usize, j: usize, k: usize) -> f64;
       fn absorption_coefficient(&self, i: usize, j: usize, k: usize) -> f64;
       fn scattering_coefficient(&self, i: usize, j: usize, k: usize) -> f64;
   }
   ```

2. Update `physics/optics/` to import, not define
3. Run tests: `cargo test --lib physics::optics`

#### Step 3.4: Remove Solver Medium Types (Week 4, Day 1, 8h)

**Tasks**:
1. Delete `solver/forward/axisymmetric/config.rs::AxisymmetricMedium`
2. Use `domain::medium` types with cylindrical projection
3. Update all solver configurations

**Verification**:
```bash
cargo test --all-features
cargo doc --no-deps --open
xtask check-architecture --strict
```

**Deliverables**:
- ✅ All medium traits in `domain/medium/heterogeneous/traits/`
- ✅ Physics modules consume, never define
- ✅ Zero medium duplication
- ✅ Architecture checker: Medium violations = 0

---

## Phase 2: Architectural Cleanup (Weeks 5-8)

### Sprint 5: Beamforming Consolidation (Week 5 - 56 hours)

**Priority**: HIGH  
**Affected Files**: 35+  
**Breaking Changes**: YES (import paths)

#### Step 5.1: Algorithm Inventory (Day 1, 8h)

**Tasks**:
1. List all beamforming algorithms in both locations
   - `analysis/signal_processing/beamforming/`
   - `domain/sensor/beamforming/`

2. Identify duplicates vs. variants
   - DAS (Delay-and-Sum): Compare implementations
   - MVDR (Minimum Variance Distortionless Response): Compare
   - MUSIC: Compare

3. Create consolidation map

#### Step 5.2: Consolidate to analysis/beamforming/ (Day 2-4, 24h)

**Tasks**:
1. Create new structure:
   ```
   analysis/beamforming/
   ├── time_domain/
   │   ├── das.rs
   │   └── delay_sum.rs
   ├── frequency_domain/
   │   ├── mvdr.rs
   │   ├── music.rs
   │   └── capon.rs
   ├── adaptive/
   │   └── subspace.rs
   └── transmit/
       └── focused.rs
   ```

2. Migrate unique algorithms from both locations
3. Write comprehensive tests for each
4. Benchmark against original implementations

#### Step 5.3: Remove domain/sensor/beamforming (Day 5, 8h)

**Tasks**:
1. Delete `domain/sensor/beamforming/` directory (6 subdirs)
2. Delete `core/utils/sparse_matrix/beamforming.rs` (inappropriate location)
3. Update imports across codebase:
   - `clinical/imaging/` → `analysis/beamforming/`
   - `domain/source/transducers/` → `analysis/beamforming/transmit/`

#### Step 5.4: Integration Testing (Day 6-7, 16h)

**Tasks**:
1. End-to-end beamforming tests
2. PAM integration tests
3. Clinical workflow tests
4. Performance benchmarks

**Deliverables**:
- ✅ Single source of truth: `analysis/beamforming/`
- ✅ Zero beamforming duplication
- ✅ All tests passing
- ✅ No performance regression

---

### Sprint 6-7: Clinical Consolidation (Week 6-7 - 64 hours)

**Priority**: MEDIUM  
**Affected Files**: 25+  
**Breaking Changes**: MODERATE (module moves)

#### Step 6.1: Define Clinical Boundary (Week 6, Day 1-2, 16h)

**Principle**: Clinical = "What to do medically", Physics = "How waves behave"

**Tasks**:
1. Audit all files in:
   - `clinical/imaging/`
   - `clinical/therapy/`
   - `physics/acoustics/imaging/modalities/`
   - `physics/acoustics/therapy/`

2. Classify each file:
   - Clinical application → Keep/move to `clinical/`
   - Core physics → Keep in `physics/acoustics/`
   - Unclear → Document for team review

#### Step 6.2: Move Imaging Modalities (Week 6, Day 3-5, 24h)

**Tasks**:
1. `physics/acoustics/imaging/modalities/ceus/` → `clinical/imaging/modalities/ceus/`
2. `physics/acoustics/imaging/modalities/elastography/` → `clinical/imaging/modalities/elastography/`
3. `physics/acoustics/imaging/modalities/ultrasound/hifu/` → `clinical/therapy/hifu/`

**For each move**:
- Update imports
- Run tests
- Update documentation

#### Step 6.3: Move Therapy Techniques (Week 7, Day 1-3, 24h)

**Tasks**:
1. `physics/acoustics/therapy/` → `clinical/therapy/techniques/`
   - Keep: Cavitation models in `physics/acoustics/mechanics/cavitation/`
   - Move: Therapy protocols, metrics, parameters

2. Verify separation:
   - `physics/acoustics/` contains NO clinical decision logic
   - `clinical/` imports physics models, doesn't define them

**Verification**:
```bash
cargo test --lib clinical
cargo test --lib physics::acoustics
xtask check-architecture
```

**Deliverables**:
- ✅ Clear clinical/physics separation
- ✅ Clinical workflows in `clinical/`
- ✅ Physics models in `physics/acoustics/`
- ✅ Zero clinical logic in physics modules

---

### Sprint 8: Math/Numerics Consolidation (Week 8 - 40 hours)

**Priority**: MEDIUM  
**Affected Files**: 20+  
**Breaking Changes**: YES (operator imports)

#### Step 8.1: Consolidate Operators (Day 1-3, 24h)

**Tasks**:
1. Move to `math/numerics/operators/`:
   - `domain/grid/operators/` → `math/numerics/operators/differential.rs`
   - `solver/forward/fdtd/numerics/finite_difference.rs` → `math/numerics/operators/finite_difference.rs`
   - `solver/forward/pstd/numerics/operators/` → `math/numerics/operators/spectral.rs`

2. Define operator traits:
   ```rust
   pub trait DifferentialOperator {
       fn gradient(&self, field: &Field, grid: &dyn GridTopology) -> Field;
       fn divergence(&self, field: &Field, grid: &dyn GridTopology) -> Field;
       fn laplacian(&self, field: &Field, grid: &dyn GridTopology) -> Field;
   }
   ```

3. Implementations:
   - `FiniteDifferenceOperator`
   - `SpectralOperator`
   - `DGOperator`

#### Step 8.2: Update Solver Consumption (Day 4-5, 16h)

**Tasks**:
1. FDTD: Use `math::numerics::operators::FiniteDifferenceOperator`
2. PSTD: Use `math::numerics::operators::SpectralOperator`
3. DG: Use `math::numerics::operators::DGOperator`

**Verification**:
```bash
cargo test --lib math::numerics
cargo test --lib solver::forward
cargo bench grid_benchmarks
```

**Deliverables**:
- ✅ All operators in `math/numerics/operators/`
- ✅ Trait-based operator interface
- ✅ Solvers consume, never implement
- ✅ No performance regression

---

## Phase 3: Documentation & Stabilization (Weeks 9-10)

### Sprint 9: Documentation Consolidation (Week 9 - 40 hours)

**Priority**: LOW  
**Affected Files**: Documentation only  
**Breaking Changes**: NONE

#### Step 9.1: Root Cleanup (Day 1, 8h)

**Execute**: `./scripts/cleanup_root_docs.sh`

**Manual tasks**:
1. Review moved files for accuracy
2. Update links in `README.md`
3. Verify `docs/architecture/README.md` is accurate

#### Step 9.2: Architecture Diagrams (Day 2-3, 16h)

**Tasks**:
1. Create layer dependency diagram (Graphviz/Mermaid)
2. Create module relationship diagram
3. Add to `docs/architecture/diagrams/`

#### Step 9.3: Migration Guide (Day 4-5, 16h)

**Create**: `docs/migration/v2.14_to_v2.15.md`

**Sections**:
1. Breaking Changes Summary
2. Import Path Changes (table format)
3. API Changes (with examples)
4. Deprecated Items (with timeline)
5. Migration Scripts (if applicable)

---

### Sprint 10: Stabilization & Release (Week 10 - 40 hours)

**Priority**: P0  
**Objective**: Ensure production readiness

#### Step 10.1: Comprehensive Testing (Day 1-3, 24h)

**Tasks**:
1. Run full test suite:
   ```bash
   cargo test --all-features
   cargo test --lib
   cargo test --tests
   ```

2. Run TIER 3 comprehensive validation:
   ```bash
   cargo test --all-features physics_validation
   cargo test --all-features literature_validation
   ```

3. Fix all test failures

#### Step 10.2: Performance Validation (Day 4, 8h)

**Tasks**:
1. Run all benchmarks:
   ```bash
   cargo bench --all-features
   ```

2. Compare to baseline (pre-refactoring)
3. Document any regressions
4. Optimize if necessary

#### Step 10.3: Documentation Audit (Day 5, 8h)

**Tasks**:
1. Run `cargo doc --all-features --no-deps --open`
2. Verify all public APIs documented
3. Check for broken intra-doc links
4. Update examples in `examples/`

#### Step 10.4: Release Preparation (Week 10, remaining hours)

**Tasks**:
1. Update `CHANGELOG.md` with all breaking changes
2. Bump version: `2.14.0` → `2.15.0`
3. Tag release: `git tag v2.15.0`
4. Create GitHub release with migration guide

---

## Phase 4: Continuous Enforcement (Ongoing)

### CI/CD Integration

**Add to `.github/workflows/ci.yml`**:

```yaml
architecture-check:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    - name: Check Architecture
      run: |
        cd xtask
        cargo run -- check-architecture --strict
```

**Pre-commit Hook**:

```bash
# .git/hooks/pre-commit
#!/bin/bash
cd xtask
cargo run -- check-architecture
if [ $? -ne 0 ]; then
    echo "❌ Architecture violations detected. Fix before committing."
    exit 1
fi
```

---

## Risk Management

### High-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes block users | HIGH | HIGH | Deprecation period, migration guide, adapter layers |
| Performance regression | MEDIUM | HIGH | Benchmark every change, optimize before release |
| Test failures during refactoring | HIGH | MEDIUM | Small atomic commits, rollback strategy |
| Team availability | MEDIUM | MEDIUM | Document everything, enable async work |

### Rollback Strategy

**For each sprint**:
1. Create branch: `refactor/sprint-N-<name>`
2. Commit atomically (1 logical change per commit)
3. Tag working states: `sprint-N-checkpoint-M`
4. If blocked: Revert to last checkpoint, reassess

---

## Success Criteria

### Quantitative

- [ ] Zero circular dependencies (measured by `cargo-depgraph`)
- [ ] <500 duplicate LOC (measured by clone detection tools)
- [ ] Build time: clean <60s, incremental <5s
- [ ] All modules <500 lines (GRASP compliant)
- [ ] Zero architecture violations (xtask check-architecture)

### Qualitative

- [ ] New contributor can understand structure in <1 hour
- [ ] Adding new solver touches only `solver/forward/<new>/`
- [ ] Adding new physics touches only `physics/<new>/`
- [ ] Clear ownership of every module
- [ ] Documentation matches reality

---

## Resource Requirements

### Time Allocation

| Phase | Duration | Effort (hours) | FTE @ 40h/week |
|-------|----------|----------------|----------------|
| Phase 0 | 3 days | 24h | 0.6 weeks |
| Phase 1 | 4 weeks | 176h | 4.4 weeks |
| Phase 2 | 4 weeks | 200h | 5.0 weeks |
| Phase 3 | 2 weeks | 80h | 2.0 weeks |
| **Total** | **10.6 weeks** | **480h** | **12 weeks @ 40h/week** |

### Team Requirements

**Minimum**: 1 senior engineer (full-time)  
**Recommended**: 1 senior + 1 mid-level (pair programming)  
**Optimal**: 2 senior engineers (parallel work streams)

---

## Timeline (Gantt Chart)

```
Week 0:  [Prep] Infrastructure setup, documentation
Week 1:  [Phase 1] Grid consolidation
Week 2:  [Phase 1] Boundary consolidation
Week 3:  [Phase 1] Medium consolidation (part 1)
Week 4:  [Phase 1] Medium consolidation (part 2)
Week 5:  [Phase 2] Beamforming consolidation
Week 6:  [Phase 2] Clinical consolidation (part 1)
Week 7:  [Phase 2] Clinical consolidation (part 2)
Week 8:  [Phase 2] Math/Numerics consolidation
Week 9:  [Phase 3] Documentation consolidation
Week 10: [Phase 3] Stabilization & release
Week 11: [Phase 4] CI/CD integration, monitoring
```

---

## Approval & Sign-off

**Technical Lead**: _________________________ Date: _______

**Product Owner**: _________________________ Date: _______

**Stakeholders Notified**: YES / NO

**Go/No-Go Decision**: GO / NO-GO

---

## References

- [Architectural Audit](ARCHITECTURAL_AUDIT_SPRINT_ANALYSIS.md) - Detailed analysis
- [Architecture Decision Records](docs/adr.md) - ADR-001 through ADR-022
- [Software Requirements](docs/srs.md) - NFR-004 (GRASP compliance)
- [Product Requirements](docs/prd.md) - Maintainability requirements

---

**Status**: READY FOR EXECUTION  
**Next Action**: Obtain approval, execute Phase 0