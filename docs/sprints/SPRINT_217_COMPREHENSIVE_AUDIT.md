# Sprint 217: Comprehensive Architectural Audit & Deep Optimization

**Date**: 2026-02-04  
**Sprint Duration**: 4-6 sessions (16-24 hours estimated)  
**Priority**: P0 - Foundation for all future work  
**Status**: 🔄 IN PROGRESS - Session 1

---

## Executive Summary

### Mission

Conduct a comprehensive architectural audit of the kwavers ultrasound/optics simulation library to:

1. **Eliminate Circular Dependencies**: Identify and resolve all circular import chains violating Clean Architecture
2. **Enforce Deep Vertical Hierarchy**: Verify proper module organization with bounded contexts
3. **Remove Dead/Deprecated Code**: Achieve 100% clean codebase with zero technical debt
4. **Optimize Module Boundaries**: Ensure single source of truth (SSOT) across all domains
5. **Prepare for Research Integration**: Establish clean foundation for k-Wave, jwave, and advanced algorithms

### Current State Baseline (2026-02-04)

**✅ Strengths:**
- Zero compilation errors (clean baseline)
- 2009/2009 tests passing (100% pass rate)
- Recent sprint success (Sprints 213-216)
- 9-layer Clean Architecture defined
- 1303 source files organized

**🔴 Critical Issues:**
- Circular dependencies mentioned by user (needs verification)
- Potential cross-contamination between modules
- Module organization needs systematic verification
- Research integration opportunities unexplored

**📊 Metrics:**
- Total source files: 1,303
- Module depth: 9 layers (Core → Math → Domain → Physics → Solver → Simulation → Clinical → Analysis → Infrastructure)
- Test coverage: 100% passing
- Build time: ~12-30s (varies by features)

---

## Phase 1: Architectural Dependency Audit (Sessions 1-2, 6-8 hours)

### Session 1 Objectives (Current)

**1.1: Module Dependency Graph Analysis (2-3 hours)**

Generate comprehensive dependency graph:
```bash
# Layer-by-layer dependency analysis
cargo tree --prefix depth --edges normal > dependency_graph.txt

# Circular dependency detection
cargo clippy -- -W clippy::module_inception

# Import pattern analysis
grep -r "^use crate::" src/ > import_patterns.txt
```

**Expected Outputs:**
- `dependency_graph.txt`: Full crate dependency tree
- `import_patterns.txt`: All internal module imports
- `circular_deps.md`: Identified circular dependency chains
- `layer_violations.md`: Upward dependencies violating architecture

**1.2: Single Source of Truth (SSOT) Verification (2-3 hours)**

Audit SSOT compliance for critical domains:

**Core Constants** (`src/physics/constants/`):
- ✅ Physical constants (fundamental.rs)
- ✅ Material properties (water, tissue, etc.)
- ⚠️ Verify no duplication in domain/medium/

**Field Indices** (`src/domain/field/indices.rs`):
- ✅ PRESSURE_IDX, VX_IDX, VY_IDX, VZ_IDX defined
- ⚠️ Check for alternative index definitions in physics/solver/

**Grid/Medium Definitions** (`src/domain/`):
- ✅ Grid: domain/grid/
- ✅ Medium: domain/medium/
- ⚠️ Check for duplicate grid utilities in solver/physics/

**Source/Sensor Definitions** (`src/domain/`):
- ✅ Source: domain/source/
- ✅ Sensor: domain/sensor/
- ⚠️ Check for duplicates in analysis/clinical/

**1.3: Duplication Detection (1-2 hours)**

Systematic duplication analysis:

```bash
# Code duplication detection (structural)
find src -name "*.rs" -exec sha256sum {} \; | sort | uniq -d -w 64

# Semantic duplication (functions/structs)
rg "^(pub )?fn " src/ --no-heading | sort | uniq -d
rg "^(pub )?struct " src/ --no-heading | sort | uniq -d
rg "^(pub )?trait " src/ --no-heading | sort | uniq -d
```

**Expected Outputs:**
- `duplication_report.md`: All identified duplications
- `refactor_candidates.md`: Functions/structs needing consolidation

### Session 2 Objectives (Planned)

**2.1: Layer Boundary Enforcement Audit**

Verify dependency rules:

**✅ ALLOWED Dependencies:**
```
Infrastructure → Analysis → Clinical → Simulation → Solver → Physics → Domain → Math → Core
     ↓              ↓          ↓           ↓          ↓         ↓         ↓       ↓      ↓
    GPU         Plotting   Imaging      Factory   FDTD/PSTD  Acoustic   Grid   FFT   Error
```

**❌ FORBIDDEN Dependencies:**
- Core → any layer above
- Math → Domain/Physics/Solver/...
- Domain → Physics/Solver/...
- Physics → Solver/Simulation/...
- Solver → Simulation/Clinical/...

**2.2: Bounded Context Verification**

Verify bounded contexts are properly isolated:

1. **Spatial Context** (`domain/grid/`, `domain/geometry/`, `domain/mesh/`)
2. **Material Context** (`domain/medium/`)
3. **Sensing/Sourcing Context** (`domain/sensor/`, `domain/source/`)
4. **Signal Context** (`domain/signal/`, `domain/signal_processing/`)
5. **Physics Context** (`physics/acoustics/`, `physics/optics/`, `physics/thermal/`)
6. **Solver Context** (`solver/fdtd/`, `solver/pstd/`, `solver/pinn/`)
7. **Clinical Context** (`clinical/imaging/`, `clinical/therapy/`, `clinical/safety/`)
8. **Analysis Context** (`analysis/signal_processing/`, `analysis/ml/`, `analysis/performance/`)

**Expected Output:**
- `bounded_context_audit.md`: Verification of context isolation
- `boundary_violations.md`: Cross-context leakage

---

## Phase 2: Code Quality Baseline (Sessions 2-3, 4-6 hours)

### 2.1: Dead Code Elimination

**Strategy:**
1. Run `cargo check --lib` with full warnings
2. Analyze unused imports, functions, types
3. Verify if "dead" code is:
   - Truly unused → Remove
   - Public API → Document and keep
   - Future API → Add `#[allow(dead_code)]` with justification

**Expected Metrics:**
- Current: ~8 warnings in benchmarks/tests
- Target: 0 warnings in production code (`src/`)
- Acceptable: Warnings only in `benches/` and `tests/` with justification

### 2.2: Warning Audit & Remediation

**Current Warnings (from diagnostics):**
```
benches/pinn_performance_benchmarks.rs: 8 warnings
tests/validation/mod.rs: 6 warnings
benches/performance_benchmark.rs: 10 warnings
tests/property_based_tests.rs: 4 warnings
benches/pinn_vs_fdtd_benchmark.rs: 3 warnings
tests/validation/energy.rs: 8 warnings
tests/validation/error_metrics.rs: 1 warning
tests/validation/convergence.rs: 3 warnings
```

**Total**: 43 warnings in benchmarks/tests (acceptable if justified)

**Action Items:**
- ✅ Verify all warnings are in `benches/` or `tests/`
- ✅ Ensure zero warnings in `src/`
- ⚠️ Document reason for test/bench warnings (incomplete stubs, future work)
- ⚠️ Remove or complete any placeholder tests/benchmarks

### 2.3: Unsafe Code Audit

**Current Unsafe Usage:**
```bash
rg "unsafe " src/ --stats
```

**Audit Criteria:**
- Every `unsafe` block must have:
  - Inline comment explaining why it's needed
  - Invariants that must be upheld
  - References to performance requirements justifying it
  - Alternative safe implementations considered

**Expected Output:**
- `unsafe_audit.md`: All unsafe blocks with justification
- Target: Minimize unsafe usage, justify remaining blocks

---

## Phase 3: Module Reorganization & Optimization (Sessions 3-4, 6-8 hours)

### 3.1: Deep Vertical Hierarchy Completion

**Current File Structure:**
```
src/
├── core/                    [✅ Foundational - 2 layers deep]
├── math/                    [✅ Primitives - 2-3 layers deep]
├── domain/                  [⚠️ Verify depth - should be 3-4 layers]
│   ├── grid/
│   ├── medium/
│   ├── sensor/
│   ├── source/
│   ├── signal/
│   └── signal_processing/  [⚠️ May belong in analysis/?]
├── physics/                 [✅ Good depth - 3-4 layers]
├── solver/                  [✅ Good depth - 3-4 layers]
├── simulation/              [⚠️ Verify organization]
├── clinical/                [⚠️ Verify organization]
├── analysis/                [⚠️ Very deep - verify logical grouping]
└── infrastructure/          [✅ HAL layer]
```

**Key Decisions:**
1. **Domain/signal_processing vs Analysis/signal_processing**
   - Current: Split between both locations
   - Decision needed: Domain should only have signal definitions (Signal, SineWave, etc.)
   - Analysis should have processing algorithms (filtering, beamforming, etc.)

2. **Physics layer depth**
   - Current: physics/{acoustics, optics, thermal, chemistry, electromagnetic}
   - Each subdomain should have 2-3 layers (e.g., acoustics/{mechanics, bubble_dynamics, radiation_force})

3. **Solver organization**
   - Current: solver/{fdtd, pstd, pinn, forward/nonlinear, integration}
   - Verify: No overlap between forward/nonlinear and top-level solvers

### 3.2: File Size Audit

**Target**: No file > 800 lines (excluding generated code)

**Large Files from Previous Sprints:**
- Most large files already refactored in Sprints 193-206
- Verify: Run file size audit
  ```bash
  find src -name "*.rs" -type f -exec wc -l {} \; | sort -rn | head -20
  ```

**Expected Output:**
- `large_files.md`: Files > 800 lines requiring refactoring
- Refactoring plan for each large file

---

## Phase 4: Research Integration Preparation (Sessions 4-5, 4-6 hours)

### 4.1: k-Wave Algorithm Integration Opportunities

**Priority Algorithms from k-Wave (MATLAB):**

1. **k-space Pseudospectral Methods** (HIGH PRIORITY)
   - Current: Basic PSTD in `solver/pstd/`
   - k-Wave: Advanced k-space correction, exact time reversal
   - Integration point: `solver/pstd/kspace_correction.rs`
   - Estimated effort: 12-16 hours

2. **Elastic Wave Propagation** (HIGH PRIORITY)
   - Current: Basic elastic wave in `physics/acoustics/mechanics/elastic_wave/`
   - k-Wave: Full elastic tensor, mode conversion
   - Integration point: `physics/acoustics/mechanics/elastic_wave/kwave_elastic.rs`
   - Estimated effort: 16-20 hours

3. **Advanced Source Modeling** (MEDIUM PRIORITY)
   - Current: Basic point/grid sources in `domain/source/`
   - k-Wave: Time-varying sources, focused transducers
   - Integration point: `domain/source/kwave_sources.rs`
   - Estimated effort: 8-12 hours

4. **Absorption Models** (HIGH PRIORITY)
   - Current: Basic absorption in `physics/acoustics/absorption.rs`
   - k-Wave: Frequency-dependent power law absorption
   - Integration point: Already implemented, verify against k-Wave
   - Estimated effort: 4-6 hours (validation)

### 4.2: jwave (JAX) Integration Opportunities

**Priority Features from jwave:**

1. **GPU Parallelization via BURN** (CRITICAL)
   - Current: Basic WGPU support in `gpu/`
   - jwave: Full GPU parallelization with JAX
   - Integration point: `gpu/burn_integration.rs`
   - Estimated effort: 20-24 hours

2. **Automatic Differentiation for PINNs** (HIGH PRIORITY)
   - Current: Basic PINN in `solver/pinn/`
   - jwave: Full autodiff with JAX
   - Integration point: Use BURN autodiff features
   - Estimated effort: 12-16 hours

3. **Differentiable Simulations** (MEDIUM PRIORITY)
   - Current: Forward simulation only
   - jwave: Gradient-based optimization
   - Integration point: `solver/inverse/` (new module)
   - Estimated effort: 16-20 hours

### 4.3: Other Research Project Integration

**k-wave-python** (Python bindings):
- Lessons: HDF5 I/O standards, API design patterns
- Integration: `infrastructure/io/hdf5_standard.rs`
- Estimated effort: 6-8 hours

**optimus** (Optimization framework):
- Lessons: Inverse problem formulations
- Integration: `solver/inverse/optimus_algorithms.rs`
- Estimated effort: 8-12 hours

**fullwave25** (Full-wave simulation):
- Lessons: Clinical workflows, HIFU modeling
- Integration: `clinical/therapy/hifu/fullwave.rs`
- Estimated effort: 12-16 hours

**dbua** (Neural beamforming):
- Lessons: Real-time inference, neural architectures
- Integration: Already in `analysis/signal_processing/beamforming/neural/`
- Status: ✅ Partially implemented, needs validation
- Estimated effort: 4-6 hours (validation)

**simsonic** (Advanced tissue models):
- Lessons: Multi-modal integration, tissue acoustics
- Integration: `domain/medium/simsonic_models.rs`
- Estimated effort: 8-12 hours

---

## Phase 5: Testing & Validation Enhancement (Sessions 5-6, 4-6 hours)

### 5.1: Mathematical Correctness Verification

**Validation Strategy:**

1. **Analytical Solutions** (HIGH PRIORITY)
   - Verify FDTD/PSTD against known solutions
   - Tests: Plane wave propagation, point source radiation, reflection coefficients
   - Location: `tests/validation/analytical_solutions.rs`
   - Current status: Partial coverage
   - Target: 100% coverage of core solvers

2. **Literature Benchmarks** (HIGH PRIORITY)
   - Implement benchmark problems from key papers
   - Sources: Treeby & Cox (2010), Hamilton & Blackstock (1998)
   - Location: `tests/validation/literature_benchmarks.rs`
   - Current status: Incomplete
   - Target: 20+ literature validation tests

3. **Property-Based Testing** (MEDIUM PRIORITY)
   - Proptest for physics invariants (energy conservation, CFL stability)
   - Location: `tests/property_based_tests.rs`
   - Current status: ✅ Implemented
   - Target: Expand to all physics modules

### 5.2: Performance Benchmarking

**Benchmark Suite Requirements:**

1. **Baseline Metrics**
   - Grid operations (creation, indexing, operators)
   - Medium property access (sound speed, density)
   - Solver stepping (FDTD, PSTD, PINN)
   - Beamforming algorithms (DAS, MVDR, MUSIC)

2. **Optimization Tracking**
   - SIMD vectorization effectiveness
   - GPU acceleration speedup
   - Memory allocation patterns
   - Cache utilization

3. **Regression Prevention**
   - Store baseline results
   - Alert on performance degradation > 10%
   - Document all optimization decisions

**Current Benchmark Status:**
- Location: `benches/`
- Files: 15+ benchmark files
- Status: ⚠️ Some stubs exist (from Sprint 209 audit)
- Target: All stubs replaced with real benchmarks

---

## Sprint 217 Session-by-Session Plan

### Session 1: Dependency Audit & SSOT Verification ✅ IN PROGRESS (3-4 hours)

**Tasks:**
1. ✅ Generate dependency graph
2. ✅ Import pattern analysis
3. 🔄 Circular dependency detection
4. 🔄 SSOT violation identification
5. 🔄 Duplication analysis

**Deliverables:**
- This document (Sprint 217 audit plan)
- `dependency_audit.md` (detailed findings)
- `ssot_violations.md` (identified violations)
- `duplication_report.md` (code duplications)

### Session 2: Layer Boundary Enforcement & Dead Code (4-5 hours)

**Tasks:**
1. Layer dependency verification
2. Bounded context audit
3. Dead code elimination
4. Warning remediation (production code)

**Deliverables:**
- `layer_violations.md` (dependency rule violations)
- `bounded_context_audit.md` (context isolation verification)
- Clean `cargo check --lib` output (zero warnings in src/)

### Session 3: Module Reorganization (5-6 hours)

**Tasks:**
1. Domain/signal_processing migration decision
2. Large file refactoring (if any > 800 lines)
3. Module structure optimization
4. API surface cleanup

**Deliverables:**
- Updated module structure
- Refactored large files
- Clean import paths
- Updated ARCHITECTURE.md

### Session 4: Research Integration Roadmap (3-4 hours)

**Tasks:**
1. k-Wave algorithm prioritization
2. jwave/BURN integration planning
3. Other research project assessment
4. Implementation effort estimation

**Deliverables:**
- `research_integration_roadmap.md` (6-12 month plan)
- Priority matrix (impact vs effort)
- Implementation specifications for top 3 priorities

### Session 5: Testing & Validation Enhancement (4-5 hours)

**Tasks:**
1. Analytical solution tests
2. Literature benchmark implementation
3. Benchmark stub remediation
4. Performance baseline establishment

**Deliverables:**
- 20+ new validation tests
- Complete benchmark suite
- Performance baseline report
- Test coverage report

### Session 6: Documentation & Closure (2-3 hours)

**Tasks:**
1. Update all documentation (README, ARCHITECTURE, ADRs)
2. Sync backlog.md and checklist.md
3. Sprint retrospective
4. Next sprint planning

**Deliverables:**
- Updated documentation
- Sprint 217 completion report
- Sprint 218+ roadmap

---

## Success Metrics

### Hard Criteria (Must Meet):

1. **Zero Circular Dependencies**: No circular import chains at any layer
2. **Zero SSOT Violations**: Single source for all constants, indices, definitions
3. **Zero Production Warnings**: `cargo check --lib` produces zero warnings in `src/`
4. **100% Test Pass Rate**: All 2009+ tests passing
5. **Layer Compliance**: All dependencies follow architecture rules

### Soft Criteria (Should Meet):

1. **Deep Hierarchy**: All modules 2-4 layers deep (no flat structures)
2. **File Size**: No file > 800 lines (excluding generated code)
3. **Duplication**: < 1% code duplication (structural)
4. **Documentation**: All modules have comprehensive rustdoc
5. **Benchmark Coverage**: All critical paths benchmarked

### Performance Targets:

1. **Build Time**: < 15s for `cargo check --lib` (currently ~12-30s)
2. **Test Time**: < 60s for `cargo test --lib` (currently varies)
3. **Memory**: No memory leaks in validation tests

---

## Risk Assessment

### High Risk:

1. **Circular Dependency Resolution**
   - Risk: Breaking API changes may be required
   - Mitigation: Identify all dependencies first, plan refactoring carefully
   - Fallback: Accept some technical debt with clear documentation

2. **Module Reorganization**
   - Risk: Large-scale file moves may break imports
   - Mitigation: Use `cargo fix` and systematic testing
   - Fallback: Incremental migration over multiple sprints

### Medium Risk:

1. **Performance Regression**
   - Risk: Refactoring may impact performance
   - Mitigation: Benchmark before/after, profile critical paths
   - Fallback: Rollback specific changes

2. **Test Suite Changes**
   - Risk: Refactoring may break tests
   - Mitigation: Fix tests incrementally, maintain 100% pass rate
   - Fallback: Skip non-critical test updates

### Low Risk:

1. **Documentation Updates**
   - Risk: Documentation may become stale
   - Mitigation: Update docs alongside code changes
   - Fallback: Document known gaps for future sprints

---

## References

### Architecture Standards:
- Clean Architecture (Robert C. Martin)
- Domain-Driven Design (Eric Evans)
- SOLID principles
- Bounded Context pattern

### Research Projects:
- k-Wave: https://github.com/ucl-bug/k-wave
- jwave: https://github.com/ucl-bug/jwave
- k-wave-python: https://k-wave-python.readthedocs.io/
- optimus: https://github.com/optimuslib/optimus
- fullwave25: https://github.com/pinton-lab/fullwave25
- dbua: https://github.com/waltsims/dbua
- simsonic: http://www.simsonic.fr/

### Key Papers:
- Treeby & Cox (2010) - k-Wave toolbox
- Hamilton & Blackstock (1998) - Nonlinear Acoustics
- Schmidt (1986) - MUSIC algorithm
- Raissi et al. (2019) - Physics-informed neural networks

---

## Next Steps (Post-Sprint 217)

### Sprint 218: k-Wave k-space Integration (16-20 hours)
- Implement advanced k-space correction
- Exact time reversal
- Validate against k-Wave benchmarks

### Sprint 219: BURN GPU Acceleration (20-24 hours)
- Full GPU parallelization via BURN
- Autodiff for PINN training
- Performance benchmarking

### Sprint 220: Advanced Elastic Wave Propagation (16-20 hours)
- Full elastic tensor support
- Mode conversion at interfaces
- Literature validation

---

## Appendix A: Current Module Inventory

### Core Layer (Foundation):
```
src/core/
├── error.rs          [Error types, Result<T>]
├── time.rs           [Time handling]
├── mod.rs            [Core exports]
```

### Math Layer (Primitives):
```
src/math/
├── linear_algebra/   [Matrices, vectors, eigendecomposition]
├── fft/              [Fast Fourier transforms]
├── interpolation/    [Interpolation methods]
├── special/          [Special functions]
├── differential/     [Differential operators]
└── mod.rs
```

### Domain Layer (Business Logic):
```
src/domain/
├── grid/             [Spatial discretization]
├── medium/           [Material properties]
├── sensor/           [Sensing devices]
├── source/           [Wave sources]
├── signal/           [Signal definitions]
├── signal_processing/[⚠️ May need migration]
├── boundary/         [Boundary conditions]
├── field/            [Field management]
├── geometry/         [Geometric primitives]
├── mesh/             [Mesh structures]
├── plugin/           [Plugin system]
└── mod.rs
```

### Physics Layer (Physical Models):
```
src/physics/
├── acoustics/        [Acoustic wave models]
│   ├── mechanics/
│   ├── bubble_dynamics/
│   └── absorption/
├── optics/           [Optical models]
├── thermal/          [Thermal models]
├── electromagnetic/  [EM models]
├── chemistry/        [Chemical models]
├── constants/        [Physical constants - SSOT]
└── mod.rs
```

### Solver Layer (Numerical Methods):
```
src/solver/
├── fdtd/             [Finite-difference time-domain]
├── pstd/             [Pseudospectral time-domain]
├── pinn/             [Physics-informed neural networks]
├── forward/          [Forward problem solvers]
├── integration/      [Time integration]
└── mod.rs
```

### Simulation Layer (Orchestration):
```
src/simulation/
├── configuration/    [Simulation config]
├── parameters/       [Simulation parameters]
├── factory/          [Factory patterns]
└── mod.rs
```

### Clinical Layer (Applications):
```
src/clinical/
├── imaging/          [Medical imaging]
├── therapy/          [Therapeutic ultrasound]
├── safety/           [Safety monitoring]
└── mod.rs
```

### Analysis Layer (Post-processing):
```
src/analysis/
├── signal_processing/[Signal processing algorithms]
│   ├── beamforming/
│   ├── filtering/
│   ├── localization/
│   └── pam/
├── ml/               [Machine learning]
├── performance/      [Performance optimization]
├── validation/       [Validation tools]
├── testing/          [Testing utilities]
└── mod.rs
```

### Infrastructure Layer (Hardware Abstraction):
```
src/infrastructure/
├── io/               [Input/output]
├── api/              [REST API]
├── cloud/            [Cloud providers]
└── mod.rs
```

### GPU Layer (Acceleration):
```
src/gpu/
├── kernels/          [GPU kernels]
├── thermal_acoustic/ [Thermal-acoustic GPU]
└── mod.rs
```

---

## Appendix B: Dependency Analysis Methodology

### Tools:
1. `cargo tree`: Dependency tree visualization
2. `cargo clippy`: Linting and pattern detection
3. `rg` (ripgrep): Fast text search for patterns
4. Custom scripts: Dependency graph generation

### Metrics:
1. **Layer Coupling**: Number of upward dependencies
2. **Module Cohesion**: Ratio of internal to external calls
3. **SSOT Compliance**: Number of definition sites per concept
4. **Duplication**: Percentage of duplicate code blocks

### Validation:
1. Unit tests: Test isolated modules
2. Integration tests: Test layer interactions
3. Property tests: Test invariants
4. Benchmark tests: Test performance

---

**End of Sprint 217 Comprehensive Audit Plan**