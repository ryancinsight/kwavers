# Architectural Recommendations Summary

**Date:** 2026-01-23  
**Based On:** Analysis of 11 ultrasound/optics simulation libraries  
**Full Report:** `architectural_research_ultrasound_libraries.md`

---

## Executive Summary

Analysis of leading ultrasound simulation libraries (j-Wave, k-Wave, Fullwave, BabelBrain, etc.) reveals three critical architectural patterns that directly address kwavers' current issues:

1. **Beamforming is post-processing, not domain logic** - Should live only in analysis layer
2. **Clinical workflows orchestrate, don't implement** - BabelBrain's step-based pattern
3. **Backend abstraction enables flexibility** - wgpu already provides multi-backend support

---

## Critical Issues and Solutions

### Issue 1: Beamforming Duplication

**Current State:**
- Beamforming code exists in both:
  - `domain/sensor/beamforming/` (120+ files)
  - `analysis/signal_processing/beamforming/`
- Violates Single Responsibility Principle
- Creates maintenance burden and divergence risk

**Industry Pattern:**
- **k-Wave-Python:** `reconstruction/beamform` (post-processing only)
- **Fullwave:** `examples/` (separate from solver)
- **Sound-Speed:** Dedicated module, zero solver coupling

**Recommendation:**

```
✅ KEEP: analysis/signal_processing/beamforming/
   - All algorithms (DAS, MVDR, MUSIC, neural, adaptive)
   - Signal processing traits and utilities
   - Performance-optimized implementations

❌ REMOVE: domain/sensor/beamforming/
   - Delete algorithmic implementations
   - Keep only: sensor geometry exports for beamforming
   - Stub with deprecation warnings pointing to analysis layer
```

**Migration Path:**
1. **Sprint 213:** Deprecate `domain/sensor/beamforming/` with warnings
2. **Sprint 214:** Migrate clinical workflows to use analysis layer
3. **Sprint 215:** Delete deprecated code

**Code Example:**

```rust
// domain/sensor/mod.rs (sensor geometry only)
pub struct SensorArray {
    pub positions: Array2<f64>,
    pub orientations: Array2<f64>,
}

impl SensorArray {
    /// Export geometry for beamforming (no algorithm implementation)
    pub fn geometry(&self) -> BeamformingGeometry {
        BeamformingGeometry {
            positions: self.positions.clone(),
            orientations: self.orientations.clone(),
        }
    }
}

// analysis/signal_processing/beamforming/mod.rs (all algorithms)
pub fn beamform(
    rf_data: &Array3<f32>,
    geometry: &BeamformingGeometry,
    algorithm: BeamformingAlgorithm,
) -> KwaversResult<Array3<f32>> {
    match algorithm {
        BeamformingAlgorithm::DAS => delay_and_sum(rf_data, geometry),
        BeamformingAlgorithm::MVDR => mvdr(rf_data, geometry),
        // ... all algorithms here
    }
}
```

---

### Issue 2: Clinical-Solver Coupling

**Current State:**
- `clinical/imaging/workflows/` directly imports solver implementations
- Tight coupling: `use crate::solver::forward::fdtd::FdtdSolver`
- Cannot swap solvers, difficult to test

**Industry Pattern: BabelBrain's 3-Step Workflow**

```
Step 1: Domain Preparation (Clinical)
   ↓ Data handoff (Nifti files)
Step 2: Acoustic Calculation (Physics Backend)
   ↓ Pressure fields
Step 3: Thermal Analysis (Physics Backend)
```

**Key Principles:**
- Workflow orchestrates, doesn't implement physics
- Each step isolated and reversible
- Backend solvers are libraries, not embedded code

**Recommendation:**

```rust
// ❌ BAD: clinical/imaging/workflows/advanced_imaging.rs
use crate::solver::forward::fdtd::FdtdSolver;  // Direct coupling!

impl Workflow {
    fn run(&self) {
        let solver = FdtdSolver::new(config)?;  // Hardcoded solver choice
        solver.step()?;
    }
}

// ✅ GOOD: clinical/imaging/workflows/advanced_imaging.rs
use crate::solver::plugin::PluginExecutor;  // Abstraction only

impl Workflow {
    fn run(&self) {
        // Step 1: Domain preparation (clinical responsibility)
        let domain = self.prepare_domain()?;
        
        // Step 2: Acoustic calculation (delegate to solver abstraction)
        let executor = PluginExecutor::new(self.solver_config)?;
        let pressure = executor.execute(&domain)?;
        
        // Step 3: Beamforming (delegate to analysis layer)
        let image = crate::analysis::signal_processing::beamforming::beamform(
            &pressure,
            &self.beamform_config,
        )?;
        
        Ok(image)
    }
}
```

**Benefits:**
- Workflows testable without solver implementation (mock executor)
- Solver swappable via configuration (FDTD → PSTD → BEM)
- Follows BabelBrain's proven pattern

**Migration Path:**
1. **Sprint 214:** Refactor all clinical workflows to use `PluginExecutor`
2. **Sprint 214:** Remove direct solver imports from clinical modules
3. **Sprint 215:** Add integration tests for end-to-end workflows

---

### Issue 3: Module Boundary Violations

**Current State:**
- No automated enforcement of dependency rules
- Ad-hoc imports across layers
- Risk of circular dependencies

**Industry Pattern: Hexagonal Architecture**

```
Allowed Dependencies (Downward Only):
Clinical → Solver (via PluginExecutor) → Physics → Math
    ↓          ↓                            ↓
Analysis → Domain ────────────────────────→ Math
```

**Recommendation: Implement Architecture Lints**

```rust
// src/architecture.rs (extend existing module)
pub struct DependencyChecker;

impl DependencyChecker {
    pub fn check_module(module: &str) -> Result<(), Vec<String>> {
        match module {
            "domain" => {
                // Domain CANNOT depend on solver, analysis, clinical
                check_no_imports(module, &["solver", "analysis", "clinical"])?;
            }
            "physics" => {
                // Physics CANNOT depend on domain (only math)
                check_no_imports(module, &["domain", "solver", "analysis", "clinical"])?;
            }
            "solver" => {
                // Solver CANNOT depend on clinical or analysis
                check_no_imports(module, &["clinical", "analysis"])?;
            }
            "clinical" => {
                // Clinical can only use PluginExecutor (not direct solver imports)
                check_no_direct_solver_imports(module)?;
            }
            _ => {}
        }
        Ok(())
    }
}

// Add to CI/CD: cargo test --test architecture_tests
```

**Migration Path:**
1. **Sprint 215:** Implement `DependencyChecker`
2. **Sprint 215:** Add CI/CD check for layering violations
3. **Sprint 216:** Document allowed dependency graph in `architecture.md`

---

## Secondary Recommendations

### 4. Tissue Property Database (Priority: Medium)

**Current State:** Tissue properties scattered as constants

**Industry Pattern:** OptimUS uses IT'IS Foundation Database V4.1

**Implementation:**

```rust
// src/domain/medium/tissue_database.rs
pub struct TissueDatabase {
    tissues: HashMap<String, TissueProperties>,
}

impl TissueDatabase {
    pub fn load_itis_foundation() -> KwaversResult<Self> {
        // Load IT'IS Foundation database
        // https://itis.swiss/virtual-population/tissue-properties/
    }
}

// Usage
let db = TissueDatabase::load_itis_foundation()?;
let brain = db.get("brain_grey_matter").unwrap();
let medium = Medium::from_tissue(brain, &grid)?;
```

**Data Source:** IT'IS Foundation Tissue Properties Database V4.1  
**Format:** JSON or TOML for serialization  
**Sprint:** 217

---

### 5. Regression Testing Framework (Priority: High)

**Current State:** No numerical baselines for regression testing

**Industry Pattern:** BabelBrain's multi-axis validation
- Unit tests (algorithms)
- Regression tests (numerical baselines)
- Cross-validation (against k-Wave, Fullwave)
- Experimental validation (physical measurements)

**Implementation:**

```rust
// tests/regression/mod.rs
use kwavers::testing::RegressionTest;

#[test]
fn regression_fdtd_homogeneous() {
    let test = RegressionTest::load("fdtd_homogeneous_v1.0.json")?;
    let result = run_fdtd_simulation(&test.config)?;
    test.assert_pressure_field_matches(&result.pressure, tolerance = 1e-6)?;
}
```

**Baselines Storage:**
- `tests/regression/baselines/` (JSON + compressed pressure fields)
- Version-tagged (v1.0.0, v1.1.0, etc.)
- SHA-256 checksums for integrity

**Cross-Validation:**
```rust
#[test]
#[ignore] // Requires external tools
fn cross_validate_against_kwave() {
    // Export config to k-Wave MATLAB
    // Run k-Wave simulation
    // Compare pressure fields (tolerance 1e-4)
}
```

**Sprint:** 216

---

### 6. Axisymmetric Solver Optimization (Priority: Low)

**Use Case:** Focused ultrasound (HIFU therapy) with cylindrical symmetry

**Benefit:** Reduce 3D → 2D, ~100x computational savings

**Industry Example:** HITU Simulator (axisymmetric beams)

**Implementation:**

```rust
// src/solver/forward/fdtd_axisymmetric.rs
pub struct FdtdAxiSymmetric {
    grid_r: Array1<f64>,  // Radial
    grid_z: Array1<f64>,  // Axial
    // 2D arrays instead of 3D
}
```

**When to Use:**
- ✅ Single-element focused transducers
- ✅ Concentric phased arrays
- ❌ Linear arrays (breaks symmetry)

**Sprint:** 218+

---

## Comparison: Kwavers vs. Industry

| Aspect | Kwavers Current | Industry Best Practice | Gap |
|--------|-----------------|------------------------|-----|
| **Beamforming Location** | Domain + Analysis | Analysis only | ❌ Duplication |
| **Clinical-Solver Coupling** | Direct imports | Abstraction (executor) | ❌ Tight coupling |
| **GPU Backend** | wgpu (good) | Multi-backend (Metal/CUDA/OpenCL) | ✅ Already solved |
| **Module Boundaries** | Informal | Enforced via lints | ⚠️ Not enforced |
| **Tissue Database** | Scattered constants | Centralized database (IT'IS) | ⚠️ Missing |
| **Regression Tests** | Benchmarks only | Numerical baselines + cross-validation | ⚠️ Incomplete |
| **Plugin System** | PluginExecutor exists | Similar to BabelBrain | ✅ Already implemented |
| **Differentiability** | PINN support | JAX auto-diff (j-Wave, DBUA) | ✅ Different approach, valid |

---

## Action Plan by Sprint

### Sprint 213: Beamforming Consolidation (Critical)
- [ ] Add deprecation warnings to `domain/sensor/beamforming/`
- [ ] Audit all beamforming algorithms exist in `analysis/signal_processing/beamforming/`
- [ ] Create migration guide in documentation
- [ ] Update examples to import from analysis layer

**Estimated Effort:** 16 hours  
**Risk:** Low (deprecation, not removal)

---

### Sprint 214: Clinical-Solver Decoupling (Critical)
- [ ] Refactor `clinical/imaging/workflows/` to use `PluginExecutor`
- [ ] Remove all direct solver imports from clinical modules
- [ ] Implement step-based workflow pattern (BabelBrain model)
- [ ] Add workflow integration tests

**Estimated Effort:** 24 hours  
**Risk:** Medium (refactoring existing workflows)

---

### Sprint 215: Architecture Enforcement (High)
- [ ] Implement `DependencyChecker` in `architecture.rs`
- [ ] Add CI/CD check for layering violations
- [ ] Document allowed dependency graph
- [ ] Delete deprecated `domain/sensor/beamforming/` code

**Estimated Effort:** 12 hours  
**Risk:** Low (tooling)

---

### Sprint 216: Regression Testing (High)
- [ ] Create `tests/regression/` directory structure
- [ ] Implement `RegressionTest` framework
- [ ] Generate baselines for FDTD, PSTD (v1.0)
- [ ] Add cross-validation test stubs (k-Wave, Fullwave)

**Estimated Effort:** 20 hours  
**Risk:** Low (additive, no breaking changes)

---

### Sprint 217: Tissue Database (Medium)
- [ ] Implement `TissueDatabase` with IT'IS Foundation data
- [ ] Add JSON/TOML serialization
- [ ] Create `Medium::from_tissue()` builder
- [ ] Document data sources and citations

**Estimated Effort:** 16 hours  
**Risk:** Low (additive feature)

---

### Sprint 218: GPU Backend Refinement (Medium)
- [ ] Expose wgpu backend selection in configuration
- [ ] Benchmark across Vulkan/Metal/DX12
- [ ] Optimize shader compilation (build-time)
- [ ] Add GPU capability detection + fallback

**Estimated Effort:** 12 hours  
**Risk:** Low (refinement of existing system)

---

## Key Metrics for Success

### Code Quality Metrics
- **Beamforming duplication:** 120+ files → 0 files in domain layer
- **Direct solver imports in clinical:** ~15 occurrences → 0
- **Module boundary violations:** Unknown → 0 (enforced by CI)

### Performance Metrics
- **Test suite execution:** <30s (SRS NFR-002 compliance)
- **GPU backend compatibility:** 1 (Vulkan) → 3+ (Vulkan/Metal/DX12)
- **Axisymmetric speedup:** N/A → ~100x for HIFU simulations

### Validation Metrics
- **Regression test coverage:** 0 baselines → 20+ scenarios
- **Cross-validation:** 0 tools → 2+ (k-Wave, Fullwave)
- **Tissue database entries:** ~10 constants → 100+ standardized tissues

---

## Conclusion

The analyzed libraries consistently demonstrate three critical patterns:

1. **Separation of Concerns:** Physics ≠ Signal Processing ≠ Clinical Workflows
   - Beamforming is post-processing (analysis layer only)
   - Clinical workflows orchestrate via abstractions
   - Solvers are backends, not embedded in high-level code

2. **Composable Architecture:** Building blocks over monoliths
   - j-Wave: Functional composition (geometry + physics + solver)
   - k-Wave: Component composition (grid + medium + source + sensor)
   - BabelBrain: Step-based workflow (domain → acoustic → thermal)

3. **Backend Abstraction:** API layer over optimized kernels
   - Fullwave: Python API + CUDA/C backend
   - BabelBrain: Python GUI + BabelViscoFDTD + Elastix + trimesh
   - kwavers: Rust API + wgpu compute shaders (already correct!)

**Kwavers is 80% aligned with industry best practices.** The critical 20% gap:
- Beamforming duplication (architectural debt)
- Clinical-solver coupling (tight dependencies)
- Missing enforcement mechanisms (lints, regression tests)

**Addressing these issues in Sprints 213-216 will elevate kwavers to industry-leading architecture.**

---

**Next Steps:**
1. Review this summary with team
2. Prioritize sprint 213-216 work
3. Begin beamforming consolidation (highest impact)
4. Iterate based on feedback

**Questions or Concerns:**
- Slack: #kwavers-architecture
- GitHub Issues: Tag with `architecture` label
- Email: architecture-review@kwavers.org (if exists)

---

**Document Version:** 1.0  
**Author:** Architecture Research Analysis  
**Reviewed By:** Pending  
**Approval Required:** Yes (before Sprint 213 kickoff)
