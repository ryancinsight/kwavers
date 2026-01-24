# Architectural Research: Index and Quick Reference

**Research Date:** 2026-01-23  
**Research Scope:** 11 ultrasound and optics simulation libraries  
**Purpose:** Inform kwavers architecture improvements

---

## Document Structure

This research is organized into four complementary documents:

### 1. [Full Research Report](architectural_research_ultrasound_libraries.md) (Main)
**Length:** ~15,000 words  
**Audience:** Architects, technical leads  
**Content:**
- Detailed analysis of all 11 libraries
- Module organization patterns
- Key algorithms and solvers
- Beamforming approaches
- Physics models
- GPU acceleration patterns
- Testing strategies
- Cross-library comparative analysis

**Read this for:** Deep understanding of industry patterns and architectural decisions.

---

### 2. [Recommendations Summary](architectural_recommendations_summary.md) (Executive)
**Length:** ~5,000 words  
**Audience:** Team leads, product managers, developers  
**Content:**
- Executive summary of findings
- Critical issues and solutions (beamforming duplication, clinical-solver coupling)
- Secondary recommendations (tissue database, regression testing)
- Action plan by sprint (213-218)
- Success metrics

**Read this for:** Quick understanding of what needs to change and why.

---

### 3. [Architecture Diagrams](architecture_diagrams.md) (Visual)
**Length:** Visual + explanatory text  
**Audience:** All team members  
**Content:**
- Current vs. recommended architecture diagrams
- Data flow illustrations
- Dependency graphs (current vs. clean)
- Module organization charts
- Clinical workflow patterns
- Code structure examples

**Read this for:** Visual understanding of architectural changes.

---

### 4. [Migration Guide](migration_guide_beamforming_consolidation.md) (Implementation)
**Length:** ~4,000 words + code examples  
**Audience:** Developers implementing changes  
**Content:**
- Step-by-step migration instructions
- Code before/after examples
- Verification checklists
- Automated migration scripts
- Common migration patterns
- Breaking changes summary

**Read this for:** Hands-on implementation guidance.

---

## Quick Reference: Key Findings

### Critical Architectural Patterns (Industry Consensus)

| Pattern | Libraries | Kwavers Status | Action |
|---------|-----------|----------------|--------|
| **Beamforming is post-processing** | k-Wave-Python, Fullwave, Sound-Speed | ❌ Split between domain/analysis | Consolidate to analysis |
| **Clinical orchestrates, doesn't implement** | BabelBrain, OptimUS | ⚠️ Some direct solver imports | Refactor to use PluginExecutor |
| **Backend abstraction** | BabelBrain, k-Wave, Fullwave | ✅ PluginExecutor exists | Enforce usage in clinical layer |
| **Component composition** | j-Wave, k-Wave | ✅ Grid + Medium + Source + Sensor | Already good |
| **Multi-backend GPU** | BabelBrain (Metal/OpenCL/CUDA) | ✅ wgpu handles this | Expose configuration |
| **Physics-first ML** | Sound-Speed, DBUA | ✅ PINN approach aligns | Continue current path |
| **Tissue property database** | OptimUS (IT'IS Foundation) | ❌ Scattered constants | Implement standardized database |
| **Regression testing** | BabelBrain (numerical baselines) | ⚠️ Benchmarks only | Add baseline comparisons |

---

## Critical Issues Addressed

### Issue 1: Beamforming Duplication
**Files affected:** 120+ in `domain/sensor/beamforming/`  
**Industry pattern:** k-Wave-Python's `reconstruction/beamform`  
**Solution:** Consolidate to `analysis/signal_processing/beamforming/`  
**Sprints:** 213-215  
**Impact:** High - eliminates maintenance burden, enforces proper layering

### Issue 2: Clinical-Solver Coupling
**Files affected:** `clinical/imaging/workflows/`, `clinical/therapy/`  
**Industry pattern:** BabelBrain's step-based workflow  
**Solution:** Use `PluginExecutor` abstraction, remove direct solver imports  
**Sprints:** 214  
**Impact:** High - enables solver swapping, improves testability

### Issue 3: Module Boundary Violations
**Issue:** No enforcement of dependency rules  
**Industry pattern:** Hexagonal architecture with enforced boundaries  
**Solution:** Implement `DependencyChecker` in `architecture.rs`  
**Sprints:** 215  
**Impact:** Medium - prevents future architectural drift

---

## Libraries Analyzed

| # | Library | Language | Focus | Key Takeaway |
|---|---------|----------|-------|--------------|
| 1 | **j-Wave** | Python (JAX) | Differentiable acoustics | Functional composition, JAX auto-diff |
| 2 | **k-Wave** | MATLAB | k-space pseudospectral | Backend substitution (MATLAB + C++/CUDA) |
| 3 | **k-Wave-Python** | Python | Python k-Wave bindings | **Beamforming in `reconstruction/` only** |
| 4 | **OptimUS** | Python | BEM Helmholtz solver | **IT'IS tissue database integration** |
| 5 | **Fullwave** | Python + CUDA | High-order FDTD | **Beamforming in `examples/` only** |
| 6 | **Sound-Speed** | MATLAB | Coherence optimization | **Physics-first with optional ML** |
| 7 | **DBUA** | Python (JAX) | Differentiable beamforming | Hybrid physics-learned approach |
| 8 | **Kranion** | Java + GLSL | Transcranial planning | Plugin architecture, DICOM integration |
| 9 | **mSOUND** | MATLAB | Westervelt equation | Dual solver strategy (TMDM/FSMDM) |
| 10 | **HITU** | MATLAB | HIFU therapy | Axisymmetric optimization |
| 11 | **BabelBrain** | Python | **MRI-guided HIFU** | **Step-based workflow (industry best practice)** |

**Most influential:** BabelBrain (clinical workflow), k-Wave-Python (beamforming location), j-Wave (composable design)

---

## Action Plan Summary

### Sprint 213: Beamforming Consolidation (Critical)
**Estimated effort:** 16 hours  
**Files changed:** ~30-40 (deprecation warnings)  
**Breaking changes:** No (deprecation only)

**Tasks:**
- Add deprecation warnings to `domain/sensor/beamforming/`
- Create transitional delegation functions
- Update migration guide
- Verify all algorithms exist in analysis layer

**Deliverables:**
- Deprecation warnings active
- Migration guide published
- v0.8.0 release with deprecations

---

### Sprint 214: Clinical-Solver Decoupling (Critical)
**Estimated effort:** 24 hours  
**Files changed:** ~20-30 (clinical workflows)  
**Breaking changes:** No (internal refactoring)

**Tasks:**
- Refactor `clinical/imaging/workflows/` to use `PluginExecutor`
- Remove direct solver imports
- Implement step-based workflow pattern
- Update examples and tests

**Deliverables:**
- All clinical workflows use abstraction
- No direct solver imports
- Integration tests pass

---

### Sprint 215: Architecture Enforcement + Cleanup (High)
**Estimated effort:** 12 hours  
**Files changed:** ~10 (architecture tools) + deletions  
**Breaking changes:** Yes (remove deprecated code)

**Tasks:**
- Implement `DependencyChecker` in `architecture.rs`
- Add CI/CD check for layering violations
- Delete deprecated `domain/sensor/beamforming/` code
- Update documentation

**Deliverables:**
- Architecture lints active in CI
- Deprecated code removed
- v0.9.0 release (breaking changes)

---

### Sprint 216: Regression Testing (High)
**Estimated effort:** 20 hours  
**Files changed:** New `tests/regression/` directory  
**Breaking changes:** No

**Tasks:**
- Create regression test framework
- Generate baselines (FDTD, PSTD, PINN)
- Add cross-validation test stubs
- Document baseline generation process

**Deliverables:**
- 20+ regression test baselines
- Cross-validation framework
- Baseline documentation

---

### Sprint 217: Tissue Database (Medium)
**Estimated effort:** 16 hours  
**Files changed:** New `domain/medium/tissue_database.rs`  
**Breaking changes:** No

**Tasks:**
- Implement `TissueDatabase` with IT'IS Foundation data
- Add JSON serialization
- Create `Medium::from_tissue()` builder
- Document data sources

**Deliverables:**
- Tissue database with 100+ entries
- Builder pattern for medium creation
- Citation documentation

---

### Sprint 218: GPU Backend Refinement (Medium)
**Estimated effort:** 12 hours  
**Files changed:** ~5 (GPU configuration)  
**Breaking changes:** No

**Tasks:**
- Expose wgpu backend selection
- Benchmark Vulkan/Metal/DX12
- Optimize shader compilation
- Add GPU capability detection

**Deliverables:**
- Backend selection in configuration
- Benchmark report
- Improved GPU performance

---

## Success Metrics

### Code Quality
- **Beamforming duplication:** 120+ files → 0 files in domain layer
- **Layering violations:** Unknown → 0 (CI enforced)
- **Test coverage:** Current → 90%+ for beamforming

### Architecture
- **Coupling:** Tight (direct imports) → Loose (abstraction)
- **Cohesion:** Split (duplication) → Unified (SSOT)
- **Testability:** Difficult (embedded) → Easy (mockable)

### Performance
- **Test suite:** <30s (maintained)
- **GPU backends:** 1 → 3+ supported
- **Regression confidence:** None → High (baselines)

---

## Reading Guide by Role

### **Software Architect / Technical Lead**
1. Read: [Full Research Report](architectural_research_ultrasound_libraries.md)
2. Review: [Architecture Diagrams](architecture_diagrams.md)
3. Approve: [Recommendations Summary](architectural_recommendations_summary.md)
4. Oversee: Sprint execution (213-218)

**Focus:** Ensure architectural decisions align with industry best practices and long-term maintainability.

---

### **Team Lead / Product Manager**
1. Read: [Recommendations Summary](architectural_recommendations_summary.md)
2. Review: [Architecture Diagrams](architecture_diagrams.md) (visual overview)
3. Plan: Sprint 213-218 capacity allocation
4. Track: Success metrics

**Focus:** Understand business impact (reduced technical debt, improved maintainability) and resource requirements.

---

### **Developer (Implementing Changes)**
1. Read: [Migration Guide](migration_guide_beamforming_consolidation.md) (hands-on)
2. Reference: [Architecture Diagrams](architecture_diagrams.md) (visual patterns)
3. Consult: [Full Research Report](architectural_research_ultrasound_libraries.md) (specific library patterns)
4. Execute: Sprint tasks (213-218)

**Focus:** Code migration, testing, verification.

---

### **QA / Test Engineer**
1. Read: [Recommendations Summary](architectural_recommendations_summary.md) (what's changing)
2. Review: [Migration Guide](migration_guide_beamforming_consolidation.md) (verification checklist)
3. Implement: Regression testing framework (Sprint 216)
4. Validate: All migration phases

**Focus:** Ensure no functional regressions, verify deprecation warnings, validate migration.

---

### **New Team Member**
1. Start: [Architecture Diagrams](architecture_diagrams.md) (visual overview)
2. Read: [Recommendations Summary](architectural_recommendations_summary.md) (current state + goals)
3. Deep dive: [Full Research Report](architectural_research_ultrasound_libraries.md) (if working on architecture)
4. Implement: [Migration Guide](migration_guide_beamforming_consolidation.md) (if coding)

**Focus:** Understand current architecture, why it's changing, and how to write code that follows new patterns.

---

## Key Concepts Glossary

### Beamforming
**Definition:** Signal processing technique to create directional acoustic images from array sensor data  
**Current issue:** Code duplicated in domain (sensor) and analysis layers  
**Solution:** Consolidate to analysis layer (post-processing only)  
**Industry example:** k-Wave-Python's `reconstruction/beamform`

### Plugin Architecture
**Definition:** Extensible system where solvers are plugins, not hardcoded  
**Current state:** `PluginExecutor` exists but underutilized  
**Solution:** Clinical workflows must use `PluginExecutor`, not direct solver imports  
**Industry example:** BabelBrain's backend delegation

### Step-Based Workflow
**Definition:** Clinical workflows as orchestrated steps (domain prep → simulation → analysis)  
**Current issue:** Workflows contain physics implementation  
**Solution:** Workflows orchestrate, backend libraries implement  
**Industry example:** BabelBrain (Step 1: domain, Step 2: acoustic, Step 3: thermal)

### Tissue Database
**Definition:** Standardized repository of tissue acoustic/thermal properties  
**Current issue:** Properties scattered as constants  
**Solution:** Centralized database with citations  
**Industry example:** OptimUS uses IT'IS Foundation Database V4.1

### Regression Testing
**Definition:** Tests that verify numerical results match historical baselines  
**Current issue:** Only benchmarks (performance), no numerical baselines  
**Solution:** Store baseline pressure fields, compare new results  
**Industry example:** BabelBrain's numerical regression tests

---

## Quick Decision Matrix

### Should beamforming code go in domain or analysis layer?
**Answer:** Analysis layer only  
**Reasoning:** Beamforming is post-processing (signal reconstruction), not domain logic (sensor geometry)  
**Industry consensus:** k-Wave-Python, Fullwave, Sound-Speed all place beamforming separate from solvers

### Should clinical workflows import specific solvers?
**Answer:** No - use `PluginExecutor` abstraction  
**Reasoning:** Enables solver swapping, improves testability, reduces coupling  
**Industry consensus:** BabelBrain delegates to backend solvers via abstraction

### Should we implement our own registration/meshing tools?
**Answer:** No - use external libraries (trimesh, Elastix equivalents)  
**Reasoning:** These are solved problems, focus on core competency (acoustic simulation)  
**Industry consensus:** BabelBrain uses trimesh, Elastix; OptimUS uses bempp

### Should we support multiple GPU backends?
**Answer:** Yes - wgpu already does this  
**Reasoning:** Cross-platform compatibility (Vulkan/Metal/DX12)  
**Industry consensus:** BabelBrain supports Metal/OpenCL/CUDA

### Should ML replace physics in beamforming?
**Answer:** No - ML enhances physics, doesn't replace it  
**Reasoning:** Physics provides correctness guarantees, ML optimizes performance  
**Industry consensus:** Sound-Speed (coherence optimization), DBUA (hybrid approach)

---

## References by Topic

### Beamforming Architecture
- **k-Wave-Python:** `reconstruction/beamform` module (post-processing)
- **Fullwave:** `examples/` only (not in solver)
- **Sound-Speed:** Dedicated SLSC module (coherence-based)
- **DBUA:** Differentiable DAS (learned delays)

### Clinical Workflow Patterns
- **BabelBrain:** Step 1 (domain) → Step 2 (acoustic) → Step 3 (thermal)
- **Kranion:** Plugin architecture (ACPC, Console, Tractography)
- **OptimUS:** `applications/` layer over physics

### GPU Acceleration
- **j-Wave:** JAX `@jit` auto-diff + XLA
- **k-Wave:** Pre-compiled C++/CUDA binaries
- **Fullwave:** Native CUDA/C backend
- **BabelBrain:** Multi-backend (Metal/OpenCL/CUDA)

### Physics Models
- **mSOUND:** Generalized Westervelt equation
- **Fullwave:** Heterogeneous attenuation (α₀ and γ vary)
- **HITU:** Axisymmetric FDTD + bioheat transfer
- **BabelBrain:** Viscoelastic FDTD

### Testing Strategies
- **BabelBrain:** Unit + regression + cross-validation + experimental
- **j-Wave:** Unit tests + coverage tracking
- **Sound-Speed:** Benchmark datasets (CUBDL)

---

## Contact and Support

### Questions about Research
- **GitHub Issues:** Tag with `architecture` label
- **Documentation:** This index + linked documents

### Implementation Questions
- **Migration guide:** See [migration_guide_beamforming_consolidation.md](migration_guide_beamforming_consolidation.md)
- **Code examples:** Check `examples/` directory (updated in Sprint 214)

### Sprint Planning
- **Sprints 213-218:** See [architectural_recommendations_summary.md](architectural_recommendations_summary.md#action-plan-by-sprint)
- **Success metrics:** See [Success Metrics](#success-metrics) above

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-23 | Architecture Research | Initial research and recommendations |

---

## Next Steps

1. **Review Period (1 week):** Team reviews all documents, provides feedback
2. **Approval (Architecture Board):** Approve recommendations and sprint plan
3. **Sprint 213 Kickoff:** Begin beamforming consolidation (deprecation phase)
4. **Continuous Updates:** Update documents as implementation progresses

---

**Research Status:** Complete  
**Implementation Status:** Pending approval  
**Target Start Date:** Sprint 213 (TBD)  
**Estimated Completion:** Sprint 218 (TBD)
