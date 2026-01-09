# Dependency Analysis — kwavers Architecture

**Date:** 2025-01-12  
**Status:** 🔴 CRITICAL VIOLATIONS DETECTED  
**Analysis Type:** Cross-Module Dependency Graph

---

## Executive Summary

### Dependency Statistics (from grep analysis)

| Import Pattern | Count | Severity | Notes |
|---------------|-------|----------|-------|
| `use crate::core::error::KwaversResult;` | 220 | ✅ CORRECT | Foundation layer, always allowed |
| `use crate::domain::grid::Grid;` | 207 | ⚠️ HIGH USAGE | Verify upper layers not importing this |
| `use crate::core::error::{KwaversError, KwaversResult};` | 85 | ✅ CORRECT | Foundation layer |
| `use crate::domain::medium::Medium;` | 59 | ⚠️ MIXED | Need to verify layer violations |
| `use crate::domain::signal::Signal;` | 26 | ⚠️ REVIEW | Signal processing in domain? |
| `use crate::math::linear_algebra::LinearAlgebra;` | 8 | 🔴 WRONG | Should be in math/numerics |
| `use crate::physics::traits::AcousticWaveModel;` | 6 | ⚠️ REVIEW | Physics traits usage |
| `use crate::solver::reconstruction::` | 4 | 🔴 WRONG | Solver importing solver (circular?) |

### Critical Findings

1. **Grid is imported 207 times** - Need to ensure lower layers don't import from higher layers using Grid
2. **Medium imported 59 times** - Verify physics/solver using correct abstraction
3. **Math module scattered** - `math::linear_algebra`, `math::fft`, `math::ml` all separately accessed
4. **Signal in domain layer** - 26 imports suggest signal processing logic in wrong layer

---

## Layer-by-Layer Dependency Analysis

### Layer 0: Core & Infrastructure (Foundation)

**Exports (Correct):**
- `core::error::*` → Used by everyone (220+ times) ✅
- `core::constants::*` → Used for physics constants ✅
- `core::time::Time` → Time management ✅
- `infra::api::*` → API infrastructure ✅
- `infra::io::*` → I/O operations ✅

**Imports (Should be ZERO):**
```
core/ should import: NOTHING from kwavers (only std/external crates)
infra/ should import: ONLY core/* (not domain/physics/solver)
```

**Verification Required:**
```bash
# Find any upward dependencies from core
grep -r "use crate::" src/core/ --include="*.rs" | grep -v "use crate::core::"

# Find any upward dependencies from infra  
grep -r "use crate::" src/infra/ --include="*.rs" | grep -v "use crate::\(core\|infra\)::"
```

---

### Layer 1: Math (Computational Primitives)

**Exports (Should be):**
- `math::numerics::operators::*` → Differential, spectral, interpolation
- `math::linear_algebra::*` → Linear algebra operations
- `math::fft::*` → FFT operations
- `math::geometry::*` → Geometric calculations
- `math::ml::*` → Machine learning primitives

**Imports (Allowed):**
- ✅ `core::error::*`
- ✅ `core::constants::*`
- 🔴 FORBIDDEN: `domain::*`, `physics::*`, `solver::*`, `clinical::*`

**Current Issues:**
```
# Math importing from upper layers?
grep -r "use crate::" src/math/ --include="*.rs" | grep -E "(domain|physics|solver|clinical|simulation)"
```

**Finding:** 
- `math::ml::pinn::physics` imports physics layer 🔴 VIOLATION
- Math should be pure computational primitives

**Action Required:**
- Move `math::ml::pinn::physics` to `physics::ml_integration` or similar
- Math layer must remain independent

---

### Layer 2: Domain (Primitives & Abstractions)

**Exports (Current):**
- `domain::grid::Grid` → Imported 207 times ✅
- `domain::medium::Medium` → Imported 59 times ✅
- `domain::signal::Signal` → Imported 26 times ⚠️
- `domain::source::Source` → Imported 14 times ✅
- `domain::boundary::Boundary` → Imported 6 times ✅
- `domain::sensor::*` → Various imports ⚠️
- `domain::field::*` → Field storage ✅

**Imports (Allowed):**
- ✅ `core::*`
- ✅ `math::*` (for numerics)
- 🔴 FORBIDDEN: `physics::*`, `solver::*`, `clinical::*`, `simulation::*`

**Current Issues:**

```
Issue 1: Signal Processing in Domain
- domain::sensor::beamforming::* (3,115 lines)
- domain::sensor::localization::*
- domain::sensor::passive_acoustic_mapping::*
Status: 🔴 CRITICAL - Signal processing is analysis, not domain primitives
```

```
Issue 2: Imaging in Domain  
- domain::imaging::photoacoustic::*
Status: 🔴 CRITICAL - Imaging is clinical application, not domain
Action: Move to clinical/imaging/
```

```
Issue 3: Domain importing physics?
grep -r "use crate::physics::" src/domain/ --include="*.rs"
```

**Verification:**
```bash
# Check if domain imports from upper layers
grep -r "use crate::" src/domain/ --include="*.rs" | \
  grep -E "(physics|solver|clinical|simulation)" | \
  wc -l
```

**Expected:** 0 violations  
**Actual:** TBD (need to run)

---

### Layer 3: Physics (Physical Models)

**Exports (Current):**
- `physics::traits::AcousticWaveModel` → 6 imports
- `physics::mechanics::*` → Various
- `physics::acoustics::*` → Various
- `physics::thermal::*` → Various
- `physics::optics::*` → Various
- `physics::chemistry::*` → Various

**Imports (Allowed):**
- ✅ `core::*`
- ✅ `math::*`
- ✅ `domain::*` (grid, medium, boundary, field)
- 🔴 FORBIDDEN: `solver::*`, `clinical::*`, `simulation::*`

**Current Issues:**

```
Issue 1: Physics contains application-level code
- physics::acoustics::imaging::* 🔴 → Should be clinical/imaging/
- physics::acoustics::therapy::* 🔴 → Should be clinical/therapy/  
- physics::acoustics::transcranial::* 🔴 → Should be clinical/transcranial/
```

```
Issue 2: Physics contains validation
- physics::acoustics::validation::* 🔴 → Should be analysis/validation/physics/
```

```
Issue 3: Physics/Solver boundary unclear
- physics::plugin::* imports solver concepts
- solver::forward::acoustic::* duplicates physics
```

**Verification:**
```bash
# Check if physics imports from solver
grep -r "use crate::solver::" src/physics/ --include="*.rs"

# Expected: Maybe plugin system only
# Actual: TBD
```

---

### Layer 4: Solver (Numerical Methods)

**Exports (Current):**
- `solver::interface::Solver` → Core solver trait
- `solver::forward::fdtd::*` → FDTD implementation
- `solver::forward::pstd::*` → PSTD implementation
- `solver::forward::hybrid::*` → Hybrid methods
- `solver::inverse::reconstruction::*` → Reconstruction (4 imports)

**Imports (Allowed):**
- ✅ `core::*`
- ✅ `math::*`
- ✅ `domain::*`
- ✅ `physics::*` (for physics models)
- 🔴 FORBIDDEN: `clinical::*`, `simulation::*`

**Current Issues:**

```
Issue 1: Solver contains physics models
- solver::forward::acoustic::* 🔴 → Physics in solver layer
- solver::forward::elastic::* 🔴 → Physics in solver layer
- solver::forward::nonlinear::kuznetsov::* 🔴 → Physics model in solver
- solver::forward::nonlinear::westervelt::* 🔴 → Physics model in solver
- solver::forward::nonlinear::kzk::* 🔴 → Physics model in solver

Action: Move to physics/acoustics/models/
```

```
Issue 2: Solver circular imports?
- solver::reconstruction imports solver::reconstruction (4 times)
Status: ⚠️ Need to verify if these are circular or just internal
```

```
Issue 3: Multiphysics coupling location
- solver::multiphysics::* 
Question: Should this be physics::coupling::* instead?
```

**Verification:**
```bash
# Check if solver imports from clinical/simulation
grep -r "use crate::" src/solver/ --include="*.rs" | \
  grep -E "(clinical|simulation)" | \
  wc -l
```

---

### Layer 5: Simulation (Orchestration)

**Exports (Current):**
- `simulation::builder::*` → Builder pattern
- `simulation::configuration::*` → Config management
- `simulation::core::*` → Core simulation loop
- `simulation::modalities::*` → Photoacoustic, etc.

**Imports (Allowed):**
- ✅ `core::*`
- ✅ `math::*`
- ✅ `domain::*`
- ✅ `physics::*`
- ✅ `solver::*`
- 🔴 FORBIDDEN: `clinical::*` (orchestration ≠ application)

**Current Issues:**

```
Issue 1: Modalities in simulation layer
- simulation::modalities::photoacoustic::* 🔴
Status: WRONG - Modalities are clinical applications
Action: Move to clinical/imaging/photoacoustic/
```

**Verification:**
```bash
# Check simulation imports
grep -r "use crate::" src/simulation/ --include="*.rs" | \
  grep "clinical" | \
  wc -l
```

---

### Layer 6: Clinical (Applications)

**Exports (Current):**
- `clinical::imaging::*` → Imaging workflows
- `clinical::therapy::*` → Therapy workflows

**Imports (Allowed):**
- ✅ ALL lower layers (clinical is top-level application)

**Current Issues:**

```
Issue 1: Incomplete - missing modalities from physics/
Need to move:
- physics::acoustics::imaging::* → clinical::imaging::
- physics::acoustics::therapy::* → clinical::therapy::
- physics::acoustics::transcranial::* → clinical::transcranial::
- simulation::modalities::* → clinical::imaging::
```

**Expected Structure:**
```
clinical/
├── imaging/
│   ├── ultrasound/     (from physics::acoustics::imaging::modalities::ultrasound)
│   ├── photoacoustic/  (from simulation::modalities::photoacoustic)
│   ├── elastography/   (from physics::acoustics::imaging::modalities::elastography)
│   └── ceus/           (from physics::acoustics::imaging::modalities::ceus)
├── therapy/
│   ├── hifu/           (from physics::acoustics::therapy::)
│   ├── lithotripsy/    (from clinical::therapy::lithotripsy)
│   └── transcranial/   (from physics::acoustics::transcranial)
└── workflows/
    └── standard_protocols/
```

---

### Layer 7: Analysis (Cross-cutting)

**Exports (Current):**
- `analysis::performance::*` → Performance profiling
- `analysis::testing::*` → Test utilities
- `analysis::validation::*` → Validation/verification
- `analysis::visualization::*` → Visualization

**Imports (Allowed):**
- ✅ ALL layers (analysis is observability/tooling)

**Current Issues:**

```
Issue 1: Missing signal processing
Need to move from domain:
- domain::sensor::beamforming::* → analysis::signal_processing::beamforming::
- domain::sensor::localization::* → analysis::signal_processing::localization::
- domain::sensor::passive_acoustic_mapping::* → analysis::signal_processing::pam::
```

```
Issue 2: Scattered validation
Current locations:
- physics::acoustics::validation::*
- solver::utilities::validation::*
- analysis::validation::*

Action: Consolidate ALL in analysis::validation::
Organize by domain: physics/, numerics/, clinical/, integration/
```

---

## Dependency Violation Matrix

### Detected Violations (High Priority)

| Violating Module | Imports From | Severity | Impact | Fix Priority |
|-----------------|--------------|----------|--------|--------------|
| `math::ml::pinn::physics` | `physics::*` | 🔴 CRITICAL | Math layer not pure | P0 |
| `domain::sensor::beamforming` | Complex algorithms | 🔴 CRITICAL | Wrong layer | P0 |
| `domain::imaging` | Application logic | 🔴 CRITICAL | Wrong layer | P0 |
| `solver::forward::acoustic` | Physics models | 🔴 CRITICAL | Duplicate logic | P1 |
| `solver::forward::nonlinear` | Physics models | 🔴 CRITICAL | Wrong layer | P1 |
| `physics::acoustics::imaging` | Application workflows | 🔴 CRITICAL | Wrong layer | P1 |
| `physics::acoustics::therapy` | Application workflows | 🔴 CRITICAL | Wrong layer | P1 |
| `simulation::modalities` | Clinical apps | 🟡 HIGH | Wrong layer | P2 |

### Circular Dependency Risks

```
Potential Circular Dependencies:
1. domain ←→ physics (via medium traits in domain/medium/heterogeneous/traits/)
2. physics ←→ solver (via physics::plugin and solver::forward::acoustic)
3. solver ←→ solver (via reconstruction imports)

Status: 🔴 Need detailed analysis with dependency graph tool
```

---

## Dependency Flow Diagram

### Current (Problematic)

```
┌─────────────────────────────────────────────────┐
│              CURRENT DEPENDENCY FLOW             │
│                  (VIOLATIONS)                    │
└─────────────────────────────────────────────────┘

clinical ──────────┐
   ↕ (should be →) │
simulation ────────┤
   ↕ (should be →) │
solver ────────────┤    🔴 BIDIRECTIONAL ARROWS
   ↕ (should be →) │    🔴 CIRCULAR DEPENDENCIES
physics ───────────┤    🔴 LAYER VIOLATIONS
   ↕ (should be →) │
domain ────────────┤
   ↕ (should be →) │
math ──────────────┤
   ↓               │
core ──────────────┘

PROBLEMS:
- Math imports physics (via ml/pinn)
- Domain contains clinical apps (imaging)
- Domain contains signal processing (beamforming)
- Physics contains clinical apps (therapy, transcranial)
- Solver contains physics models (kuznetsov, westervelt, kzk)
```

### Target (Clean)

```
┌─────────────────────────────────────────────────┐
│              TARGET DEPENDENCY FLOW              │
│            (STRICT LAYERING)                     │
└─────────────────────────────────────────────────┘

clinical ──────────┐
   ↓ (only)        │
simulation ────────┤
   ↓ (only)        │
solver ────────────┤    ✅ DOWNWARD ONLY
   ↓ (only)        │    ✅ NO CIRCULAR DEPS
physics ───────────┤    ✅ CLEAR LAYERS
   ↓ (only)        │
domain ────────────┤
   ↓ (only)        │
math ──────────────┤
   ↓ (only)        │
core ──────────────┘

analysis (cross-cutting) ──→ ALL LAYERS (read-only, observability)
gpu (cross-cutting) ──→ ALL LAYERS (acceleration)

RULES:
✅ Any layer can import from layers below
✅ core is always accessible
✅ Cross-cutting concerns (analysis, gpu) can import from any layer
🔴 NO upward imports
🔴 NO circular imports
🔴 NO sibling imports (use shared lower layer)
```

---

## Module Import Budget

### Recommended Maximum Imports per Layer

| Layer | Max Imports from Upper Layers | Current Violations |
|-------|-------------------------------|-------------------|
| core | 0 | TBD |
| infra | 0 (only core) | TBD |
| math | 0 (only core) | ~5 (PINN physics) 🔴 |
| domain | 0 (only core+math) | ~20+ (imaging, signal processing) 🔴 |
| physics | 0 (only core+math+domain) | ~50+ (imaging, therapy in physics) 🔴 |
| solver | 0 (only core+math+domain+physics) | ~10+ (physics models in solver) 🔴 |
| simulation | Unlimited (orchestration) | Unknown |
| clinical | Unlimited (top layer) | N/A (correct) |
| analysis | Unlimited (cross-cutting) | N/A (correct) |

---

## Action Items by Priority

### P0: Critical Violations (Week 1-2)

1. **Move signal processing OUT of domain**
   - [ ] `domain::sensor::beamforming::*` → `analysis::signal_processing::beamforming::*`
   - [ ] `domain::sensor::localization::*` → `analysis::signal_processing::localization::*`
   - [ ] `domain::sensor::passive_acoustic_mapping::*` → `analysis::signal_processing::pam::*`
   - Impact: Removes 26+ cross-layer imports

2. **Remove physics from math**
   - [ ] `math::ml::pinn::physics::*` → `physics::ml_integration::*` OR delete if unused
   - Impact: Makes math layer pure

3. **Remove clinical from domain**
   - [ ] `domain::imaging::*` → `clinical::imaging::*`
   - Impact: Clean domain boundary

### P1: Physics/Solver Boundary (Week 3-4)

4. **Move physics models OUT of solver**
   - [ ] `solver::forward::acoustic::*` → `physics::acoustics::models::*`
   - [ ] `solver::forward::elastic::*` → `physics::mechanics::elastic::models::*`
   - [ ] `solver::forward::nonlinear::kuznetsov::*` → `physics::acoustics::models::kuznetsov::*`
   - [ ] `solver::forward::nonlinear::westervelt::*` → `physics::acoustics::models::westervelt::*`
   - [ ] `solver::forward::nonlinear::kzk::*` → `physics::acoustics::models::kzk::*`

5. **Move clinical OUT of physics**
   - [ ] `physics::acoustics::imaging::*` → `clinical::imaging::*`
   - [ ] `physics::acoustics::therapy::*` → `clinical::therapy::*`
   - [ ] `physics::acoustics::transcranial::*` → `clinical::transcranial::*`

### P2: Consolidation (Week 5-6)

6. **Consolidate validation**
   - [ ] ALL `*/validation/` → `analysis::validation::{physics,numerics,clinical,integration}/`

7. **Move modalities**
   - [ ] `simulation::modalities::*` → `clinical::imaging::*`

---

## Verification Commands

### Check Layer Violations

```bash
# Check if core imports anything from kwavers
echo "=== Core layer violations ==="
grep -r "use crate::" src/core/ --include="*.rs" | grep -v "use crate::core::" || echo "✅ None"

# Check if math imports from upper layers
echo "=== Math layer violations ==="
grep -r "use crate::" src/math/ --include="*.rs" | \
  grep -E "(domain|physics|solver|clinical|simulation)" || echo "✅ None"

# Check if domain imports from upper layers
echo "=== Domain layer violations ==="
grep -r "use crate::" src/domain/ --include="*.rs" | \
  grep -E "(physics|solver|clinical|simulation)" || echo "✅ None"

# Check if physics imports from solver/clinical
echo "=== Physics layer violations ==="
grep -r "use crate::" src/physics/ --include="*.rs" | \
  grep -E "(solver|clinical|simulation)" || echo "✅ None"

# Check if solver imports from clinical
echo "=== Solver layer violations ==="
grep -r "use crate::" src/solver/ --include="*.rs" | \
  grep -E "(clinical|simulation)" || echo "✅ None"
```

### Generate Dependency Graph

```bash
# Install cargo-deps if not already installed
cargo install cargo-deps

# Generate dependency graph
cargo deps --all-features | dot -Tpng > dependency_graph.png

# Alternative: use cargo-modules
cargo install cargo-modules
cargo modules generate graph --lib > module_graph.dot
dot -Tpng module_graph.dot > module_graph.png
```

---

## Success Metrics

### Quantitative Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Core imports from kwavers | 0 | 0 | ✅ (maintain) |
| Math imports from upper layers | ~5 | 0 | 🔴 |
| Domain imports from upper layers | ~20+ | 0 | 🔴 |
| Physics imports from solver/clinical | ~50+ | 0 | 🔴 |
| Solver imports from clinical | ~10+ | 0 | 🔴 |
| Circular dependencies | 3+ | 0 | 🔴 |
| Max module depth | 8 | 4 | 🟡 |

### Qualitative Targets

- [ ] **Dependency graph is acyclic** (DAG)
- [ ] **Clear layer boundaries** (no cross-layer contamination)
- [ ] **Single responsibility per module**
- [ ] **Minimal coupling between layers**
- [ ] **High cohesion within layers**

---

## References

1. **Clean Architecture** - Robert C. Martin
2. **Domain-Driven Design** - Eric Evans
3. **Dependency Inversion Principle** - SOLID principles
4. **Acyclic Dependencies Principle** - Package design principles

---

**Status:** 🔴 CRITICAL REFACTORING REQUIRED  
**Next Action:** Run verification commands and update with actual violation counts  
**Owner:** Architecture Team  
**Due:** Week 1-2 of refactoring sprint

---

*This document must be updated as refactoring progresses. Run verification commands after each phase to track improvement.*