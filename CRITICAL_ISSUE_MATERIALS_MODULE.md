# 🚨 CRITICAL ARCHITECTURAL ISSUE: Materials Module

**Status**: IDENTIFIED  
**Priority**: P1 (Must Fix)  
**Date Found**: 2026-01-29  
**Effort**: 6-8 hours  
**Impact**: Architectural Correctness + SSOT Violation

---

## The Problem

The `physics/materials/` module contains **material property specifications** that should be in the **domain layer**, not the physics layer.

### What's Wrong

```
physics/materials/                    ❌ WRONG LAYER
├── MaterialProperties struct
│   - speed_speed
│   - density
│   - impedance
│   - absorption_coefficient
│   - etc...
├── tissue.rs (property lookup tables)
├── fluids.rs (property lookup tables)
└── implants.rs (property lookup tables)
```

### Why It's Wrong

1. **Layer Violation**: Properties are SPECIFICATIONS (domain concern), not physics EQUATIONS
2. **SSOT Violation**: Same data already exists in `domain/medium/properties/`
3. **Separation of Concerns**: Physics layer should contain "HOW" (equations), not "WHAT" (specifications)

### Architectural Principles Violated

```
Correct Hierarchy:
Domain Layer (Layer 2)     - Defines WHAT (specifications)
    ↓ depends on
Physics Layer (Layer 3)    - Defines HOW (equations)

Current State (WRONG):
Physics Layer contains WHAT (specifications)
    ↓ violates upward dependency
Domain Layer also has similar specifications
```

---

## The Solution

### Move Everything to Domain Layer

```
domain/medium/properties/                 ✅ CORRECT LAYER
├── material.rs (unified MaterialProperties struct)
├── tissue_catalog.rs (tissue properties)
├── fluid_catalog.rs (fluid properties)
└── implant_catalog.rs (implant properties)
```

### Keep Physics Calculations in Physics Layer

Physics layer should have functions that **USE** the properties, not DEFINE them:

```rust
// physics/acoustics/boundary.rs
pub fn reflection_coefficient(
    material1: &MaterialProperties,  // INPUT from domain
    material2: &MaterialProperties,  // INPUT from domain
) -> f64 {
    // Physics calculation using properties
    let z1 = material1.impedance;
    let z2 = material2.impedance;
    ((z2 - z1) / (z2 + z1)).abs()
}
```

---

## Files Affected

### To Move (from physics → domain)
- ❌ `physics/materials/mod.rs`
- ❌ `physics/materials/tissue.rs`
- ❌ `physics/materials/fluids.rs`
- ❌ `physics/materials/implants.rs`

### To Update Imports
- `physics/acoustics/mechanics/cavitation/mod.rs`
- `physics/acoustics/mechanics/cavitation/damage.rs`

### To Merge With
- ✅ `domain/medium/properties/mod.rs` (already exists!)
- ✅ `domain/medium/properties/acoustic.rs` (consolidate with)
- ✅ `domain/medium/properties/thermal.rs` (consolidate with)
- ✅ `domain/medium/properties/optical.rs` (consolidate with)

---

## Key Insight

**Domain layer ALREADY HAS property definitions!** The `physics/materials/` module is a **duplicate that never should have existed**.

```
domain/medium/properties/ exists with:
  - AcousticProperties
  - ThermalProperties
  - OpticalProperties
  - ElasticProperties
  - etc...

physics/materials/ is a DUPLICATE:
  - MaterialProperties (similar to above)
  - tissue, fluids, implants (lookup tables)
```

This is why it's classified as both a **layer violation** AND an **SSOT violation**.

---

## Implementation Steps

### Step 1: Consolidate Property Definitions (2 hours)
1. Merge `physics/materials/MaterialProperties` into `domain/medium/properties/material.rs`
2. Move tissue, fluid, implant catalogs to `domain/medium/properties/`
3. Update validation logic to domain layer

### Step 2: Update Physics Imports (1-2 hours)
1. Change `use crate::physics::materials::MaterialProperties`
2. To: `use crate::domain::medium::MaterialProperties`
3. Update 2 cavitation files

### Step 3: Delete Physics Module (15 min)
1. Remove `physics/materials/` directory
2. Update `physics/mod.rs` to remove module declaration

### Step 4: Testing (2-3 hours)
1. `cargo build --lib` (verify compiles)
2. `cargo test --lib` (verify tests pass)
3. Grep for any remaining `physics::materials` references

---

## Why This Matters

### Architectural Purity
Without fixing this:
- Physics layer violates dependency rules
- SSOT principle is broken
- New developers will be confused about where to put property definitions

### Future Problems
If not fixed:
- Adding new materials becomes confusing (which layer?)
- Updating properties might miss the other definition
- Documentation will be contradictory

### This Is A Pattern Issue
Fixing this teaches everyone:
- Properties (WHAT things have) → Domain layer
- Equations (HOW things work) → Physics layer
- Physics uses domain, not vice versa

---

## Current Status

- ✅ Issue identified and documented
- ✅ Root cause analyzed
- ✅ Solution designed
- ✅ Implementation plan created
- ✅ Impact analysis completed
- ⏳ Ready for implementation next sprint

---

## Reference Documents

- **MATERIALS_MODULE_REFACTORING_PLAN.md** - Detailed step-by-step plan
- **ARCHITECTURE.md** - Updated with this pattern example
- **ARCHITECTURE_COMPLIANCE_REPORT.md** - Detailed analysis

---

## Quick Reference

| Aspect | Status |
|--------|--------|
| Issue Severity | 🚨 CRITICAL |
| Effort | 6-8 hours |
| Priority | P1 (Next Sprint) |
| Impact | High (architectural correctness) |
| Risk | Low (contained change) |
| Documentation | Complete |
| Implementation Ready | Yes |

---

**Bottom Line**: This is the most important architectural fix remaining. It's straightforward to implement and will restore proper layer separation and SSOT compliance.
