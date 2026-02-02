# Materials Module Refactoring Plan
**Priority**: HIGH (Architectural Correctness)  
**Severity**: CRITICAL SSOT Violation  
**Date**: 2026-01-29  
**Effort**: 6-8 hours

---

## 🚨 Issue Identified

The `physics/materials/` module contains **material property specifications** that should reside in the **domain layer**, not the physics layer.

### Current Problematic Structure
```
physics/materials/               ❌ WRONG LAYER
├── tissue.rs       - Tissue properties
├── fluids.rs       - Fluid properties  
├── implants.rs     - Implant properties
└── MaterialProperties - Core property struct
```

### Correct Target Structure
```
domain/medium/properties/        ✅ CORRECT LAYER
├── acoustic.rs
├── thermal.rs
├── optical.rs
└── composite.rs
```

---

## 📚 Background: Layer Definitions

### Domain Layer (Layer 2) - What Things Have
**Responsibility**: Define WHAT properties exist and their specifications

**Examples**:
- ✅ Material property definitions (speed of sound, density, etc.)
- ✅ Property categories (acoustic, thermal, optical)
- ✅ Material type hierarchies (tissue, fluid, implant)
- ✅ Property constraints and validation rules

**SHOULD contain**: `physics/materials/`

### Physics Layer (Layer 3) - How Things Behave
**Responsibility**: Define HOW properties relate through physical laws

**Examples**:
- ✅ Physical equations (wave equation, heat equation)
- ✅ Constitutive relations (stress-strain, etc.)
- ✅ Material behavior models (nonlinear, dispersive)
- ❌ Material property definitions
- ❌ Property lookup tables

**Should NOT contain**: `physics/materials/`

---

## 🔍 Architecture Violation Analysis

### Violation Type 1: Layer Inversion
```
Domain Layer
    ↑ (WRONG! upward dependency)
    |
Physics Layer contains material properties
```

**Why This Is Wrong**:
- Physics layer depends on domain layer specifications
- Having material properties in physics violates this dependency
- Creates confusion about where properties come from

### Violation Type 2: SSOT (Single Source of Truth) Violation
```
domain/medium/properties/        Properties used by domain
    ↓
physics/materials/              Duplicate/different properties
    ↓
Both layers have property definitions (SSOT BROKEN!)
```

### Violation Type 3: Separation of Concerns
```
physics/materials/               Contains:
├── MaterialProperties struct    - Property container (DOMAIN concern)
├── validation()                 - Property validation (DOMAIN concern)
├── reflection_coefficient()     - Physics calculation (PHYSICS concern)
└── impedance_ratio()            - Physics calculation (PHYSICS concern)
```

---

## ✅ Solution: Consolidate to Domain Layer

### Step 1: Audit Current State (COMPLETE ✅)
- [x] Identify all files in `physics/materials/`
- [x] Check usage patterns
- [x] Verify duplication with `domain/medium/`

**Finding**: `physics/materials/` is self-contained with minimal external use

### Step 2: Consolidate Properties to Domain (TO DO)

**Action**: Move property definitions to `domain/medium/properties/`

```rust
// physics/materials/mod.rs content should migrate to:
domain/medium/properties/
├── material.rs              // New: unified MaterialProperties
├── tissue_catalog.rs        // New: tissue property lookup
├── fluid_catalog.rs         // New: fluid property lookup  
└── implant_catalog.rs       // New: implant property lookup
```

### Step 3: Keep Physics Calculations in Physics Layer

**Important**: Physics calculations (reflection, transmission, attenuation) that use properties should:
1. Accept `&MaterialProperties` as input
2. Remain in physics layer as utility functions
3. NOT duplicate validation logic

```rust
// GOOD: Physics calculations stay in physics layer
pub mod physics::acoustics::boundary {
    pub fn reflection_coefficient(
        material1: &MaterialProperties,  // From domain
        material2: &MaterialProperties,  // From domain
    ) -> f64 {
        let z1 = material1.impedance;
        let z2 = material2.impedance;
        ((z2 - z1) / (z2 + z1)).abs()
    }
}
```

### Step 4: Update Imports Throughout Codebase

**Files needing updates**:
- `physics/acoustics/mechanics/cavitation/mod.rs` - uses MaterialProperties
- `physics/acoustics/mechanics/cavitation/damage.rs` - uses MaterialProperties
- Any other physics files importing from `physics::materials`

### Step 5: Create Domain Re-export for Physics

**Ensure physics layer can access properties**:
```rust
// physics/mod.rs
use crate::domain::medium::MaterialProperties;
pub use crate::domain::medium::MaterialProperties;
```

---

## 📋 Detailed Refactoring Steps

### Phase 1: Prepare Domain Layer (1-2 hours)

1. **Create unified property struct in domain**
   ```rust
   // domain/medium/properties/material.rs (NEW)
   pub struct MaterialProperties {
       // All properties from physics/materials/mod.rs
   }
   ```

2. **Move validation logic to domain**
   ```rust
   // domain/medium/properties/material.rs
   impl MaterialProperties {
       pub fn validate(&self) -> Result<()> {
           // Validation rules from physics/materials/mod.rs
       }
   }
   ```

3. **Create material catalogs in domain**
   ```rust
   // domain/medium/properties/tissue_catalog.rs
   pub const WATER: MaterialProperties = MaterialProperties { ... };
   pub const TISSUE: MaterialProperties = MaterialProperties { ... };
   pub const SKULL: MaterialProperties = MaterialProperties { ... };
   ```

### Phase 2: Update Physics Layer (2-3 hours)

1. **Remove `physics/materials/` module** (after migration)

2. **Import properties from domain**
   ```rust
   // physics/acoustics/mechanics/mod.rs
   use crate::domain::medium::MaterialProperties;
   ```

3. **Keep physics calculations in physics layer**
   ```rust
   // physics/acoustics/boundary/mod.rs (NEW)
   pub fn reflection_coefficient(
       mat1: &MaterialProperties,
       mat2: &MaterialProperties,
   ) -> f64 { ... }
   ```

4. **Update all cavitation references**
   - `physics/acoustics/mechanics/cavitation/mod.rs`
   - `physics/acoustics/mechanics/cavitation/damage.rs`

### Phase 3: Verify and Test (1-2 hours)

1. **Build verification**
   ```bash
   cargo build --lib
   ```

2. **Run test suite**
   ```bash
   cargo test --lib
   ```

3. **Check imports**
   ```bash
   grep -r "physics::materials" src/
   # Should return 0 matches
   ```

---

## 📊 Impact Analysis

### Files Affected
- `physics/materials/mod.rs` - DELETE
- `physics/materials/tissue.rs` - MOVE
- `physics/materials/fluids.rs` - MOVE
- `physics/materials/implants.rs` - MOVE
- `domain/medium/properties/mod.rs` - CONSOLIDATE
- `physics/acoustics/mechanics/cavitation/mod.rs` - UPDATE IMPORTS
- `physics/acoustics/mechanics/cavitation/damage.rs` - UPDATE IMPORTS

### Build Impact
- **Compilation**: Should still compile (just import changes)
- **Tests**: All existing tests should pass
- **Performance**: No impact (same code, different location)

### Breaking Changes
- ❌ None for external API (if physics is private)
- ⚠️ Yes for physics module users (import path change)

---

## 🎯 Acceptance Criteria

- [x] Issue identified and documented
- [ ] `physics/materials/` consolidated to `domain/medium/properties/`
- [ ] All imports updated throughout codebase
- [ ] Build succeeds with zero errors
- [ ] All tests pass
- [ ] Zero duplicate property definitions
- [ ] Documentation updated in ARCHITECTURE.md

---

## 💡 Why This Matters

### Architectural Purity
- Restores proper layer hierarchy
- Domain layer owns specifications (WHAT)
- Physics layer owns equations (HOW)

### SSOT Enforcement
- Single source of truth for material properties
- No duplicate definitions
- Clear ownership

### Maintainability
- Easier to add new materials (domain layer)
- Easier to add physics calculations (physics layer)
- Clear separation reduces bugs

### Documentation Value
- ARCHITECTURE.md can use this as example
- Shows correct layer separation pattern
- Helps other developers understand structure

---

## 🚀 Recommendation

**Priority**: HIGH  
**Urgency**: Should be done in next sprint  
**Effort**: 6-8 hours  
**Benefit**: Architectural correctness + SSOT enforcement  

This is a **must-fix** for maintaining architectural integrity. The current state violates both SSOT and layer principles.

---

## 🔄 Alternatives Considered

### Alternative 1: Keep Both (❌ NOT RECOMMENDED)
**Reason**: Violates SSOT, maintains duplication, confuses developers

### Alternative 2: Move Both to Physics (❌ NOT RECOMMENDED)
**Reason**: Violates layer hierarchy, physics layer becomes too broad

### Alternative 3: Move to Domain ✅ RECOMMENDED
**Reason**: Correct layer placement, restores hierarchy, maintains SSOT

---

## 📝 Rollback Plan

If needed during implementation:
1. Keep original `physics/materials/` in git history
2. Revert changes to that commit
3. Re-attempt with smaller scope

---

## ✅ Success Metrics

After implementation:
- [ ] Zero compilation errors
- [ ] Zero warnings
- [ ] All tests pass (100% of previous test count)
- [ ] No references to `physics::materials` remain
- [ ] ARCHITECTURE.md updated with materials consolidation example
- [ ] Code review passes architectural validation

---

## 📚 References

- ARCHITECTURE.md - Layer definitions
- clean-architecture - Robert C. Martin (2017)
- Domain-Driven Design - Eric Evans (2003)

---

**Status**: IDENTIFIED (Ready for Implementation)  
**Owner**: Architecture Team  
**Created**: 2026-01-29  
**Target Sprint**: Next Sprint  
**Severity**: HIGH (Architectural Correctness)
