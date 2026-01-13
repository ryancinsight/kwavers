# Sprint 188 - Phase 6: Domain Consolidation Plan

**Date**: 2024-12-19  
**Sprint**: 188  
**Phase**: 6 - Domain Layer SSOT Enforcement  
**Status**: Planning Complete  
**Priority**: P0 - Architectural Foundation

---

## Executive Summary

Phase 6 establishes **Single Source of Truth (SSOT)** for boundary conditions and material properties in the domain layer. This consolidation eliminates 6+ duplicate boundary type systems and 5+ material property definitions, ensuring all modules import from canonical domain types.

### Goals

1. ✅ **Canonical Boundary Types**: All boundary conditions in `domain/boundary/types.rs`
2. ✅ **Canonical Material Properties**: All material properties in `domain/medium/properties.rs`
3. ✅ **Eliminate Duplications**: Remove all duplicate definitions outside domain layer
4. ✅ **Update All References**: Systematically update all import sites
5. ✅ **Maintain Tests**: 100% test pass rate throughout consolidation

---

## Phase 6 Strategy

### Incremental Consolidation Approach

**Principle**: Fix one module at a time, verify tests pass, commit, repeat.

**Benefits**:
- Isolates errors to single module
- Maintains working codebase at each step
- Easy rollback if needed
- Clear progress tracking

### Consolidation Order

```
Priority 1: Domain Layer Foundation (COMPLETE)
  ├─ Create domain/boundary/types.rs ✅
  ├─ Create domain/medium/properties.rs (NEXT)
  └─ Export from domain/boundary/mod.rs ✅

Priority 2: High-Impact Modules (3-4 hours)
  ├─ PINN modules (6 files)
  ├─ Physics modules (3 files)
  └─ Clinical modules (1 file)

Priority 3: Low-Impact Modules (1-2 hours)
  ├─ Examples
  ├─ Tests
  └─ Documentation updates

Priority 4: Verification & Cleanup (1 hour)
  ├─ Run full test suite
  ├─ Verify no duplicates remain
  ├─ Update documentation
  └─ Create ADR
```

---

## Detailed Consolidation Plan

### Step 1: Create Canonical Material Properties (30 min)

**File**: `src/domain/medium/properties.rs`

```rust
//! Canonical Material Properties - Single Source of Truth
//!
//! All material property definitions for the entire codebase.
//! Other modules MUST import from here.

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Base acoustic material properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticProperties {
    /// Density ρ (kg/m³)
    pub density: f64,
    /// Sound speed c (m/s)
    pub sound_speed: f64,
    /// Acoustic impedance Z = ρc (kg/m²s)
    pub impedance: f64,
    /// Absorption coefficient α (Np/m)
    pub absorption: f64,
    /// Nonlinearity parameter B/A (dimensionless)
    pub nonlinearity: f64,
}

impl AcousticProperties {
    /// Create new acoustic properties with computed impedance
    pub fn new(density: f64, sound_speed: f64, absorption: f64, nonlinearity: f64) -> Self {
        Self {
            density,
            sound_speed,
            impedance: density * sound_speed,
            absorption,
            nonlinearity,
        }
    }

    /// Water at 20°C (reference material)
    pub fn water() -> Self {
        Self::new(1000.0, 1500.0, 0.002, 5.0)
    }

    /// Tissue (average soft tissue)
    pub fn tissue() -> Self {
        Self::new(1050.0, 1540.0, 0.5, 6.0)
    }
}

/// Elastic material properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElasticProperties {
    /// Density ρ (kg/m³)
    pub density: f64,
    /// Young's modulus E (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio ν (dimensionless, 0 < ν < 0.5)
    pub poisson_ratio: f64,
    /// Shear modulus G = E/(2(1+ν)) (Pa)
    pub shear_modulus: f64,
    /// Bulk modulus K (Pa)
    pub bulk_modulus: f64,
    /// Lamé first parameter λ (Pa)
    pub lame_lambda: f64,
    /// Lamé second parameter μ = G (Pa)
    pub lame_mu: f64,
}

impl ElasticProperties {
    /// Create elastic properties with derived parameters
    pub fn new(density: f64, youngs_modulus: f64, poisson_ratio: f64) -> Self {
        let shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
        let bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
        let lame_lambda =
            youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        let lame_mu = shear_modulus;

        Self {
            density,
            youngs_modulus,
            poisson_ratio,
            shear_modulus,
            bulk_modulus,
            lame_lambda,
            lame_mu,
        }
    }

    /// Steel (reference material)
    pub fn steel() -> Self {
        Self::new(7850.0, 200e9, 0.3)
    }

    /// Aluminum
    pub fn aluminum() -> Self {
        Self::new(2700.0, 70e9, 0.33)
    }
}

/// Mechanical strength properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrengthProperties {
    /// Yield strength σ_y (Pa)
    pub yield_strength: f64,
    /// Ultimate tensile strength σ_u (Pa)
    pub ultimate_strength: f64,
    /// Hardness (Pa)
    pub hardness: f64,
    /// Fatigue strength exponent (dimensionless)
    pub fatigue_exponent: f64,
    /// Fracture toughness K_IC (Pa·m^½)
    pub fracture_toughness: f64,
}

/// Electromagnetic material properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectromagneticProperties {
    /// Relative permittivity ε_r (dimensionless)
    pub permittivity: f64,
    /// Relative permeability μ_r (dimensionless)
    pub permeability: f64,
    /// Electrical conductivity σ (S/m)
    pub conductivity: f64,
    /// Dielectric relaxation time τ (s)
    pub relaxation_time: Option<f64>,
}

impl ElectromagneticProperties {
    /// Vacuum/air properties
    pub fn vacuum() -> Self {
        Self {
            permittivity: 1.0,
            permeability: 1.0,
            conductivity: 0.0,
            relaxation_time: None,
        }
    }

    /// Water at optical frequencies
    pub fn water_optical() -> Self {
        Self {
            permittivity: 1.77, // n^2 = 1.33^2
            permeability: 1.0,
            conductivity: 0.0,
            relaxation_time: None,
        }
    }
}

/// Spatially-varying electromagnetic properties
#[derive(Debug, Clone)]
pub struct ElectromagneticPropertiesArray {
    /// Relative permittivity ε_r (dimensionless)
    pub permittivity: ArrayD<f64>,
    /// Relative permeability μ_r (dimensionless)
    pub permeability: ArrayD<f64>,
    /// Electrical conductivity σ (S/m)
    pub conductivity: ArrayD<f64>,
    /// Dielectric relaxation time τ (s)
    pub relaxation_time: Option<ArrayD<f64>>,
}

/// Thermal properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalProperties {
    /// Thermal conductivity k (W/m·K)
    pub conductivity: f64,
    /// Specific heat capacity c_p (J/kg·K)
    pub specific_heat: f64,
    /// Thermal diffusivity α = k/(ρ·c_p) (m²/s)
    pub diffusivity: f64,
}

/// Unified material properties (composition pattern)
///
/// Combines all physics-domain properties into a single structure.
/// Only populate fields relevant to your simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaterialProperties {
    /// Acoustic properties (for ultrasound simulation)
    pub acoustic: Option<AcousticProperties>,
    /// Elastic properties (for solid mechanics)
    pub elastic: Option<ElasticProperties>,
    /// Strength properties (for damage/fracture)
    pub strength: Option<StrengthProperties>,
    /// Electromagnetic properties (for optics/EM)
    pub electromagnetic: Option<ElectromagneticProperties>,
    /// Thermal properties (for heat transfer)
    pub thermal: Option<ThermalProperties>,
}

impl MaterialProperties {
    /// Create properties with only acoustic data
    pub fn acoustic_only(acoustic: AcousticProperties) -> Self {
        Self {
            acoustic: Some(acoustic),
            ..Default::default()
        }
    }

    /// Create properties with acoustic and elastic data
    pub fn acoustic_elastic(acoustic: AcousticProperties, elastic: ElasticProperties) -> Self {
        Self {
            acoustic: Some(acoustic),
            elastic: Some(elastic),
            ..Default::default()
        }
    }
}
```

**Update**: `src/domain/medium/mod.rs`
```rust
pub mod properties;
pub use properties::{
    AcousticProperties, ElasticProperties, ElectromagneticProperties,
    ElectromagneticPropertiesArray, MaterialProperties, StrengthProperties,
    ThermalProperties,
};
```

---

### Step 2: Fix domain/boundary/advanced.rs (15 min)

**Problem**: `MaterialProperties` defined in boundary module (wrong layer)

**Solution**: Remove duplicate, import from medium

```rust
// BEFORE (domain/boundary/advanced.rs)
pub struct MaterialProperties {
    pub density: f64,
    pub sound_speed: f64,
    pub impedance: f64,
    pub absorption: f64,
}

// AFTER
use crate::domain::medium::properties::AcousticProperties as MaterialProperties;

// OR if we need the impedance field name:
pub struct MaterialInterface {
    pub position: [f64; 3],
    pub normal: [f64; 3],
    pub material_1: AcousticProperties, // renamed from MaterialProperties
    pub material_2: AcousticProperties,
    pub thickness: f64,
}
```

---

### Step 3: Fix PINN Modules (2 hours)

#### 3.1: burn_wave_equation_2d.rs

**Remove**:
```rust
pub enum BoundaryCondition2D {
    Dirichlet,
    Neumann,
    Periodic,
    Absorbing,
}
```

**Add**:
```rust
use crate::domain::boundary::types::BoundaryType;

// Update all usage sites:
// BoundaryCondition2D::Dirichlet → BoundaryType::Dirichlet
// BoundaryCondition2D::Neumann → BoundaryType::Neumann
// etc.
```

**Affected Functions**: Search for `BoundaryCondition2D` and replace with `BoundaryType`

#### 3.2: burn_wave_equation_3d.rs

**Remove**:
```rust
pub enum BoundaryCondition3D { ... }
```

**Add**:
```rust
use crate::domain::boundary::types::BoundaryType;
```

#### 3.3: electromagnetic.rs

**Remove**:
```rust
pub enum ElectromagneticBoundarySpec { ... }
```

**Add**:
```rust
use crate::domain::boundary::types::ElectromagneticBoundaryType;
```

#### 3.4: electromagnetic_gpu.rs

**Remove**:
```rust
pub enum BoundaryCondition { ... }
```

**Add**:
```rust
use crate::domain::boundary::types::ElectromagneticBoundaryType;
```

#### 3.5: acoustic_wave.rs

**Remove**:
```rust
pub enum AcousticBoundaryType { ... }
```

**Add**:
```rust
use crate::domain::boundary::types::AcousticBoundaryType;
```

#### 3.6: physics.rs

**Remove**:
```rust
pub enum BoundaryConditionSpec { ... }
pub enum BoundaryPosition { ... }
```

**Add**:
```rust
use crate::domain::boundary::types::{BoundaryType, BoundaryFace, BoundaryComponent};
```

---

### Step 4: Fix Physics Modules (1 hour)

#### 4.1: physics/acoustics/mechanics/cavitation/damage.rs

**Remove**:
```rust
pub struct MaterialProperties {
    pub yield_strength: f64,
    pub ultimate_strength: f64,
    pub hardness: f64,
    pub density: f64,
    pub fatigue_exponent: f64,
}
```

**Add**:
```rust
use crate::domain::medium::properties::{AcousticProperties, StrengthProperties};

pub struct DamageProperties {
    pub acoustic: AcousticProperties,
    pub strength: StrengthProperties,
}
```

#### 4.2: physics/electromagnetic/equations.rs

**Remove**:
```rust
pub struct EMMaterialProperties {
    pub permittivity: ArrayD<f64>,
    pub permeability: ArrayD<f64>,
    pub conductivity: ArrayD<f64>,
    pub relaxation_time: Option<ArrayD<f64>>,
}
```

**Add**:
```rust
use crate::domain::medium::properties::ElectromagneticPropertiesArray;

// Rename all EMMaterialProperties → ElectromagneticPropertiesArray
```

#### 4.3: physics/acoustics/mechanics/poroelastic/mod.rs

**Keep** `PoroelasticMaterial` but add reference to base properties:

```rust
use crate::domain::medium::properties::{AcousticProperties, ElasticProperties};

pub struct PoroelasticMaterial {
    /// Base acoustic properties
    pub acoustic: AcousticProperties,
    /// Base elastic properties
    pub elastic: ElasticProperties,
    /// Porosity (0 < φ < 1)
    pub porosity: f64,
    // ... other poro-specific fields
}
```

---

### Step 5: Fix Clinical Modules (30 min)

#### 5.1: clinical/therapy/lithotripsy/stone_fracture.rs

**Remove**:
```rust
pub struct StoneMaterial {
    pub density: f64,
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub tensile_strength: f64,
}
```

**Add**:
```rust
use crate::domain::medium::properties::{ElasticProperties, StrengthProperties};

pub struct StoneMaterial {
    pub elastic: ElasticProperties,
    pub strength: StrengthProperties,
}

impl StoneMaterial {
    pub fn kidney_stone_calcium_oxalate() -> Self {
        Self {
            elastic: ElasticProperties::new(2000.0, 10e9, 0.25),
            strength: StrengthProperties {
                yield_strength: 5e6,
                ultimate_strength: 10e6,
                hardness: 3e6,
                fatigue_exponent: 8.0,
                fracture_toughness: 1.0e6,
            },
        }
    }
}
```

---

### Step 6: Verification (1 hour)

#### 6.1: Search for Remaining Duplicates

```bash
# Search for boundary condition enums
rg "enum.*Boundary.*Condition" --type rust

# Search for material property structs
rg "struct.*Material.*Properties" --type rust

# Should only find domain/boundary/types.rs and domain/medium/properties.rs
```

#### 6.2: Run Tests

```bash
# Incremental testing after each module
cargo test --lib <module_path>

# Full suite at end
cargo test --workspace --lib
```

#### 6.3: Build Verification

```bash
cargo build --lib
cargo clippy --lib -- -D warnings
```

---

## Migration Checklist

### Domain Layer (Foundation)
- [x] Create `domain/boundary/types.rs` with canonical boundary types
- [ ] Create `domain/medium/properties.rs` with canonical material properties
- [x] Export types from `domain/boundary/mod.rs`
- [ ] Export properties from `domain/medium/mod.rs`

### Boundary Consolidation
- [ ] Remove `MaterialProperties` from `domain/boundary/advanced.rs`
- [ ] Update `MaterialInterface` to use `domain/medium/properties`
- [ ] Remove `BoundaryCondition2D` from `analysis/ml/pinn/burn_wave_equation_2d.rs`
- [ ] Remove `BoundaryCondition3D` from `analysis/ml/pinn/burn_wave_equation_3d.rs`
- [ ] Remove `ElectromagneticBoundarySpec` from `analysis/ml/pinn/electromagnetic.rs`
- [ ] Remove `BoundaryCondition` from `analysis/ml/pinn/electromagnetic_gpu.rs`
- [ ] Remove `AcousticBoundaryType` from `analysis/ml/pinn/acoustic_wave.rs`
- [ ] Remove `BoundaryConditionSpec` from `analysis/ml/pinn/physics.rs`
- [ ] Remove `BoundaryPosition` from `analysis/ml/pinn/physics.rs`

### Material Properties Consolidation
- [ ] Remove `MaterialProperties` from `physics/acoustics/mechanics/cavitation/damage.rs`
- [ ] Remove `EMMaterialProperties` from `physics/electromagnetic/equations.rs`
- [ ] Refactor `PoroelasticMaterial` to reference base properties
- [ ] Remove `StoneMaterial` from `clinical/therapy/lithotripsy/stone_fracture.rs`

### Testing & Verification
- [ ] Run tests after each module update
- [ ] Full test suite passes (1073/1073)
- [ ] No boundary types outside domain/boundary
- [ ] No material properties outside domain/medium
- [ ] Documentation updated

---

## Rollback Plan

If consolidation causes unforeseen issues:

1. **Per-Module Rollback**: Git revert specific module changes
2. **Full Rollback**: Git revert entire consolidation branch
3. **Incremental Retry**: Fix identified issues, retry failed module

**Safety Net**: Each module update is a separate commit with passing tests.

---

## Expected Outcomes

### Before Phase 6
- 6+ boundary condition type systems (duplicated across modules)
- 5+ material property definitions (violating SSOT)
- High coupling between layers
- Unclear architectural boundaries

### After Phase 6
- ✅ Single boundary type system: `domain/boundary/types.rs`
- ✅ Single material properties system: `domain/medium/properties.rs`
- ✅ All modules import from domain layer (dependency inversion)
- ✅ SSOT principle enforced
- ✅ Clear architectural boundaries
- ✅ 100% test pass rate maintained
- ✅ Zero architectural violations

---

## Success Criteria

1. ✅ Zero boundary type definitions outside `domain/boundary/types.rs`
2. ✅ Zero material property definitions outside `domain/medium/properties.rs`
3. ✅ All tests passing: 1073/1073
4. ✅ Build successful with zero errors
5. ✅ Clippy clean (or documented exceptions)
6. ✅ Documentation synchronized
7. ✅ ADR created documenting consolidation

---

## Estimated Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Create domain/boundary/types.rs | 30 min | ✅ COMPLETE |
| 2 | Create domain/medium/properties.rs | 30 min | NEXT |
| 3 | Fix domain/boundary/advanced.rs | 15 min | Pending |
| 4 | Fix PINN modules (6 files) | 2 hours | Pending |
| 5 | Fix physics modules (3 files) | 1 hour | Pending |
| 6 | Fix clinical modules (1 file) | 30 min | Pending |
| 7 | Verification & testing | 1 hour | Pending |
| 8 | Documentation & ADR | 30 min | Pending |
| **TOTAL** | | **6.5 hours** | 8% complete |

---

## References

1. **Clean Architecture** (Martin, 2017): Dependency inversion - depend on abstractions
2. **Domain-Driven Design** (Evans, 2003): Ubiquitous language in bounded contexts
3. **SOLID Principles**: Single Responsibility - one module, one truth

---

**Next Action**: Create `domain/medium/properties.rs` with canonical material properties

**Document Status**: Planning Complete - Ready for Execution