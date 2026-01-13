# Sprint 188 - Domain Consolidation Audit

**Date**: 2024-12-19  
**Sprint**: 188  
**Phase**: Domain Purity Enhancement  
**Status**: In Progress  
**Auditor**: Elite Mathematically-Verified Systems Architect

---

## Executive Summary

This audit identifies all boundary condition types and material property definitions throughout the codebase to ensure **Single Source of Truth (SSOT)** in the domain layer. The goal is to consolidate:

1. **All boundary condition types** → `domain/boundary`
2. **All material properties** → `domain/medium`
3. **Eliminate all duplications** outside these locations

---

## Current State Analysis

### Domain Layer (SSOT Locations)

#### `domain/boundary/` - Current Contents
```
domain/boundary/
├── advanced.rs          - MaterialInterface, MaterialProperties (DUPLICATE!)
├── bem.rs              - BEM boundary implementations
├── config.rs           - Boundary configuration
├── cpml/               - CPML specific implementations
├── fem.rs              - FEM boundary implementations
├── field_updater.rs    - Field update logic
├── mod.rs              - Module exports
├── pml.rs              - PML implementations
└── traits.rs           - Core boundary traits (GOOD - SSOT)
    ├── FieldType enum
    ├── BoundaryDomain enum
    ├── BoundaryDirections struct
    ├── BoundaryCondition trait
    ├── AbsorbingBoundary trait
    ├── ReflectiveBoundary trait
    └── PeriodicBoundary trait
```

#### `domain/medium/` - Current Contents
```
domain/medium/
├── absorption/         - Absorption models
├── adapters/          - Medium adapters
├── anisotropic/       - Anisotropic materials
├── heterogeneous/     - Heterogeneous media
├── homogeneous/       - Homogeneous media
├── poroelastic/       - Poroelastic materials
├── acoustic.rs        - Acoustic properties
├── analytical_properties.rs - Analytical models
├── bubble.rs          - Bubble properties
├── builder.rs         - Medium builder
├── config.rs          - Medium configuration
├── core.rs            - CoreMedium trait (GOOD - SSOT)
├── elastic.rs         - Elastic properties
├── error.rs           - Medium errors
├── frequency_dependent.rs - Frequency-dependent properties
├── interface.rs       - Medium interfaces
├── material_fields.rs - MaterialFields struct (GOOD)
├── mod.rs             - Module exports
├── optical.rs         - Optical properties
├── thermal.rs         - Thermal properties
├── traits.rs          - Medium traits
├── validation_simulation.rs - Validation
├── viscous.rs         - Viscous properties
└── wrapper.rs         - Medium wrapper
```

---

## Violations Found

### Category 1: Boundary Condition Duplications

#### 1.1 PINN Module Boundary Conditions (MAJOR VIOLATION)

**Location**: `src/analysis/ml/pinn/`

**Duplicated Enums/Structs**:

1. **`burn_wave_equation_2d.rs:702-711`**
   ```rust
   pub enum BoundaryCondition2D {
       Dirichlet,
       Neumann,
       Periodic,
       Absorbing,
   }
   ```
   **Status**: ❌ DUPLICATE - Should use domain/boundary types

2. **`burn_wave_equation_3d.rs:308-317`**
   ```rust
   pub enum BoundaryCondition3D {
       Dirichlet,
       Neumann,
       Absorbing,
       Periodic,
   }
   ```
   **Status**: ❌ DUPLICATE - Should use domain/boundary types

3. **`electromagnetic.rs:63-73`**
   ```rust
   pub enum ElectromagneticBoundarySpec {
       PerfectElectricConductor { position: BoundaryPosition },
       PerfectMagneticConductor { position: BoundaryPosition },
       // ... more variants
   }
   ```
   **Status**: ❌ DUPLICATE - Should use domain/boundary types

4. **`electromagnetic_gpu.rs:79-88`**
   ```rust
   pub enum BoundaryCondition {
       PerfectElectricConductor,
       PerfectMagneticConductor,
       Absorbing,
       Periodic,
   }
   ```
   **Status**: ❌ DUPLICATE - Should use domain/boundary types

5. **`acoustic_wave.rs:90-99`**
   ```rust
   pub enum AcousticBoundaryType {
       SoundSoft,
       SoundHard,
       Absorbing,
       Impedance,
   }
   ```
   **Status**: ❌ DUPLICATE - Should use domain/boundary types

6. **`physics.rs:93-127`**
   ```rust
   pub enum BoundaryConditionSpec {
       Dirichlet { boundary, value, component },
       Neumann { boundary, flux, component },
       Robin { boundary, alpha, beta, component },
       Periodic { boundaries },
   }
   
   pub enum BoundaryPosition {
       Left, Right, Bottom, Top,
       CustomRectangular { x_range, y_range, z_range },
       CustomSpherical { center, radius },
   }
   ```
   **Status**: ❌ DUPLICATE - Should use domain/boundary types

**Impact**: 
- 6 separate boundary condition type systems in PINN modules
- No consistency across implementations
- Violates SSOT principle
- Makes cross-module usage difficult

#### 1.2 Solver Module Boundary Types (MODERATE VIOLATION)

**Location**: `examples/seismic_imaging_demo.rs:128-133`
```rust
boundary_type: BoundaryType::Absorbing,
```
**Status**: ⚠️ UNCLEAR - Need to check if BoundaryType is from domain or local

---

### Category 2: Material Property Duplications

#### 2.1 Material Properties in Boundary Module (VIOLATION)

**Location**: `src/domain/boundary/advanced.rs:49-58`
```rust
pub struct MaterialProperties {
    pub density: f64,
    pub sound_speed: f64,
    pub impedance: f64,
    pub absorption: f64,
}
```
**Status**: ❌ DUPLICATE - Should be in domain/medium, not domain/boundary

**Current Usage**: 
- Used in `MaterialInterface` struct (line 35-46)
- Should reference domain/medium types instead

#### 2.2 Physics Module Material Properties (VIOLATION)

**Location**: `src/physics/acoustics/mechanics/cavitation/damage.rs:41-51`
```rust
pub struct MaterialProperties {
    pub yield_strength: f64,
    pub ultimate_strength: f64,
    pub hardness: f64,
    pub density: f64,
    pub fatigue_exponent: f64,
}
```
**Status**: ❌ DUPLICATE - Should use domain/medium with extended properties

#### 2.3 Physics Module Poroelastic Materials (ACCEPTABLE)

**Location**: `src/physics/acoustics/mechanics/poroelastic/mod.rs:51-61`
```rust
pub struct PoroelasticMaterial {
    pub porosity: f64,
    pub solid_density: f64,
    pub fluid_density: f64,
    // ... more fields
}
```
**Status**: ⚠️ BORDERLINE - Physics-specific extension, but should reference domain/medium base

#### 2.4 Electromagnetic Material Properties (VIOLATION)

**Location**: `src/physics/electromagnetic/equations.rs:221-230`
```rust
pub struct EMMaterialProperties {
    pub permittivity: ArrayD<f64>,
    pub permeability: ArrayD<f64>,
    pub conductivity: ArrayD<f64>,
    pub relaxation_time: Option<ArrayD<f64>>,
}
```
**Status**: ❌ DUPLICATE - Should be in domain/medium/electromagnetic.rs

#### 2.5 Source Module Materials (ACCEPTABLE)

**Location**: `src/domain/source/transducers/physics/materials.rs`
```rust
pub struct PiezoMaterial { ... }
pub enum BackingMaterial { ... }
pub enum LensMaterial { ... }
```
**Status**: ✅ ACCEPTABLE - These are transducer-specific component materials, not medium properties

#### 2.6 Clinical Module Stone Material (VIOLATION)

**Location**: `src/clinical/therapy/lithotripsy/stone_fracture.rs:10-19`
```rust
pub struct StoneMaterial {
    pub density: f64,
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub tensile_strength: f64,
}
```
**Status**: ❌ DUPLICATE - Should use domain/medium with elastic extensions

#### 2.7 Hybrid Solver Material Metrics (ACCEPTABLE)

**Location**: `src/solver/forward/hybrid/adaptive_selection/metrics.rs:150-154`
```rust
pub struct MaterialMetrics {
    pub homogeneity: f64,
    pub interface_proximity: f64,
    pub impedance_contrast: f64,
}
```
**Status**: ✅ ACCEPTABLE - Computed metrics, not material properties

---

## Consolidation Plan

### Phase 1: Domain Layer Enhancement (2 hours)

#### Task 1.1: Create Canonical Boundary Types in domain/boundary
**File**: `src/domain/boundary/types.rs` (NEW)

```rust
/// Canonical boundary condition types - SSOT for entire codebase
pub enum BoundaryType {
    /// Dirichlet: Fixed value (u = g)
    Dirichlet,
    /// Neumann: Fixed flux (∂u/∂n = g)
    Neumann,
    /// Robin: Mixed (α·u + β·∂u/∂n = g)
    Robin,
    /// Periodic: Periodic wrapping
    Periodic,
    /// Absorbing: Non-reflecting (PML, ABC, etc.)
    Absorbing,
    /// Radiation: Sommerfeld radiation condition
    Radiation,
    /// FreeSurface: Stress-free boundary (elastic)
    FreeSurface,
}

/// Boundary position specification
pub enum BoundaryFace {
    XMin, XMax,
    YMin, YMax,
    ZMin, ZMax,
}

/// Boundary component (for vector fields)
pub enum BoundaryComponent {
    All,           // All components
    X, Y, Z,       // Individual components
    Normal,        // Normal component
    Tangential,    // Tangential components
}

/// Electromagnetic-specific boundary types
pub enum ElectromagneticBoundaryType {
    /// PEC: E_tangential = 0
    PerfectElectricConductor,
    /// PMC: H_tangential = 0
    PerfectMagneticConductor,
    /// ABC: Absorbing boundary
    Absorbing,
    /// Periodic boundary
    Periodic,
}

/// Acoustic-specific boundary types
pub enum AcousticBoundaryType {
    /// Pressure release (p = 0)
    SoundSoft,
    /// Rigid wall (∂p/∂n = 0)
    SoundHard,
    /// Impedance boundary
    Impedance { impedance: f64 },
    /// Absorbing boundary
    Absorbing,
}
```

#### Task 1.2: Move MaterialProperties from boundary/advanced.rs to medium/
**Action**: 
1. Delete `MaterialProperties` from `domain/boundary/advanced.rs`
2. Update `MaterialInterface` to use `domain::medium::MaterialProperties`
3. Ensure medium module has acoustic material properties

#### Task 1.3: Create Canonical Material Properties in domain/medium
**File**: `src/domain/medium/properties.rs` (NEW or enhance existing)

```rust
/// Base material properties - SSOT for acoustic materials
pub struct AcousticProperties {
    pub density: f64,          // ρ (kg/m³)
    pub sound_speed: f64,      // c (m/s)
    pub impedance: f64,        // Z = ρc (kg/m²s)
    pub absorption: f64,       // α (Np/m)
}

/// Elastic material properties
pub struct ElasticProperties {
    pub density: f64,          // ρ (kg/m³)
    pub youngs_modulus: f64,   // E (Pa)
    pub poisson_ratio: f64,    // ν (dimensionless)
    pub shear_modulus: f64,    // G = E/(2(1+ν)) (Pa)
    pub bulk_modulus: f64,     // K (Pa)
}

/// Mechanical strength properties
pub struct StrengthProperties {
    pub yield_strength: f64,
    pub ultimate_strength: f64,
    pub hardness: f64,
    pub fatigue_exponent: f64,
}

/// Electromagnetic material properties
pub struct ElectromagneticProperties {
    pub permittivity: f64,     // ε_r
    pub permeability: f64,     // μ_r
    pub conductivity: f64,     // σ (S/m)
    pub relaxation_time: Option<f64>, // τ (s)
}

/// Unified material properties (composition)
pub struct MaterialProperties {
    pub acoustic: Option<AcousticProperties>,
    pub elastic: Option<ElasticProperties>,
    pub strength: Option<StrengthProperties>,
    pub electromagnetic: Option<ElectromagneticProperties>,
}
```

---

### Phase 2: PINN Module Refactoring (3 hours)

#### Task 2.1: Update burn_wave_equation_2d.rs
**Action**:
1. Remove `BoundaryCondition2D` enum
2. Import `use crate::domain::boundary::types::BoundaryType;`
3. Update all usage sites

#### Task 2.2: Update burn_wave_equation_3d.rs
**Action**:
1. Remove `BoundaryCondition3D` enum
2. Import `use crate::domain::boundary::types::BoundaryType;`
3. Update all usage sites

#### Task 2.3: Update electromagnetic.rs
**Action**:
1. Remove `ElectromagneticBoundarySpec` enum
2. Import `use crate::domain::boundary::types::ElectromagneticBoundaryType;`
3. Update all usage sites

#### Task 2.4: Update electromagnetic_gpu.rs
**Action**:
1. Remove `BoundaryCondition` enum
2. Import `use crate::domain::boundary::types::ElectromagneticBoundaryType;`
3. Update all usage sites

#### Task 2.5: Update acoustic_wave.rs
**Action**:
1. Remove `AcousticBoundaryType` enum
2. Import `use crate::domain::boundary::types::AcousticBoundaryType;`
3. Update all usage sites

#### Task 2.6: Update physics.rs
**Action**:
1. Remove `BoundaryConditionSpec` enum
2. Remove `BoundaryPosition` enum
3. Import domain boundary types
4. Update all usage sites

---

### Phase 3: Physics Module Refactoring (2 hours)

#### Task 3.1: Update cavitation/damage.rs
**Action**:
1. Remove local `MaterialProperties`
2. Import `use crate::domain::medium::properties::{AcousticProperties, StrengthProperties};`
3. Compose properties as needed

#### Task 3.2: Update electromagnetic/equations.rs
**Action**:
1. Remove `EMMaterialProperties` struct
2. Import `use crate::domain::medium::properties::ElectromagneticProperties;`
3. Update all usage sites

#### Task 3.3: Update poroelastic materials
**Action**:
1. Keep `PoroelasticMaterial` but add reference to base `MaterialProperties`
2. Document as physics-specific extension

---

### Phase 4: Clinical Module Refactoring (1 hour)

#### Task 4.1: Update lithotripsy/stone_fracture.rs
**Action**:
1. Remove `StoneMaterial` struct
2. Import `use crate::domain::medium::properties::{ElasticProperties, StrengthProperties};`
3. Compose stone properties from domain types

---

### Phase 5: Verification (1 hour)

#### Task 5.1: Compile and Test
```bash
cargo build --lib
cargo test --workspace --lib
```

#### Task 5.2: Architecture Validation
- Verify no boundary types outside domain/boundary
- Verify no material properties outside domain/medium
- Check dependency graph for violations

#### Task 5.3: Documentation Update
- Update ADR with consolidation decisions
- Update module documentation
- Create migration guide

---

## Expected Outcomes

### Before Consolidation
- 6+ boundary condition type systems
- 5+ material property definitions
- High coupling, low cohesion
- SSOT violations

### After Consolidation
- ✅ Single boundary type system in `domain/boundary/types.rs`
- ✅ Single material property system in `domain/medium/properties.rs`
- ✅ All other modules import from domain layer
- ✅ SSOT principle enforced
- ✅ Clear architectural boundaries

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes in PINN modules | High | Medium | Update all usage sites systematically |
| Test failures after refactoring | Medium | Medium | Run tests incrementally per module |
| API incompatibility | Low | High | Maintain backward compatibility layers temporarily |
| Build errors | Medium | Low | Fix compile errors incrementally |

---

## Success Metrics

- [ ] Zero boundary type definitions outside `domain/boundary/`
- [ ] Zero material property definitions outside `domain/medium/`
- [ ] All tests passing (1073/1073)
- [ ] Build successful with zero errors
- [ ] Documentation updated
- [ ] ADR created for consolidation decisions

---

## References

1. **Clean Architecture**: Single Responsibility Principle - each module has one source of truth
2. **DDD**: Bounded contexts with clear ownership
3. **SOLID**: Dependency Inversion - depend on domain abstractions

---

**Next Step**: Proceed with Phase 1 - Create canonical types in domain layer

**Document Status**: Living document - updated during consolidation