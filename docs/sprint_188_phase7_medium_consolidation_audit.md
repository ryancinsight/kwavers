# Sprint 188 Phase 7: Medium and Materials Consolidation Audit

**Date**: 2024
**Phase**: 7 - Domain Medium Consolidation
**Objective**: Consolidate all material property definitions into `domain/medium` as Single Source of Truth (SSOT)
**Status**: Phase 7.1 ✅ COMPLETE | Phase 7.2-7.7 In Progress

---

## Executive Summary

### Current State
- **Test Status**: 1101 passing, 0 failing, 11 ignored ✅
- **Domain Boundary**: Canonical types established in Phase 6 ✅
- **Domain Medium**: Canonical property structs created in Phase 7.1 ✅
- **Material Properties**: Phase 7.1 complete, 6 duplicates ready for migration
- **Phase 7.1**: COMPLETE - Canonical structs implemented and tested

### Violations Inventory

#### Critical: Material Property Duplicates (6 instances)
1. **`domain/boundary/advanced.rs::MaterialProperties`**
   - Fields: density, sound_speed, impedance, absorption
   - Used by: Impedance and admittance boundary conditions
   - **Violation**: Domain boundary should not define medium properties

2. **`physics/acoustics/mechanics/cavitation/damage.rs::MaterialProperties`**
   - Fields: yield_strength, ultimate_strength, hardness, density, fatigue_exponent
   - Used by: Cavitation damage modeling
   - **Violation**: Physics layer redefining strength properties

3. **`physics/acoustics/mechanics/elastic_wave/properties.rs::ElasticProperties`**
   - Fields: density, lambda, mu, youngs_modulus, poisson_ratio, bulk_modulus, shear_modulus
   - Used by: Elastic wave propagation
   - **Violation**: Physics layer redefining elastic properties (domain/medium/elastic.rs already has trait)

4. **`physics/thermal/mod.rs::ThermalProperties`**
   - Fields: k (conductivity), c (specific heat), rho (density), w_b (blood perfusion), c_b (blood heat)
   - Used by: Bio-heat equation solver
   - **Violation**: Physics layer redefining thermal properties (domain/medium/thermal.rs already has trait)

5. **`physics/electromagnetic/equations.rs::EMMaterialProperties`**
   - Fields: permittivity, permeability, conductivity, relaxation_time (all ArrayD<f64>)
   - Used by: EM wave equation solvers
   - **Violation**: Physics layer defining EM material arrays; no canonical struct in domain

6. **`clinical/therapy/lithotripsy/stone_fracture.rs::StoneMaterial`**
   - Fields: density, youngs_modulus, poisson_ratio, tensile_strength
   - Used by: Kidney stone fracture mechanics
   - **Violation**: Clinical layer defining material properties; should compose domain types

---

## Architecture Analysis

### Existing Domain Medium Structure

#### Trait-Based Architecture (✓ Good Foundation)
```
domain/medium/
├── traits.rs              → Medium (top-level trait)
├── core.rs                → CoreMedium (density, sound speed)
├── acoustic.rs            → AcousticProperties trait
├── elastic.rs             → ElasticProperties, ElasticArrayAccess traits
├── optical.rs             → OpticalProperties trait
├── thermal.rs             → ThermalProperties, ThermalField traits
├── viscous.rs             → ViscousProperties trait
└── bubble.rs              → BubbleProperties, BubbleState traits
```

**Observation**: Domain layer has trait definitions but lacks canonical **data structs** for property composition.

#### Missing Canonical Property Structs
- No `AcousticPropertyData` struct (only trait methods)
- No `ElasticPropertyData` struct (only trait methods)
- No `ElectromagneticPropertyData` struct (trait missing entirely)
- No `StrengthPropertyData` struct (mechanical failure properties)
- No `MaterialPropertyComposite` struct (multi-physics composition)

**Root Cause**: Trait-only architecture forces downstream modules to define their own structs.

---

## Canonical Design Specification

### Principle: Trait + Struct Duality
Each property domain should have:
1. **Trait**: Behavioral contract (computation, spatial variation)
2. **Struct**: Canonical data representation (composition, serialization)

### Proposed Canonical Structs

#### 1. Acoustic Property Data
```rust
/// Canonical acoustic material properties
///
/// Mathematical Foundation:
/// - Wave equation: ∂²p/∂t² = c²∇²p - α∂p/∂t + β(∇p)²
/// - Impedance: Z = ρc (kg/m²s)
/// - Absorption: α(f) = α₀ f^y (Np/m)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticPropertyData {
    /// Density ρ (kg/m³)
    pub density: f64,
    /// Sound speed c (m/s)
    pub sound_speed: f64,
    /// Absorption coefficient α₀ (Np/(MHz^y m))
    pub absorption_coefficient: f64,
    /// Absorption power exponent y (dimensionless)
    pub absorption_power: f64,
    /// Nonlinearity parameter B/A (dimensionless)
    pub nonlinearity: f64,
}

impl AcousticPropertyData {
    /// Acoustic impedance Z = ρc
    pub fn impedance(&self) -> f64 {
        self.density * self.sound_speed
    }
    
    /// Absorption at frequency f (MHz) → α(f) = α₀ f^y (Np/m)
    pub fn absorption_at_frequency(&self, freq_mhz: f64) -> f64 {
        self.absorption_coefficient * freq_mhz.powf(self.absorption_power)
    }
}
```

#### 2. Elastic Property Data
```rust
/// Canonical elastic material properties
///
/// Mathematical Foundation:
/// - Stress-strain: σ = λ tr(ε)I + 2με
/// - Wave speeds: c_p = √((λ+2μ)/ρ), c_s = √(μ/ρ)
/// - Relations: λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticPropertyData {
    /// Density ρ (kg/m³)
    pub density: f64,
    /// Lamé first parameter λ (Pa)
    pub lambda: f64,
    /// Lamé second parameter μ (shear modulus) (Pa)
    pub mu: f64,
}

impl ElasticPropertyData {
    /// Construct from Young's modulus E and Poisson's ratio ν
    pub fn from_engineering(density: f64, youngs_modulus: f64, poisson_ratio: f64) -> Self {
        let lambda = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
        Self { density, lambda, mu }
    }
    
    /// Young's modulus E = μ(3λ+2μ)/(λ+μ)
    pub fn youngs_modulus(&self) -> f64 {
        self.mu * (3.0 * self.lambda + 2.0 * self.mu) / (self.lambda + self.mu)
    }
    
    /// Poisson's ratio ν = λ/(2(λ+μ))
    pub fn poisson_ratio(&self) -> f64 {
        self.lambda / (2.0 * (self.lambda + self.mu))
    }
    
    /// Bulk modulus K = λ + 2μ/3
    pub fn bulk_modulus(&self) -> f64 {
        self.lambda + 2.0 * self.mu / 3.0
    }
    
    /// P-wave speed c_p = √((λ+2μ)/ρ)
    pub fn p_wave_speed(&self) -> f64 {
        ((self.lambda + 2.0 * self.mu) / self.density).sqrt()
    }
    
    /// S-wave speed c_s = √(μ/ρ)
    pub fn s_wave_speed(&self) -> f64 {
        (self.mu / self.density).sqrt()
    }
}
```

#### 3. Electromagnetic Property Data
```rust
/// Canonical electromagnetic material properties
///
/// Mathematical Foundation:
/// - Maxwell: ∇×E = -∂B/∂t, ∇×H = J + ∂D/∂t
/// - Constitutive: D = εE, B = μH, J = σE
/// - Wave speed: c = 1/√(εμ)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectromagneticPropertyData {
    /// Relative permittivity ε_r (dimensionless)
    pub permittivity: f64,
    /// Relative permeability μ_r (dimensionless)
    pub permeability: f64,
    /// Electrical conductivity σ (S/m)
    pub conductivity: f64,
    /// Dielectric relaxation time τ (s, optional)
    pub relaxation_time: Option<f64>,
}

impl ElectromagneticPropertyData {
    /// Wave speed c = c₀/√(ε_r μ_r)
    pub fn wave_speed(&self) -> f64 {
        const C0: f64 = 299_792_458.0; // m/s
        C0 / (self.permittivity * self.permeability).sqrt()
    }
    
    /// Impedance Z = √(μ/ε)
    pub fn impedance(&self) -> f64 {
        const Z0: f64 = 376.730_313_668; // Ω (vacuum impedance)
        Z0 * (self.permeability / self.permittivity).sqrt()
    }
}
```

#### 4. Strength Property Data
```rust
/// Canonical mechanical strength properties
///
/// Mathematical Foundation:
/// - Von Mises: σ_eq = √(3J₂) ≤ σ_y
/// - Fatigue: N = C (Δσ)^(-b)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrengthPropertyData {
    /// Yield strength σ_y (Pa)
    pub yield_strength: f64,
    /// Ultimate tensile strength σ_u (Pa)
    pub ultimate_strength: f64,
    /// Hardness H (Pa)
    pub hardness: f64,
    /// Fatigue strength exponent b (dimensionless)
    pub fatigue_exponent: f64,
}
```

#### 5. Thermal Property Data
```rust
/// Canonical thermal material properties
///
/// Mathematical Foundation:
/// - Heat equation: ρc ∂T/∂t = ∇·(k∇T) + Q
/// - Bio-heat: add perfusion term w_b c_b (T_b - T)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalPropertyData {
    /// Thermal conductivity k (W/m/K)
    pub conductivity: f64,
    /// Specific heat capacity c (J/kg/K)
    pub specific_heat: f64,
    /// Density ρ (kg/m³)
    pub density: f64,
    /// Blood perfusion rate w_b (kg/m³/s, optional for bio-heat)
    pub blood_perfusion: Option<f64>,
    /// Blood specific heat c_b (J/kg/K, optional for bio-heat)
    pub blood_specific_heat: Option<f64>,
}

impl ThermalPropertyData {
    /// Thermal diffusivity α = k/(ρc)
    pub fn thermal_diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }
}
```

#### 6. Material Property Composite
```rust
/// Composite material properties for multi-physics simulations
///
/// Domain Rule: This is the canonical composition point for all material properties.
/// Each physics module should extract only the properties it needs.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialProperties {
    /// Acoustic properties (always present)
    pub acoustic: AcousticPropertyData,
    /// Elastic properties (optional, for solids)
    pub elastic: Option<ElasticPropertyData>,
    /// Electromagnetic properties (optional, for EM waves)
    pub electromagnetic: Option<ElectromagneticPropertyData>,
    /// Strength properties (optional, for damage/fracture)
    pub strength: Option<StrengthPropertyData>,
    /// Thermal properties (optional, for thermal effects)
    pub thermal: Option<ThermalPropertyData>,
}

impl MaterialProperties {
    /// Create acoustic-only material (e.g., water)
    pub fn acoustic_only(acoustic: AcousticPropertyData) -> Self {
        Self {
            acoustic,
            elastic: None,
            electromagnetic: None,
            strength: None,
            thermal: None,
        }
    }
    
    /// Create elastic material with acoustic coupling
    pub fn elastic(acoustic: AcousticPropertyData, elastic: ElasticPropertyData) -> Self {
        Self {
            acoustic,
            elastic: Some(elastic),
            electromagnetic: None,
            strength: None,
            thermal: None,
        }
    }
    
    /// Builder pattern for multi-physics
    pub fn builder() -> MaterialPropertiesBuilder { ... }
}
```

---

## Migration Plan

### Phase 7.1: Create Canonical Property Structs ✅ COMPLETE

**File**: `kwavers/src/domain/medium/properties.rs` (1480 lines)

**Status**: ✅ **COMPLETE**
**Completion Date**: 2024
**Test Results**: 26 tests passing, 0 failing
**Full Suite**: 1101 passing, 0 failing, 11 ignored

**Implemented**:
1. ✅ 5 property data structs:
   - `AcousticPropertyData` (with impedance, absorption at frequency)
   - `ElasticPropertyData` (with E,ν ↔ λ,μ conversion, wave speeds)
   - `ElectromagneticPropertyData` (with wave speed, impedance, skin depth)
   - `StrengthPropertyData` (with Von Mises, Basquin's law)
   - `ThermalPropertyData` (with diffusivity, bio-heat support)
2. ✅ `MaterialProperties` composite with builder pattern
3. ✅ Comprehensive unit tests (26 tests):
   - Engineering parameter conversions (E,ν ↔ λ,μ) ✅
   - Derived quantities (impedance, wave speeds) ✅
   - Boundary value validation ✅
4. ✅ Mathematical invariant tests:
   - Poisson ratio: -1 < ν < 0.5 ✅
   - Positive definiteness: ρ,c,k,E,μ > 0 ✅
   - Physical bounds: ε_r ≥ 1, μ_r ≥ 0 ✅

**Verified**:
```bash
cargo test --lib properties::tests
# Result: 26 passed, 0 failed

cargo test --workspace --lib
# Result: 1101 passed, 0 failed, 11 ignored
```

**Exported** via `domain/medium/mod.rs`:
```rust
pub mod properties;
pub use properties::{
    AcousticPropertyData,
    ElasticPropertyData,
    ElectromagneticPropertyData,
    StrengthPropertyData,
    ThermalPropertyData,
    MaterialProperties,
    MaterialPropertiesBuilder,
};
```

**Documentation**: See `docs/sprint_188_phase7_1_complete.md` for full report

---

### Phase 7.2: Replace Domain Boundary Violation (0.5 hours)

**File**: `kwavers/src/domain/boundary/advanced.rs`

**Current Violation**:
```rust
pub struct MaterialProperties { // Line 49-58
    pub density: f64,
    pub sound_speed: f64,
    pub impedance: f64,
    pub absorption: f64,
}
```

**Action**: Remove struct, import canonical type
```rust
use crate::domain::medium::properties::AcousticPropertyData;

// Replace MaterialProperties usage with AcousticPropertyData
// Update ImpedanceBoundary and AdmittanceBoundary to use new type
```

**Affected Structs**:
- `ImpedanceBoundary` (uses `material1`, `material2`)
- `AdmittanceBoundary` (uses `material1`, `material2`)

**Test Coverage**:
```bash
cargo test --lib domain::boundary::advanced
```

---

### Phase 7.3: Replace Physics Elastic Violation (1 hour)

**File**: `kwavers/src/physics/acoustics/mechanics/elastic_wave/properties.rs`

**Current Violation**:
```rust
pub struct ElasticProperties { // Lines 11-27
    pub density: f64,
    pub lambda: f64,
    pub mu: f64,
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub bulk_modulus: f64,
    pub shear_modulus: f64,
}
```

**Action**: Remove struct, import canonical type
```rust
use crate::domain::medium::properties::ElasticPropertyData;

// Remove redundant ElasticProperties struct
// Use ElasticPropertyData directly
// Note: Canonical struct computes derived quantities on-demand via methods
```

**Rationale**: Storing derived quantities (E, ν, K from λ,μ) violates DRY and creates consistency risks.

**Test Coverage**:
```bash
cargo test --lib physics::acoustics::mechanics::elastic_wave
```

---

### Phase 7.4: Replace Physics Thermal Violation (1 hour)

**File**: `kwavers/src/physics/thermal/mod.rs`

**Current Violation**:
```rust
pub struct ThermalProperties { // Lines 23-38
    pub k: f64,    // conductivity
    pub c: f64,    // specific heat
    pub rho: f64,  // density
    pub w_b: f64,  // blood perfusion
    pub c_b: f64,  // blood specific heat
}
```

**Action**: Remove struct, import canonical type
```rust
use crate::domain::medium::properties::ThermalPropertyData;

// Replace ThermalProperties with ThermalPropertyData
// Update BioHeatEquation and related solvers
```

**Note**: Canonical type uses `Option<f64>` for bio-heat parameters (not all materials are biological).

**Test Coverage**:
```bash
cargo test --lib physics::thermal
```

---

### Phase 7.5: Replace Physics Cavitation Violation (0.5 hours)

**File**: `kwavers/src/physics/acoustics/mechanics/cavitation/damage.rs`

**Current Violation**:
```rust
pub struct MaterialProperties { // Lines 41-52
    pub yield_strength: f64,
    pub ultimate_strength: f64,
    pub hardness: f64,
    pub density: f64,
    pub fatigue_exponent: f64,
}
```

**Action**: Remove struct, import canonical types
```rust
use crate::domain::medium::properties::{AcousticPropertyData, StrengthPropertyData};

// Replace with StrengthPropertyData
// Note: density comes from AcousticPropertyData, not strength properties
// Cavitation models should take both acoustic and strength inputs
```

**Test Coverage**:
```bash
cargo test --lib physics::acoustics::mechanics::cavitation
```

---

### Phase 7.6: Replace Physics EM Violation (1 hour)

**File**: `kwavers/src/physics/electromagnetic/equations.rs`

**Current Violation**:
```rust
pub struct EMMaterialProperties { // Lines 221-230
    pub permittivity: ArrayD<f64>,
    pub permeability: ArrayD<f64>,
    pub conductivity: ArrayD<f64>,
    pub relaxation_time: Option<ArrayD<f64>>,
}
```

**Action**: Import canonical type, adapt for array usage
```rust
use crate::domain::medium::properties::ElectromagneticPropertyData;
use ndarray::ArrayD;

// Strategy: Keep array-based struct for efficient spatial representation
// But provide conversion to/from canonical pointwise type
impl EMMaterialProperties {
    /// Construct from pointwise property function
    pub fn from_property_fn<F>(shape: &[usize], f: F) -> Self
    where
        F: Fn(usize, usize, usize) -> ElectromagneticPropertyData
    { ... }
    
    /// Get canonical properties at index
    pub fn at(&self, i: usize, j: usize, k: usize) -> ElectromagneticPropertyData { ... }
}
```

**Rationale**: EM solvers use array-based storage for performance. Keep the array struct but document its relationship to canonical pointwise type.

**Alternative**: Rename to `EMPropertyArrays` to clarify it's an array storage format, not a canonical definition.

**Test Coverage**:
```bash
cargo test --lib physics::electromagnetic
```

---

### Phase 7.7: Replace Clinical Stone Violation (0.5 hours)

**File**: `kwavers/src/clinical/therapy/lithotripsy/stone_fracture.rs`

**Current Violation**:
```rust
pub struct StoneMaterial { // Lines 10-19
    pub density: f64,
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub tensile_strength: f64,
}
```

**Action**: Remove struct, compose canonical types
```rust
use crate::domain::medium::properties::{ElasticPropertyData, StrengthPropertyData};

pub struct StoneMaterial {
    pub elastic: ElasticPropertyData,
    pub strength: StrengthPropertyData,
}

impl StoneMaterial {
    pub fn new(density: f64, youngs_modulus: f64, poisson_ratio: f64, tensile_strength: f64) -> Self {
        Self {
            elastic: ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio),
            strength: StrengthPropertyData {
                yield_strength: tensile_strength * 0.8, // Approximation
                ultimate_strength: tensile_strength,
                hardness: tensile_strength * 3.0, // Approximation for brittle materials
                fatigue_exponent: 10.0, // Default for ceramics/stones
            },
        }
    }
}
```

**Test Coverage**:
```bash
cargo test --lib clinical::therapy::lithotripsy
```

---

## Validation Strategy

### Post-Migration Test Matrix

| Module | Unit Tests | Integration Tests | Property Tests |
|--------|-----------|-------------------|----------------|
| `domain/medium/properties` | ✓ Engineering conversions | - | ✓ Physical bounds |
| `domain/boundary/advanced` | ✓ Impedance calculations | ✓ Reflection coefficients | - |
| `physics/acoustics/elastic_wave` | ✓ Wave speeds | ✓ Propagation accuracy | - |
| `physics/thermal` | ✓ Diffusivity | ✓ Bio-heat equation | - |
| `physics/acoustics/cavitation` | ✓ Damage criteria | ✓ Fatigue modeling | - |
| `physics/electromagnetic` | ✓ Array conversion | ✓ Maxwell solver | - |
| `clinical/lithotripsy` | ✓ Stone fracture | ✓ Pressure thresholds | - |

### Acceptance Criteria

**All must pass**:
1. Full test suite: `cargo test --workspace --lib` → 0 failures
2. No material property structs outside `domain/medium/properties.rs`
3. All physics modules import from canonical location
4. Derived quantities computed on-demand (no storage of redundant E,ν,K)
5. Documentation updated with canonical type usage examples

### Regression Prevention

**Search Patterns** (should return zero results after migration):
```bash
# No struct MaterialProperties outside domain/medium
rg "pub struct MaterialProperties" --type rust | grep -v "domain/medium/properties.rs"

# No struct ElasticProperties outside domain/medium
rg "pub struct ElasticProperties" --type rust | grep -v "domain/medium/properties.rs"

# No struct ThermalProperties outside domain/medium
rg "pub struct ThermalProperties" --type rust | grep -v "domain/medium/properties.rs"
```

---

## Timeline Estimate

| Phase | Task | Estimated Time | Status |
|-------|------|----------------|--------|
| 7.1 | Create canonical property structs + tests | 1.5 hours | ✅ COMPLETE |
| 7.2 | Domain boundary migration | 0.5 hours | ⏳ Next |
| 7.3 | Physics elastic migration | 1.0 hours | ⏳ Pending |
| 7.4 | Physics thermal migration | 1.0 hours | ⏳ Pending |
| 7.5 | Physics cavitation migration | 0.5 hours | ⏳ Pending |
| 7.6 | Physics EM migration | 1.0 hours | ⏳ Pending |
| 7.7 | Clinical stone migration | 0.5 hours | ⏳ Pending |
| **Total** | | **6.0 hours** | **1.5h / 6.0h (25%)** |

**Incremental Execution**: Commit and test after each phase to maintain green build.

---

## Success Metrics

1. **Code Deduplication**: 6 duplicate material property structs eliminated
2. **SSOT Enforcement**: 100% of material properties defined in `domain/medium`
3. **Test Coverage**: All migrated modules retain 100% passing tests
4. **Zero Regressions**: No test failures introduced during migration
5. **Documentation**: ADR published documenting canonical property architecture

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Test failures during migration | High | Incremental approach; test after each phase |
| API breaking changes | Medium | Provide type aliases and deprecation warnings |
| Performance regression (EM arrays) | Low | Keep array-based storage; add conversion methods |
| Incomplete coverage of use cases | Medium | Comprehensive property test coverage |

---

## Next Steps

1. **Review and Approve**: Validate design of canonical property structs
2. **Execute Phase 7.1**: Create `domain/medium/properties.rs`
3. **Incremental Migration**: Phases 7.2-7.7 in sequence
4. **Final Verification**: Run full test suite and search for violations
5. **Documentation**: Publish ADR and update developer guide

**Phase 7.1 Complete ✅ - Ready to proceed with Phase 7.2 migration**

---

## Phase 7.1 Completion Summary

### What Was Delivered

1. **Canonical Structs** (6 types):
   - `AcousticPropertyData`: density, sound_speed, absorption, nonlinearity
   - `ElasticPropertyData`: density, lambda, mu (with E,ν conversion)
   - `ElectromagneticPropertyData`: permittivity, permeability, conductivity
   - `StrengthPropertyData`: yield, ultimate, hardness, fatigue
   - `ThermalPropertyData`: conductivity, specific_heat, density, bio-heat
   - `MaterialProperties`: Multi-physics composite with builder

2. **Validation & Invariants**:
   - All constructors enforce physical constraints
   - Derived quantities computed on-demand (no redundant storage)
   - Mathematical relationships verified in tests

3. **Material Presets**:
   - Water, soft tissue, bone, steel
   - Each with appropriate multi-physics properties

4. **Test Coverage**: 26 tests, 100% passing
   - Acoustic: impedance, absorption, validation
   - Elastic: conversions, wave speeds, bounds, moduli
   - EM: wave speed, refractive index, skin depth, validation
   - Strength: hardness estimation, validation
   - Thermal: diffusivity, bio-heat, validation
   - Composite: builder, presets, bounds

### Integration
- Exported via `domain/medium/mod.rs`
- No regressions: 1101 tests passing (11 ignored)
- Ready for downstream migration (Phases 7.2-7.7)