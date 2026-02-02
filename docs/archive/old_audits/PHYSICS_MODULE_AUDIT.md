# Physics Module Comprehensive Audit & Enhancement Plan

**Date:** January 28, 2026  
**Focus:** Physics-driven architecture, clean implementation, latest research integration  
**Priority:** Production-quality physics implementations with proper module organization

---

## Executive Summary

The kwavers physics module is extensive (~200+ files covering acoustics, thermal, chemistry, optics). This audit identifies:

1. **Architecture strengths** - Well-organized domain separation
2. **Implementation gaps** - Where physics needs enhancement with latest research
3. **Code quality issues** - Dead code, improper module placement
4. **Integration opportunities** - How to connect physics across layers

**Scope:** Phase 4 physics enhancements + cleanup

---

## Physics Module Structure Analysis

### Current Organization

```
src/physics/
├── acoustics/              # Wave propagation physics
│   ├── analysis/          # Post-processing metrics
│   ├── analytical/        # Analytical solutions
│   ├── bubble_dynamics/   # Cavitation physics
│   ├── functional/        # Functional analysis
│   ├── imaging/          # Imaging modalities
│   ├── mechanics/        # Acoustic mechanics
│   ├── skull/            # Transcranial propagation
│   ├── transducer/       # Transducer physics
│   └── state.rs          # Acoustic state types
├── chemistry/             # Chemical processes
│   ├── photochemistry/
│   ├── radical_initiation/
│   ├── reaction_kinetics/
│   ├── ros_plasma/
│   └── reactions.rs
├── optics/               # Optical physics
│   ├── polarization/
│   ├── scattering/
│   └── sonoluminescence/
├── thermal/              # Thermal processes
│   ├── diffusion/
│   ├── perfusion.rs
│   └── thermal_dose.rs
└── factory/             # Physics factory
    └── models.rs
```

### Module Count & Coverage

**Acoustics:** ~100 files (well-developed)  
**Thermal:** ~8 files (basic)  
**Chemistry:** ~12 files (intermediate)  
**Optics:** ~6 files (basic)  
**Factory:** ~1 file (needs enhancement)

**Total Physics LOC:** ~15,000+

---

## Layer 2 (Physics) Responsibilities

Per 8-layer architecture:

```
Layer 2 (Physics) SHALL:
├── Define physical models (PDE, ODEs)
├── Implement constitutive equations
├── Provide material properties
├── Model interactions (thermal, acoustic, chemical)
├── NOT implement solvers (Layer 4 responsibility)
└── NOT define domain structures (Layer 3 responsibility)
```

---

## Audit Findings

### 1. Acoustics Module ✅ WELL-DEVELOPED

**Strengths:**
- Comprehensive wave equations (linear, nonlinear)
- Cavitation physics with control
- Bubble dynamics (Keller-Miksis)
- Transcranial propagation
- Elastic wave support
- Poroelastic waves (Biot)

**Latest Research Integration:**
- ✅ Nonlinear acoustics (Westervelt, KZK)
- ✅ Cavitation detection (broadband, subharmonic)
- ✅ Multi-frequency interactions
- ⚠️ Mode conversion (needs validation)
- ⚠️ Streaming/radiation force (partial)

**Recommended Enhancements:**
- [ ] Add acoustic radiation pressure tensor
- [ ] Implement acoustic streaming (steady)
- [ ] Add harmonic generation models
- [ ] Enhance elastic wave anisotropy
- [ ] Add shear wave conversion

---

### 2. Thermal Module ⚠️ NEEDS EXPANSION

**Current:**
- Bioheat equation (Pennes)
- Thermal dose (CEM43)
- Perfusion models
- Heat diffusion (isotropic)

**Missing (Critical Research Areas):**
- [ ] Hyperbolic heat conduction
- [ ] Temperature-dependent properties
- [ ] Phase change (tissue ablation)
- [ ] Perfusion feedback (temperature-dependent)
- [ ] Anisotropic heat diffusion
- [ ] Coupled thermal-acoustic effects

**Action Items:**
1. Implement hyperbolic heat equation
2. Add tissue phase change models
3. Temperature-dependent material properties
4. Thermal-acoustic coupling terms

**LOC Estimate:** ~500 lines needed

---

### 3. Chemistry Module ⚠️ NEEDS VALIDATION

**Current:**
- ROS (reactive oxygen species) kinetics
- Sonochemistry
- Radical initiation
- Reaction kinetics framework

**Issues:**
- [ ] Verify reaction rate constants (literature values)
- [ ] Validate ROS production mechanisms
- [ ] Check free radical pathway accuracy
- [ ] Implement missing reactions
- [ ] Add concentration-dependent effects

**Action Items:**
1. Audit reaction mechanisms against literature
2. Validate with experimental data
3. Add temperature-dependent kinetics
4. Implement bubble-liquid interface chemistry

**LOC Estimate:** ~300 lines enhancement

---

### 4. Optics Module ⚠️ MINIMAL

**Current:**
- Sonoluminescence (blackbody, spectral)
- Scattering (stub)
- Polarization (stub)

**Missing (Advanced Optics):**
- [ ] Kerr effect (nonlinear optics)
- [ ] Photoacoustic conversion
- [ ] Light scattering models
- [ ] Emission spectroscopy
- [ ] Photon-phonon coupling

**Action Items:**
1. Implement Kerr effect for SL
2. Add photoacoustic energy conversion
3. Implement Mie/Rayleigh scattering
4. Add spectral analysis

**LOC Estimate:** ~400 lines

---

### 5. Code Quality Issues

#### Deprecation/Dead Code

**Location:** Various modules  
**Issues:**
- Old material models (archived, not removed)
- Deprecated parameter names
- Unused analytical solutions
- Test support code (not production)

**Action:** Remove all dead code, keep only production paths

#### Architectural Violations

**Issue 1:** Physics implementing domain concepts  
**Example:** `bubble_dynamics/bubble_field.rs` uses Array3<f64>  
**Fix:** Physics should only provide models, domains implement fields

**Issue 2:** Physics importing solver types  
**Example:** Some modules reference FdtdConfig  
**Fix:** Physics should not know about solvers (Layer 4)

**Issue 3:** Material properties scattered  
**Example:** Tissue properties in multiple locations  
**Fix:** Create unified `MaterialPropertyDatabase` (SSOT)

---

## Architecture Compliance Issues

### Issue A: Material Property SSOT

**Problem:** Tissue properties defined in multiple places:
- `physics/acoustics/transducer/mod.rs`
- `physics/acoustics/skull/mod.rs`
- `domain/medium/properties.rs`
- `clinical/safety/tissue_properties.rs`

**Solution:** 
```
physics/
  └── materials/
      ├── mod.rs                    (SSOT)
      ├── tissue.rs               (tissue types)
      ├── fluids.rs              (blood, CSF, etc.)
      ├── implants.rs            (contrast agents, etc.)
      └── temperature_dependent.rs (T-dependent props)

// All modules import from physics/materials, not elsewhere
use crate::physics::materials::BRAIN_WHITE_MATTER;
```

### Issue B: Wave Equation Separation

**Problem:** Wave equations mixed with solvers  
**Fix:** Physics defines equation, solver implements algorithm

```
// Physics layer defines:
pub struct WesterveltEquation {
    pub nonlinearity: f64,  // B/A coefficient
    pub absorption: f64,    // absorption coefficient
}

// Solver layer implements:
pub struct WesterveltSolver {
    equation: WesterveltEquation,
    // ... solver-specific fields
}
```

### Issue C: Boundary Condition SSOT

**Problem:** BCs defined in multiple layers  
**Fix:** Physics defines BC models, domain implements BC operators

---

## Enhancement Priorities

### Priority 1: CRITICAL (Must do)

1. **Material Properties SSOT** (4 hours)
   - Create unified physics/materials module
   - Remove duplicates from domain/clinical
   - Update all references
   - Add temperature-dependent properties

2. **Thermal Module Expansion** (6 hours)
   - Implement hyperbolic heat equation
   - Add tissue ablation model
   - Implement thermal-acoustic coupling
   - Validate against literature

3. **Code Cleanup** (3 hours)
   - Remove dead code
   - Fix architectural violations
   - Clean up deprecated items
   - Verify zero warnings

### Priority 2: HIGH (Should do)

4. **Acoustics Validation** (8 hours)
   - Verify all equations against peer-reviewed literature
   - Validate material parameters
   - Add reference implementations
   - Document sources

5. **Chemistry Module Validation** (6 hours)
   - Audit reaction kinetics
   - Verify ROS production mechanisms
   - Add missing reactions
   - Temperature-dependent rates

6. **Optics Module Enhancement** (6 hours)
   - Implement Kerr effect
   - Add photoacoustic conversion
   - Implement Mie scattering
   - Add emission spectroscopy

### Priority 3: MEDIUM (Nice to have)

7. **Advanced Acoustic Features** (8 hours)
   - Streaming steady-state solutions
   - Radiation pressure tensor
   - Mode conversion validation
   - Elastic anisotropy

---

## Physics Module Implementation Guide

### 1. New Physics Models

**Template:**
```rust
// src/physics/acoustics/[feature]/mod.rs

//! [Feature] Physics
//! 
//! Implements [physical model] based on [peer-reviewed paper].
//! 
//! # Physical Model
//! 
//! [Governing equation in mathematical notation]
//! 
//! # References
//! 
//! [1] Author et al., "Title", Journal, Year
//! [2] ...

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// [Phenomenon] model parameters
#[derive(Debug, Clone, Copy)]
pub struct [Model]Parameters {
    /// Parameter 1 [units]
    pub param1: f64,
    /// Parameter 2 [units]
    pub param2: f64,
}

impl [Model]Parameters {
    /// Create parameters from physical constants
    pub fn from_physical_constants(
        frequency: f64,
        medium_properties: &crate::physics::materials::MaterialProperties,
    ) -> KwaversResult<Self> {
        // Calculate parameters from frequency, medium, etc.
        Ok(Self {
            param1: calculate_param1(frequency),
            param2: calculate_param2(medium_properties),
        })
    }
}

/// [Phenomenon] model
#[derive(Debug)]
pub struct [Model] {
    parameters: [Model]Parameters,
}

impl [Model] {
    /// Create model from parameters
    pub fn new(parameters: [Model]Parameters) -> Self {
        Self { parameters }
    }
    
    /// Calculate [property] given [inputs]
    pub fn calculate_[property](
        &self,
        input: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // Physics implementation
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_known_values() {
        // Test against published data
    }
    
    #[test]
    fn test_limits() {
        // Test physical limits (e.g., at zero frequency)
    }
}
```

### 2. Material Properties Database

**Template:**
```rust
// src/physics/materials/mod.rs

use serde::{Deserialize, Serialize};

/// Material properties (SSOT - Single Source of Truth)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MaterialProperties {
    /// Speed of sound [m/s]
    pub sound_speed: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Acoustic impedance [kg/(m²·s)]
    pub impedance: f64,
    /// Absorption coefficient [Np/m/MHz]
    pub absorption: f64,
    /// Nonlinearity parameter B/A
    pub nonlinearity: f64,
    /// Specific heat [J/(kg·K)]
    pub specific_heat: f64,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,
}

impl MaterialProperties {
    /// Verify physical constraints
    pub fn validate(&self) -> KwaversResult<()> {
        if self.sound_speed <= 0.0 {
            return Err(PhysicsError::InvalidParameter("speed_of_sound"));
        }
        // ... more validation
        Ok(())
    }
}

/// Tissue type with base properties
pub mod tissue {
    use super::*;
    
    pub const WATER: MaterialProperties = MaterialProperties {
        sound_speed: 1500.0,
        density: 1000.0,
        impedance: 1500000.0,
        absorption: 0.002,
        nonlinearity: 5.0,
        specific_heat: 4186.0,
        thermal_conductivity: 0.6,
    };
    
    pub const BRAIN_WHITE_MATTER: MaterialProperties = MaterialProperties {
        sound_speed: 1540.0,
        density: 1040.0,
        impedance: 1601600.0,
        absorption: 0.6,
        nonlinearity: 6.5,
        specific_heat: 3650.0,
        thermal_conductivity: 0.5,
    };
    
    // ... more tissues (single definition)
}
```

### 3. Physics Constants Module

**Location:** `src/physics/constants.rs`

```rust
//! Physical constants and standard values
//! 
//! Single source of truth for all physical constants used in simulations.
//! Based on CODATA 2018 values and standard literature.

// Fundamental constants
pub const SPEED_OF_LIGHT: f64 = 299792458.0; // m/s
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K
pub const PLANCK_CONSTANT: f64 = 6.62607015e-34; // J·s

// Acoustic constants
pub const STANDARD_SOUND_SPEED_WATER: f64 = 1480.0; // m/s (20°C)
pub const STANDARD_DENSITY_WATER: f64 = 998.2; // kg/m³ (20°C)

// Temperature effects
pub const WATER_SOUND_SPEED_GRADIENT: f64 = 2.0; // (m/s)/°C near 20°C
pub const WATER_THERMAL_EXPANSION: f64 = 2.07e-4; // 1/K

// Cavitation
pub const CAVITATION_THRESHOLD_MECHANICAL_INDEX: f64 = 0.6; // FDA limit
pub const NUCLEATION_RADIUS_MINIMUM: f64 = 1e-6; // 1 µm (typical)

// Tissue properties (defaults)
pub const DEFAULT_TISSUE_PERFUSION: f64 = 0.5; // mL/100g/min (baseline)
pub const DEFAULT_TISSUE_DENSITY: f64 = 1050.0; // kg/m³
pub const DEFAULT_TISSUE_HEAT_CAPACITY: f64 = 3639.0; // J/(kg·K)
```

---

## Physics Development Workflow

### For Each New Physics Feature:

1. **Literature Review** (1-2 hours)
   - Find peer-reviewed papers
   - Identify governing equations
   - Note parameter ranges
   - Document assumptions

2. **Physics Implementation** (2-4 hours)
   - Implement in physics layer
   - Add parameter validation
   - Create reference tests
   - Document equations

3. **Integration** (1 hour)
   - Create solver integration if needed
   - Update domain if needed
   - Add clinical examples

4. **Validation** (2-4 hours)
   - Test against analytical solutions
   - Validate with published data
   - Check physical limits
   - Performance benchmarking

5. **Documentation** (1 hour)
   - Add to physics reference
   - Include equations in doc comments
   - List parameters and units
   - Cite literature

---

## Validation Framework

### For Each Physics Model:

```rust
#[cfg(test)]
mod validation {
    use super::*;
    
    /// Test against analytical solution
    #[test]
    fn test_analytical_solution() {
        // Use known solution (e.g., plane wave)
        let expected = /* analytical solution */;
        let computed = model.compute(/* inputs */);
        assert!((expected - computed).abs() < 1e-6);
    }
    
    /// Test published benchmark
    #[test]
    fn test_published_benchmark() {
        // Reference: Author et al., Journal, Year
        let result = model.compute(/* inputs */);
        let expected = /* published value */;
        assert!((expected - result).abs() / expected < 0.05); // 5% tolerance
    }
    
    /// Test physical limits
    #[test]
    fn test_physical_limits() {
        // As frequency → 0, attenuation → 0
        let attenuation_low = model.attenuation(1.0); // 1 Hz
        let attenuation_zero = model.attenuation(0.0); // 0 Hz
        assert!(attenuation_low <= attenuation_zero);
    }
    
    /// Test dimensional correctness
    #[test]
    fn test_dimensional_analysis() {
        // Output units should match expected
        let impedance = DENSITY * SOUND_SPEED; // kg/(m²·s)
        assert_units_match(impedance, "kg/(m²·s)");
    }
}
```

---

## Integration with Other Layers

### Physics ↔ Domain (One-way: Physics → Domain)

**Allowed:**
- Physics defines material properties
- Domain reads from physics materials

**Not Allowed:**
- Domain tells physics what to calculate
- Physics depends on domain structures

### Physics ↔ Solver (One-way: Physics → Solver)

**Allowed:**
- Solver reads physics equations
- Solver implements physics models

**Not Allowed:**
- Physics references solver types
- Physics implements solver algorithms

### Physics ↔ Clinical (One-way: Physics → Clinical)

**Allowed:**
- Clinical uses physics models
- Clinical references physics parameters

**Not Allowed:**
- Physics contains clinical workflows
- Physics implements clinical logic

---

## Quality Checklist

For each physics module:

- [ ] Documented with governing equations
- [ ] References cited in code
- [ ] Parameters match literature values
- [ ] Validated against analytical solutions
- [ ] Validated against published data
- [ ] Physical limits verified
- [ ] Dimensional analysis passed
- [ ] Zero architectural violations
- [ ] No cross-layer dependencies
- [ ] Comprehensive unit tests
- [ ] Performance benchmarks (if applicable)
- [ ] Example usage provided

---

## Expected Outcomes

### Phase 4 Physics Goals:

1. ✅ Material Properties SSOT
2. ✅ Thermal Module Enhancement
3. ✅ Chemistry Module Validation
4. ✅ Optics Module Expansion
5. ✅ Code Cleanup & Architecture Fix
6. ✅ Comprehensive Validation Tests
7. ✅ Physics Reference Documentation

### Code Quality:

- **Build Status:** 0 errors (maintained)
- **Warnings:** Reduce from 40 to 0 (Phase 5 task)
- **Architecture:** Perfect 8-layer compliance
- **Tests:** 100% of new physics features

### Research Integration:

- Latest acoustics research integrated
- Thermal models based on recent literature
- Chemistry kinetics validated
- Optics properly implemented

---

## Timeline & Effort

**Total Effort:** ~40-50 hours over multiple sessions  
**Phase 4 Focus:** ~10-12 hours on priority 1 items

**Breakdown:**
- Material Properties: 4h
- Thermal Enhancement: 6h
- Code Cleanup: 3h
- Documentation: 2h

**Can complete in this session:** Priority 1 items (13 hours)

---

**Plan Status:** Ready for Implementation  
**Architecture:** Clean, research-driven  
**Quality:** Production-ready standards  

Proceed with Physics Module Enhancement Phase 4.
