//! Stone fracture mechanics for lithotripsy simulation.
//!
//! This module implements material fracture models for kidney stone fragmentation
//! under shock wave loading.
//!
//! # Architecture
//!
//! Stone materials compose canonical domain property types:
//! - `ElasticPropertyData`: Young's modulus, Poisson's ratio, wave speeds
//! - `StrengthPropertyData`: Yield/ultimate strength, hardness, fatigue
//!
//! This ensures consistency with the domain SSOT and enables reuse of validated
//! material models across physics modules.

use crate::domain::medium::properties::{ElasticPropertyData, StrengthPropertyData};
use ndarray::Array3;

/// Stone material properties composing canonical domain types
///
/// # Domain Composition
///
/// Stone materials are characterized by:
/// 1. **Elastic behavior**: Stress-strain response under loading (wave propagation)
/// 2. **Strength limits**: Yield and fracture thresholds (damage initiation)
///
/// By composing canonical domain types, we ensure:
/// - Consistent property validation across all physics modules
/// - Derived quantities (wave speeds, moduli) computed correctly
/// - Reusable material definitions for multi-physics coupling
///
/// # Example
///
/// ```
/// use kwavers::clinical::therapy::lithotripsy::stone_fracture::StoneMaterial;
///
/// let stone = StoneMaterial::calcium_oxalate_monohydrate();
/// assert_eq!(stone.elastic().density, 2000.0);
/// assert!(stone.elastic().p_wave_speed() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct StoneMaterial {
    /// Elastic properties (density, Young's modulus, Poisson's ratio, wave speeds)
    elastic: ElasticPropertyData,

    /// Strength properties (yield, ultimate tensile strength, hardness)
    strength: StrengthPropertyData,
}

impl StoneMaterial {
    /// Construct stone material from canonical domain property types
    ///
    /// # Arguments
    ///
    /// - `elastic`: Elastic material properties (density, moduli, wave speeds)
    /// - `strength`: Strength properties (yield, ultimate, hardness)
    ///
    /// # Invariants
    ///
    /// Both property types enforce their own validation constraints:
    /// - Elastic: positive density, valid Poisson's ratio bounds, consistent moduli
    /// - Strength: yield ≤ ultimate, positive hardness, valid fatigue parameters
    pub fn new(elastic: ElasticPropertyData, strength: StrengthPropertyData) -> Self {
        Self { elastic, strength }
    }

    /// Get elastic properties
    #[inline]
    pub fn elastic(&self) -> &ElasticPropertyData {
        &self.elastic
    }

    /// Get strength properties
    #[inline]
    pub fn strength(&self) -> &StrengthPropertyData {
        &self.strength
    }

    /// Density (kg/m³) — convenience accessor
    ///
    /// Forwards to elastic property data for ergonomics at call sites.
    #[inline]
    pub fn density(&self) -> f64 {
        self.elastic.density
    }

    /// Young's modulus (Pa) — convenience accessor
    #[inline]
    pub fn youngs_modulus(&self) -> f64 {
        self.elastic.youngs_modulus()
    }

    /// Poisson's ratio (dimensionless) — convenience accessor
    #[inline]
    pub fn poisson_ratio(&self) -> f64 {
        self.elastic.poisson_ratio()
    }

    /// Tensile strength (Pa) — convenience accessor
    ///
    /// Uses ultimate tensile strength as the fracture threshold.
    #[inline]
    pub fn tensile_strength(&self) -> f64 {
        self.strength.ultimate_strength
    }

    /// Calcium oxalate monohydrate (common kidney stone)
    ///
    /// # Material Properties
    ///
    /// - Density: 2000 kg/m³
    /// - Young's modulus: 20 GPa
    /// - Poisson's ratio: 0.3
    /// - Tensile strength: 5 MPa
    /// - Yield strength: 4 MPa (estimated as 0.8 × tensile)
    /// - Hardness: 12 MPa (estimated as 3 × yield)
    ///
    /// # References
    ///
    /// - Williams et al. (2003): "Mechanical properties of urinary stones"
    /// - Zohdi & Kuypers (2006): "Modeling and simulation of kidney stone fragmentation"
    pub fn calcium_oxalate_monohydrate() -> Self {
        let density = 2000.0;
        let youngs_modulus = 20e9;
        let poisson_ratio = 0.3;
        let tensile_strength = 5e6;

        // Estimate yield strength as 80% of ultimate tensile strength (common for brittle materials)
        let yield_strength = 0.8 * tensile_strength;

        // Estimate hardness using standard relationship H ≈ 3σ_y
        let hardness = 3.0 * yield_strength;

        // Fatigue exponent for ceramic-like materials (high value = low fatigue sensitivity)
        let fatigue_exponent = 15.0;

        let elastic = ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio);
        let strength =
            StrengthPropertyData::new(yield_strength, tensile_strength, hardness, fatigue_exponent)
                .expect("Valid stone strength parameters");

        Self { elastic, strength }
    }

    /// Uric acid stone (softer, more deformable)
    ///
    /// # Material Properties
    ///
    /// - Density: 1800 kg/m³
    /// - Young's modulus: 10 GPa (softer than calcium oxalate)
    /// - Poisson's ratio: 0.35
    /// - Tensile strength: 3 MPa
    /// - Yield strength: 2.4 MPa
    /// - Hardness: 7.2 MPa
    ///
    /// # Clinical Notes
    ///
    /// Uric acid stones are generally more friable and fragment more easily
    /// than calcium-based stones, requiring lower shock wave energies.
    pub fn uric_acid() -> Self {
        let density = 1800.0;
        let youngs_modulus = 10e9;
        let poisson_ratio = 0.35;
        let tensile_strength = 3e6;
        let yield_strength = 0.8 * tensile_strength;
        let hardness = 3.0 * yield_strength;
        let fatigue_exponent = 12.0;

        let elastic = ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio);
        let strength =
            StrengthPropertyData::new(yield_strength, tensile_strength, hardness, fatigue_exponent)
                .expect("Valid stone strength parameters");

        Self { elastic, strength }
    }

    /// Cystine stone (hardest, most resistant)
    ///
    /// # Material Properties
    ///
    /// - Density: 2100 kg/m³
    /// - Young's modulus: 30 GPa (very hard)
    /// - Poisson's ratio: 0.28
    /// - Tensile strength: 8 MPa
    /// - Yield strength: 6.4 MPa
    /// - Hardness: 19.2 MPa
    ///
    /// # Clinical Notes
    ///
    /// Cystine stones are notoriously difficult to fragment and often require
    /// higher shock wave energies and longer treatment times.
    pub fn cystine() -> Self {
        let density = 2100.0;
        let youngs_modulus = 30e9;
        let poisson_ratio = 0.28;
        let tensile_strength = 8e6;
        let yield_strength = 0.8 * tensile_strength;
        let hardness = 3.0 * yield_strength;
        let fatigue_exponent = 18.0; // High fatigue resistance

        let elastic = ElasticPropertyData::from_engineering(density, youngs_modulus, poisson_ratio);
        let strength =
            StrengthPropertyData::new(yield_strength, tensile_strength, hardness, fatigue_exponent)
                .expect("Valid stone strength parameters");

        Self { elastic, strength }
    }
}

/// Stone fracture mechanics model
///
/// # Mathematical Foundation
///
/// Implements cumulative damage mechanics for shock wave loading:
///
/// ```text
/// dD/dt = f(σ, σ_threshold, strain_rate)
/// ```
///
/// Where:
/// - `D`: Damage parameter (0 = intact, 1 = failed)
/// - `σ`: Applied stress (tensile component)
/// - `σ_threshold`: Material tensile strength
///
/// # Damage Accumulation
///
/// Each shock wave pulse contributes incremental damage based on:
/// 1. Peak tensile stress magnitude
/// 2. Stress duration and rate of loading
/// 3. Material fatigue characteristics
///
/// When `D ≥ 1` at a spatial location, the material is considered fractured.
#[derive(Debug, Clone)]
pub struct StoneFractureModel {
    /// Stone material properties (elastic + strength)
    material: StoneMaterial,

    /// Spatial damage field (0 = intact, 1 = fractured)
    damage_field: Array3<f64>,

    /// Fragment size distribution [m]
    fragment_sizes: Vec<f64>,
}

impl StoneFractureModel {
    /// Create new fracture model for given material and grid dimensions
    ///
    /// # Arguments
    ///
    /// - `material`: Stone material properties (elastic + strength composition)
    /// - `dimensions`: Grid dimensions (nx, ny, nz) for damage field
    pub fn new(material: StoneMaterial, dimensions: (usize, usize, usize)) -> Self {
        Self {
            material,
            damage_field: Array3::zeros(dimensions),
            fragment_sizes: Vec::new(),
        }
    }

    /// Get material properties
    #[inline]
    pub fn material(&self) -> &StoneMaterial {
        &self.material
    }

    /// Apply stress loading to update damage field
    ///
    /// # Mathematical Model
    ///
    /// Implements a threshold-based damage accumulation model:
    ///
    /// ```text
    /// If σ_tensile > σ_threshold:
    ///     ΔD = k · (σ_tensile / σ_threshold - 1)
    /// ```
    ///
    /// Where:
    /// - `σ_tensile`: Tensile stress (negative pressure in shock wave)
    /// - `σ_threshold`: Material tensile strength
    /// - `k`: Damage rate parameter (currently 0.01 per event)
    ///
    /// # Arguments
    ///
    /// - `stress_field`: 3D stress field (Pa), where negative values = tension
    /// - `_dt`: Time step (currently unused, for future rate-dependent models)
    /// - `_strain_rate`: Strain rate (currently unused, for future models)
    ///
    /// # Implementation Notes
    ///
    /// - Shock waves produce large tensile stresses during rarefaction phase
    /// - Tensile failure is the primary fragmentation mechanism in lithotripsy
    /// - Current model is quasi-static; future versions should include:
    ///   - Strain rate effects (higher rates → higher apparent strength)
    ///   - Fatigue accumulation across multiple pulses
    ///   - Crack propagation and coalescence
    pub fn apply_stress_loading(
        &mut self,
        stress_field: &Array3<f64>,
        _dt: f64,
        _strain_rate: f64,
    ) {
        // Get tensile strength from canonical strength properties
        let threshold = self.material.tensile_strength();

        // Damage rate parameter (incremental damage per threshold exceedance)
        const DAMAGE_RATE: f64 = 0.01;

        for ((i, j, k), &stress) in stress_field.indexed_iter() {
            // Tensile stress corresponds to negative pressure in shock wave convention
            // If stress magnitude exceeds tensile strength, accumulate damage
            if stress < -threshold {
                // Compute overstress ratio
                let overstress_ratio = (-stress / threshold) - 1.0;

                // Accumulate damage proportional to overstress
                self.damage_field[[i, j, k]] += DAMAGE_RATE * (1.0 + overstress_ratio);

                // Clamp damage to [0, 1]
                if self.damage_field[[i, j, k]] > 1.0 {
                    self.damage_field[[i, j, k]] = 1.0;
                }
            }
        }
    }

    /// Get current damage field
    ///
    /// # Returns
    ///
    /// Reference to 3D damage field where:
    /// - `0.0`: Intact material
    /// - `1.0`: Fully fractured
    /// - Intermediate values: Partial damage
    #[inline]
    pub fn damage_field(&self) -> &Array3<f64> {
        &self.damage_field
    }

    /// Get distribution of fragment sizes
    ///
    /// # Returns
    ///
    /// Slice of fragment characteristic sizes in meters.
    ///
    /// # Implementation Note
    ///
    /// Fragment size analysis is performed by post-processing the damage field
    /// to identify connected fractured regions. Currently placeholder.
    #[inline]
    pub fn fragment_sizes(&self) -> &[f64] {
        &self.fragment_sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calcium_oxalate_properties() {
        let stone = StoneMaterial::calcium_oxalate_monohydrate();

        // Verify elastic properties
        assert_eq!(stone.density(), 2000.0);
        assert_eq!(stone.youngs_modulus(), 20e9);
        assert_eq!(stone.poisson_ratio(), 0.3);

        // Verify derived elastic quantities
        assert!(stone.elastic().p_wave_speed() > 0.0);
        assert!(stone.elastic().s_wave_speed() > 0.0);
        assert!(stone.elastic().s_wave_speed() < stone.elastic().p_wave_speed());

        // Verify strength properties
        assert_eq!(stone.tensile_strength(), 5e6);
        assert!(stone.strength().yield_strength < stone.strength().ultimate_strength);
        assert!(stone.strength().hardness > 0.0);
    }

    #[test]
    fn test_material_composition() {
        let stone = StoneMaterial::calcium_oxalate_monohydrate();

        // Verify composition of canonical types
        assert!(stone.elastic().density > 0.0);
        assert!(stone.strength().ultimate_strength > 0.0);

        // Verify convenience accessors match underlying data
        assert_eq!(stone.density(), stone.elastic().density);
        assert_eq!(stone.youngs_modulus(), stone.elastic().youngs_modulus());
        assert_eq!(stone.tensile_strength(), stone.strength().ultimate_strength);
    }

    #[test]
    fn test_stone_type_differences() {
        let oxalate = StoneMaterial::calcium_oxalate_monohydrate();
        let uric = StoneMaterial::uric_acid();
        let cystine = StoneMaterial::cystine();

        // Uric acid should be softer
        assert!(uric.youngs_modulus() < oxalate.youngs_modulus());
        assert!(uric.tensile_strength() < oxalate.tensile_strength());

        // Cystine should be hardest
        assert!(cystine.youngs_modulus() > oxalate.youngs_modulus());
        assert!(cystine.tensile_strength() > oxalate.tensile_strength());
    }

    #[test]
    fn test_fracture_model_initialization() {
        let stone = StoneMaterial::calcium_oxalate_monohydrate();
        let model = StoneFractureModel::new(stone, (10, 10, 10));

        // Verify initial state
        assert_eq!(model.damage_field().dim(), (10, 10, 10));
        assert_eq!(model.damage_field().sum(), 0.0); // No initial damage
        assert!(model.fragment_sizes().is_empty());
    }

    #[test]
    fn test_damage_accumulation() {
        let stone = StoneMaterial::calcium_oxalate_monohydrate();
        let mut model = StoneFractureModel::new(stone.clone(), (5, 5, 5));

        // Create stress field with tensile stress exceeding threshold
        let mut stress_field = Array3::zeros((5, 5, 5));
        let threshold = stone.tensile_strength();

        // Apply stress twice the tensile strength (negative = tension)
        stress_field[[2, 2, 2]] = -2.0 * threshold;

        // Apply loading
        model.apply_stress_loading(&stress_field, 1e-6, 1e6);

        // Verify damage accumulated at stressed location
        assert!(model.damage_field()[[2, 2, 2]] > 0.0);
        assert!(model.damage_field()[[2, 2, 2]] <= 1.0);

        // Verify no damage at unstressed locations
        assert_eq!(model.damage_field()[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_damage_saturation() {
        let stone = StoneMaterial::calcium_oxalate_monohydrate();
        let mut model = StoneFractureModel::new(stone.clone(), (3, 3, 3));

        // Create extreme stress field
        let mut stress_field = Array3::zeros((3, 3, 3));
        stress_field[[1, 1, 1]] = -100.0 * stone.tensile_strength();

        // Apply loading multiple times
        for _ in 0..100 {
            model.apply_stress_loading(&stress_field, 1e-6, 1e6);
        }

        // Verify damage saturates at 1.0
        assert_eq!(model.damage_field()[[1, 1, 1]], 1.0);
    }

    #[test]
    fn test_no_damage_below_threshold() {
        let stone = StoneMaterial::calcium_oxalate_monohydrate();
        let mut model = StoneFractureModel::new(stone.clone(), (4, 4, 4));

        // Apply stress below threshold
        let mut stress_field = Array3::zeros((4, 4, 4));
        stress_field[[2, 2, 2]] = -0.5 * stone.tensile_strength();

        model.apply_stress_loading(&stress_field, 1e-6, 1e6);

        // Verify no damage occurs
        assert_eq!(model.damage_field()[[2, 2, 2]], 0.0);
    }

    #[test]
    fn test_material_accessor() {
        let stone = StoneMaterial::uric_acid();
        let model = StoneFractureModel::new(stone.clone(), (8, 8, 8));

        // Verify material accessor returns correct properties
        assert_eq!(model.material().density(), stone.density());
        assert_eq!(
            model.material().tensile_strength(),
            stone.tensile_strength()
        );
    }
}
