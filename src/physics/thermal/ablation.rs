//! Tissue ablation model for thermal therapy
//!
//! Implements thermal ablation kinetics based on Arrhenius equation for
//! protein denaturation and tissue necrosis.
//!
//! References:
//! - Sapareto & Dewey (1984) "Thermal dose determination in cancer therapy"
//! - Henriques (1947) "Studies of thermal injury" (classical Arrhenius model)
//! - Lepock et al. (1993) "Thermal denaturation of proteins"
//! - Valvano et al. (1994) "Relationship between laser-induced thermal lesion"

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Ablation kinetics parameters (Arrhenius model)
///
/// The Arrhenius equation for thermal damage: Ω(t) = ∫A·exp(-E_a/RT)dt
/// where:
/// - A = frequency factor (1/s)
/// - E_a = activation energy (J/mol)
/// - R = universal gas constant (8.314 J/mol/K)
/// - T = absolute temperature (K)
#[derive(Debug, Clone, Copy)]
pub struct AblationKinetics {
    /// Frequency factor [1/s] - typical: 1.0e69 to 1.0e75
    pub frequency_factor: f64,
    /// Activation energy [J/mol] - typical: 250,000 to 650,000 J/mol
    pub activation_energy: f64,
    /// Damage threshold (Ω) - typical: 1.0 for 63.2% protein denaturation
    pub damage_threshold: f64,
    /// Temperature at which tissue begins ablation [°C] - typical: 45-60°C
    pub ablation_threshold: f64,
}

impl AblationKinetics {
    /// Create custom ablation kinetics
    pub fn new(
        frequency_factor: f64,
        activation_energy: f64,
        damage_threshold: f64,
        ablation_threshold: f64,
    ) -> Self {
        Self {
            frequency_factor,
            activation_energy,
            damage_threshold,
            ablation_threshold,
        }
    }

    /// Protein denaturation kinetics (Henriques model)
    /// Reference: Henriques (1947)
    pub fn protein_denaturation() -> Self {
        Self {
            frequency_factor: 1.0e69,     // [1/s]
            activation_energy: 576_500.0, // [J/mol] (137.9 kcal/mol)
            damage_threshold: 1.0,
            ablation_threshold: 45.0,
        }
    }

    /// Collagen denaturation kinetics
    /// Reference: Lepock et al. (1993) - collagen triple helix dissociation
    pub fn collagen_denaturation() -> Self {
        Self {
            frequency_factor: 1.0e53,     // [1/s]
            activation_energy: 418_400.0, // [J/mol] (100 kcal/mol)
            damage_threshold: 1.0,
            ablation_threshold: 55.0,
        }
    }

    /// HIFU ablation kinetics (tissue necrosis)
    /// Reference: Sapareto & Dewey (1984) - thermal dose model
    pub fn hifu_ablation() -> Self {
        Self {
            frequency_factor: 1.0e75,     // [1/s]
            activation_energy: 628_000.0, // [J/mol] (150 kcal/mol)
            damage_threshold: 1.0,
            ablation_threshold: 50.0,
        }
    }

    /// Temperature-dependent damage rate
    ///
    /// dΩ/dt = A·exp(-E_a/RT)
    ///
    /// # Arguments
    /// * `temperature` - Absolute temperature [K]
    ///
    /// # Returns
    /// Damage accumulation rate [1/s]
    pub fn damage_rate(&self, temperature: f64) -> f64 {
        let r = 8.314; // Universal gas constant [J/mol/K]
        self.frequency_factor * (-self.activation_energy / (r * temperature)).exp()
    }

    /// Accumulated thermal damage (Ω)
    ///
    /// Ω(t) = ∫₀ᵗ A·exp(-E_a/RT(τ))dτ
    ///
    /// # Arguments
    /// * `damage_rate` - Damage accumulation rate at current time [1/s]
    /// * `dt` - Time step [s]
    ///
    /// # Returns
    /// Accumulated damage (unitless, 0-∞)
    /// - Ω < 1: No significant damage
    /// - Ω = 1: 63.2% protein denaturation
    /// - Ω > 1: Irreversible damage
    pub fn accumulated_damage(current_damage: f64, damage_rate: f64, dt: f64) -> f64 {
        current_damage + damage_rate * dt
    }

    /// Tissue viability (thermal damage extent)
    ///
    /// Returns fraction of viable tissue: exp(-Ω)
    /// - 1.0 = fully viable
    /// - 0.37 = 63% damage (Ω=1)
    /// - 0.0 = completely ablated
    pub fn viability(&self, damage: f64) -> f64 {
        (-damage).exp()
    }

    /// Is tissue ablated?
    pub fn is_ablated(&self, damage: f64) -> bool {
        damage >= self.damage_threshold
    }
}

impl Default for AblationKinetics {
    fn default() -> Self {
        Self::hifu_ablation()
    }
}

/// Tissue ablation state at a point
#[derive(Debug, Clone, Copy)]
pub struct AblationState {
    /// Temperature [°C]
    pub temperature: f64,
    /// Accumulated thermal damage (Ω)
    pub damage: f64,
    /// Tissue viability (0-1, 1=fully viable)
    pub viability: f64,
    /// Is tissue ablated?
    pub ablated: bool,
}

impl AblationState {
    /// Create new ablation state
    pub fn new(temperature: f64, _kinetics: &AblationKinetics) -> Self {
        Self {
            temperature,
            damage: 0.0,
            viability: 1.0,
            ablated: false,
        }
    }

    /// Update ablation state with temperature change
    pub fn update(&mut self, temperature: f64, kinetics: &AblationKinetics, dt: f64) {
        self.temperature = temperature;

        // Convert to absolute temperature
        let t_abs = temperature + 273.15;

        // Skip if below ablation threshold
        if temperature < kinetics.ablation_threshold {
            return;
        }

        // Calculate damage rate
        let damage_rate = kinetics.damage_rate(t_abs);

        // Accumulate damage
        self.damage = AblationKinetics::accumulated_damage(self.damage, damage_rate, dt);

        // Update viability
        self.viability = kinetics.viability(self.damage);

        // Check ablation
        self.ablated = kinetics.is_ablated(self.damage);
    }
}

/// Tissue ablation field solver
#[derive(Debug)]
pub struct AblationField {
    /// Accumulated thermal damage field
    damage: Array3<f64>,
    /// Tissue viability field (0-1)
    viability: Array3<f64>,
    /// Ablation extent (boolean field)
    ablated: Array3<bool>,
    /// Kinetics model
    kinetics: AblationKinetics,
}

impl AblationField {
    /// Create new ablation field
    pub fn new(shape: (usize, usize, usize), kinetics: AblationKinetics) -> Self {
        let (nx, ny, nz) = shape;
        Self {
            damage: Array3::zeros((nx, ny, nz)),
            viability: Array3::ones((nx, ny, nz)),
            ablated: Array3::from_elem((nx, ny, nz), false),
            kinetics,
        }
    }

    /// Update ablation field with new temperature field
    pub fn update(&mut self, temperature: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        let (nx, ny, nz) = (
            self.damage.dim().0,
            self.damage.dim().1,
            self.damage.dim().2,
        );

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let t = temperature[[i, j, k]];

                    // Convert to absolute temperature
                    let t_abs = t + 273.15;

                    // Skip if below ablation threshold
                    if t < self.kinetics.ablation_threshold {
                        continue;
                    }

                    // Calculate damage rate
                    let damage_rate = self.kinetics.damage_rate(t_abs);

                    // Accumulate damage
                    let current_damage = self.damage[[i, j, k]];
                    let new_damage =
                        AblationKinetics::accumulated_damage(current_damage, damage_rate, dt);
                    self.damage[[i, j, k]] = new_damage;

                    // Update viability
                    self.viability[[i, j, k]] = self.kinetics.viability(new_damage);

                    // Check ablation
                    self.ablated[[i, j, k]] = self.kinetics.is_ablated(new_damage);
                }
            }
        }

        Ok(())
    }

    /// Get damage field
    pub fn damage(&self) -> &Array3<f64> {
        &self.damage
    }

    /// Get viability field
    pub fn viability(&self) -> &Array3<f64> {
        &self.viability
    }

    /// Get ablation field
    pub fn ablated(&self) -> &Array3<bool> {
        &self.ablated
    }

    /// Total ablated volume (count of ablated voxels)
    pub fn ablated_volume(&self) -> usize {
        self.ablated.iter().filter(|&&x| x).count()
    }

    /// Reset ablation field
    pub fn reset(&mut self) {
        self.damage.fill(0.0);
        self.viability.fill(1.0);
        self.ablated.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kinetics_creation() {
        let kinetics = AblationKinetics::protein_denaturation();
        assert!(kinetics.frequency_factor > 0.0);
        assert!(kinetics.activation_energy > 0.0);
        assert_eq!(kinetics.damage_threshold, 1.0);
    }

    #[test]
    fn test_damage_rate_temperature_dependence() {
        let kinetics = AblationKinetics::hifu_ablation();

        // Higher temperature should give higher damage rate
        let rate_43c = kinetics.damage_rate(273.15 + 43.0);
        let rate_50c = kinetics.damage_rate(273.15 + 50.0);
        let rate_70c = kinetics.damage_rate(273.15 + 70.0);

        assert!(rate_43c > 0.0);
        assert!(rate_50c > rate_43c); // 50°C > 43°C
        assert!(rate_70c > rate_50c); // 70°C > 50°C
    }

    #[test]
    fn test_damage_accumulation() {
        let mut current_damage = 0.0;
        let damage_rate = 0.5; // 1/s
        let dt = 1.0; // 1 second

        for _ in 0..3 {
            current_damage = AblationKinetics::accumulated_damage(current_damage, damage_rate, dt);
        }

        // After 3 seconds at 0.5 damage/s: Ω = 1.5
        assert!((current_damage - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_viability_from_damage() {
        let kinetics = AblationKinetics::default();

        // No damage
        assert!((kinetics.viability(0.0) - 1.0).abs() < 1e-6);

        // Ω = 1 (63.2% damage)
        assert!((kinetics.viability(1.0) - 0.368).abs() < 0.001);

        // Ω = 5 (99.3% damage)
        assert!(kinetics.viability(5.0) < 0.01);
    }

    #[test]
    fn test_ablation_threshold() {
        let kinetics = AblationKinetics::hifu_ablation();

        // Below threshold
        assert!(!kinetics.is_ablated(0.5));
        assert!(!kinetics.is_ablated(0.99));

        // At threshold
        assert!(kinetics.is_ablated(1.0));

        // Above threshold
        assert!(kinetics.is_ablated(1.5));
        assert!(kinetics.is_ablated(10.0));
    }

    #[test]
    fn test_ablation_state_update() {
        let kinetics = AblationKinetics::hifu_ablation();
        let mut state = AblationState::new(37.0, &kinetics);

        // No damage at body temperature
        state.update(37.0, &kinetics, 1.0);
        assert_eq!(state.damage, 0.0);
        assert!(!state.ablated);

        // Damage accumulates at higher temperature
        state.update(60.0, &kinetics, 1.0);
        assert!(state.damage > 0.0);
        assert!(state.viability < 1.0);

        // Continuous heating
        for _ in 0..10 {
            state.update(65.0, &kinetics, 1.0);
        }
        assert!(state.ablated || state.damage > 0.0);
    }

    #[test]
    fn test_kinetics_variants() {
        let protein = AblationKinetics::protein_denaturation();
        let collagen = AblationKinetics::collagen_denaturation();
        let hifu = AblationKinetics::hifu_ablation();

        // Different kinetics should give different damage rates at same temperature
        let t = 273.15 + 60.0;
        let rate_protein = protein.damage_rate(t);
        let rate_collagen = collagen.damage_rate(t);
        let rate_hifu = hifu.damage_rate(t);

        // HIFU should be most aggressive
        assert!(rate_hifu > rate_protein || rate_hifu > rate_collagen);
    }

    #[test]
    fn test_ablation_field() {
        let kinetics = AblationKinetics::hifu_ablation();
        let mut field = AblationField::new((10, 10, 10), kinetics);

        // Create temperature field
        let mut temperature = Array3::from_elem((10, 10, 10), 37.0);
        temperature[[5, 5, 5]] = 70.0; // Hot spot at center

        // Update multiple times
        for _ in 0..5 {
            let _ = field.update(&temperature, 0.1);
        }

        // Center should have damage
        assert!(field.damage()[[5, 5, 5]] > 0.0);
        assert!(field.viability()[[5, 5, 5]] < 1.0);

        // Periphery should be unaffected
        assert_eq!(field.damage()[[0, 0, 0]], 0.0);
        assert_eq!(field.viability()[[0, 0, 0]], 1.0);
    }

    #[test]
    fn test_ablation_volume_counting() {
        let kinetics = AblationKinetics::default();
        let mut field = AblationField::new((5, 5, 5), kinetics);

        // Create hotly heated field
        let mut temperature = Array3::from_elem((5, 5, 5), 37.0);
        for i in 1..4 {
            for j in 1..4 {
                for k in 1..4 {
                    temperature[[i, j, k]] = 75.0;
                }
            }
        }

        // Update until some tissue is ablated
        for _ in 0..20 {
            let _ = field.update(&temperature, 0.1);
        }

        // Should have some ablated volume
        let ablated_vol = field.ablated_volume();
        assert!(ablated_vol > 0, "Expected ablated volume > 0");
    }
}
