// physics/mechanics/viscosity/shear_viscosity.rs
use crate::physics::acoustics::mechanics::viscosity::ViscosityModel;
use log::debug;

#[derive(Debug)]
pub struct ShearViscosityModel {
    base_viscosity: f64,   // Base viscosity at reference temperature (Pa·s)
    ref_temperature: f64,  // Reference temperature (K)
    temp_sensitivity: f64, // Temperature sensitivity factor (1/K)
}

impl ShearViscosityModel {
    #[must_use]
    pub fn new(base_viscosity: f64, ref_temperature: f64, temp_sensitivity: f64) -> Self {
        debug!(
            "Initializing ShearViscosityModel: base = {:.6e} Pa·s, ref_temperature = {:.2} K, sensitivity = {:.6e} 1/K",
            base_viscosity, ref_temperature, temp_sensitivity
        );
        Self {
            base_viscosity,
            ref_temperature,
            temp_sensitivity,
        }
    }
}

impl ViscosityModel for ShearViscosityModel {
    fn viscosity(&self, x: f64, y: f64, z: f64, temperature: f64) -> f64 {
        // Temperature-dependent viscosity with potential for spatial variation
        // Currently uniform in space, but position parameters preserved for
        // future heterogeneous media support
        let _ = (x, y, z); // Acknowledge spatial parameters for future use
        let delta_t = temperature - self.ref_temperature;
        self.base_viscosity * (-self.temp_sensitivity * delta_t).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the reference temperature, the Arrhenius exponential is exp(0) = 1,
    /// so viscosity equals the base viscosity exactly.
    #[test]
    fn viscosity_at_reference_temperature_equals_base_viscosity() {
        let eta_0 = 1e-3_f64; // water at 20°C (Pa·s)
        let t_ref = 293.15_f64; // 20°C in K
        let s = 0.02_f64; // 1/K
        let model = ShearViscosityModel::new(eta_0, t_ref, s);

        let eta = model.viscosity(0.0, 0.0, 0.0, t_ref);
        assert!(
            (eta - eta_0).abs() < f64::EPSILON * eta_0,
            "η(T_ref) must equal η_0 = {eta_0:.3e}, got {eta:.3e}"
        );
    }

    /// For positive sensitivity, viscosity decreases as temperature rises above
    /// the reference (Arrhenius model for liquids: η ∝ exp(−s·ΔT)).
    #[test]
    fn viscosity_decreases_with_temperature_for_positive_sensitivity() {
        let model = ShearViscosityModel::new(1e-3, 293.15, 0.02);
        let eta_cold = model.viscosity(0.0, 0.0, 0.0, 283.15); // 10 K below T_ref
        let eta_hot = model.viscosity(0.0, 0.0, 0.0, 303.15); // 10 K above T_ref
        assert!(
            eta_cold > eta_hot,
            "viscosity must decrease with temperature (cold={eta_cold:.3e} > hot={eta_hot:.3e})"
        );
    }

    /// Analytical formula: η(T) = η_0 · exp(−s · (T − T_ref)).
    ///
    /// For s=0.02, ΔT=10: η = η_0 · exp(−0.2) ≈ η_0 · 0.8187.
    #[test]
    fn viscosity_matches_arrhenius_formula_analytically() {
        let eta_0 = 1e-3_f64;
        let t_ref = 293.15_f64;
        let s = 0.02_f64;
        let model = ShearViscosityModel::new(eta_0, t_ref, s);

        let delta_t = 10.0_f64;
        let expected = eta_0 * (-s * delta_t).exp();
        let got = model.viscosity(1.0, 2.0, 3.0, t_ref + delta_t);
        assert!(
            (got - expected).abs() < expected * 1e-12,
            "η(T_ref+10): got {got:.6e} expected {expected:.6e}"
        );
    }

    /// The model is spatially uniform: identical temperatures at different
    /// positions produce the same viscosity value.
    #[test]
    fn viscosity_is_spatially_uniform_for_same_temperature() {
        let model = ShearViscosityModel::new(1e-3, 293.15, 0.02);
        let t = 300.0_f64;
        let eta1 = model.viscosity(0.0, 0.0, 0.0, t);
        let eta2 = model.viscosity(0.1, 0.5, 0.9, t);
        let eta3 = model.viscosity(100.0, -50.0, 12.5, t);
        assert_eq!(eta1, eta2, "viscosity must be position-independent");
        assert_eq!(eta1, eta3, "viscosity must be position-independent");
    }

    /// For negative sensitivity, viscosity increases with temperature
    /// (models with anomalous temperature dependence or gels).
    #[test]
    fn viscosity_increases_with_temperature_for_negative_sensitivity() {
        let model = ShearViscosityModel::new(1e-3, 293.15, -0.02);
        let eta_low = model.viscosity(0.0, 0.0, 0.0, 283.15);
        let eta_high = model.viscosity(0.0, 0.0, 0.0, 303.15);
        assert!(
            eta_high > eta_low,
            "negative sensitivity → viscosity increases with T (low={eta_low:.3e} < high={eta_high:.3e})"
        );
    }
}
