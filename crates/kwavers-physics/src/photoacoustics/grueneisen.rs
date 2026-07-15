//! Temperature-dependent Gruneisen conversion model.

use kwavers_core::constants::thermodynamic::{
    BODY_TEMPERATURE_C, GRUNEISEN_SOFT_TISSUE, GRUNEISEN_SOFT_TISSUE_TEMP_COEFF,
    GRUNEISEN_WATER_20C, GRUNEISEN_WATER_TEMP_COEFF, GRUNEISEN_WATER_T_REF_C,
};

/// Temperature-dependent Gruneisen parameter Gamma(T) = Gamma_0 + c_T * (T - T_ref).
///
/// This pure material model is available without the `clinical-imaging` feature:
/// electromagnetic photoacoustic simulation needs thermoelastic conversion, but
/// not CT, registration, or image-I/O types.
///
/// ## References
///
/// - Sigrist MW (1986). "Laser generation of acoustic waves in liquids and
///   gases." *J Appl Phys* **60**(7), R83. DOI: 10.1063/1.337089
/// - Xu M, Wang LV (2006). "Photoacoustic imaging in biomedicine." *Rev Sci
///   Instrum* **77**, 041101. DOI: 10.1063/1.2195024
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrueneisenModel {
    /// Baseline Gruneisen parameter Gamma_0 (dimensionless, > 0).
    pub gamma_0: f64,
    /// Temperature coefficient c_T = dGamma/dT [K^-1]; `None` is temperature-independent.
    pub d_gamma_d_t: Option<f64>,
    /// Reference temperature T_ref [degrees C] at which Gamma_0 applies.
    pub t_ref_celsius: f64,
}

impl GrueneisenModel {
    /// Constant Gruneisen parameter (temperature-independent).
    #[must_use]
    pub fn constant(gamma_0: f64) -> Self {
        Self {
            gamma_0,
            d_gamma_d_t: None,
            t_ref_celsius: 0.0,
        }
    }

    /// Gruneisen model with linear temperature dependence Gamma(T) = Gamma_0 + c_T * (T - T_ref).
    #[must_use]
    pub fn with_temperature_coefficient(gamma_0: f64, c_t: f64, t_ref: f64) -> Self {
        Self {
            gamma_0,
            d_gamma_d_t: Some(c_t),
            t_ref_celsius: t_ref,
        }
    }

    /// Water model at the published 20 degrees C reference state.
    #[must_use]
    pub fn water() -> Self {
        Self::with_temperature_coefficient(
            GRUNEISEN_WATER_20C,
            GRUNEISEN_WATER_TEMP_COEFF,
            GRUNEISEN_WATER_T_REF_C,
        )
    }

    /// Soft-tissue model at the 37 degrees C reference state.
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self::with_temperature_coefficient(
            GRUNEISEN_SOFT_TISSUE,
            GRUNEISEN_SOFT_TISSUE_TEMP_COEFF,
            BODY_TEMPERATURE_C,
        )
    }

    /// Evaluate Gamma at the given temperature in degrees C.
    #[must_use]
    pub fn evaluate(&self, t_celsius: f64) -> f64 {
        match self.d_gamma_d_t {
            None => self.gamma_0,
            Some(c_t) => c_t.mul_add(t_celsius - self.t_ref_celsius, self.gamma_0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;

    #[test]
    fn water_at_body_temperature_matches_the_published_linear_model() {
        let actual = GrueneisenModel::water().evaluate(BODY_TEMPERATURE_C);
        assert!(
            (actual - 0.188).abs() <= 1.0e-12,
            "expected 0.188, got {actual}"
        );
    }

    #[test]
    fn temperature_independent_model_preserves_its_baseline() {
        let model = GrueneisenModel::constant(0.2);
        assert_eq!(model.evaluate(0.0), 0.2);
        assert_eq!(model.evaluate(100.0), 0.2);
        assert_eq!(model.evaluate(-KELVIN_OFFSET_C), 0.2);
    }

    #[test]
    fn soft_tissue_reference_temperature_preserves_its_baseline() {
        let actual = GrueneisenModel::soft_tissue().evaluate(BODY_TEMPERATURE_C);
        assert!((actual - GRUNEISEN_SOFT_TISSUE).abs() <= 1.0e-12);
    }

    #[test]
    fn soft_tissue_temperature_response_matches_the_linear_law() {
        let model = GrueneisenModel::soft_tissue();
        let at_20_c = model.evaluate(20.0);
        let at_37_c = model.evaluate(BODY_TEMPERATURE_C);
        let expected_at_20_c =
            GRUNEISEN_SOFT_TISSUE + GRUNEISEN_SOFT_TISSUE_TEMP_COEFF * (20.0 - BODY_TEMPERATURE_C);

        assert!((at_20_c - expected_at_20_c).abs() <= 1.0e-12);
        assert!((at_37_c - GRUNEISEN_SOFT_TISSUE).abs() <= 1.0e-12);
        assert!(
            ((at_37_c / at_20_c) - (GRUNEISEN_SOFT_TISSUE / expected_at_20_c)).abs() <= 1.0e-10
        );
    }
}
