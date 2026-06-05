use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::constants::REFERENCE_FREQUENCY_HZ;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

use super::SpatiallyVaryingAbsorption;

impl SpatiallyVaryingAbsorption {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        alpha_0_field: Array3<f64>,
        gamma_field: Array3<f64>,
        f_ref: f64,
    ) -> KwaversResult<Self> {
        if alpha_0_field.dim() != gamma_field.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "Field dimension mismatch: alpha_0 {:?} vs gamma {:?}",
                alpha_0_field.dim(),
                gamma_field.dim()
            )));
        }

        if alpha_0_field.iter().any(|&a| a < 0.0 || !a.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "alpha_0_field contains negative or non-finite values".to_owned(),
            ));
        }

        if gamma_field
            .iter()
            .any(|&g| !(0.0..=3.0).contains(&g) || !g.is_finite())
        {
            return Err(KwaversError::InvalidInput(
                "gamma_field contains invalid values (must be in [0, 3])".to_owned(),
            ));
        }

        if f_ref <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Reference frequency must be positive, got {}",
                f_ref
            )));
        }

        Ok(Self {
            alpha_0_field,
            gamma_field,
            f_ref,
            dispersion_correction: true,
            temperature_field: None,
            temperature_coefficient: 0.0,
            reference_temperature: BODY_TEMPERATURE_K,
        })
    }
    /// Uniform.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn uniform(
        nx: usize,
        ny: usize,
        nz: usize,
        alpha_0: f64,
        gamma: f64,
    ) -> KwaversResult<Self> {
        let alpha_0_field = Array3::from_elem((nx, ny, nz), alpha_0);
        let gamma_field = Array3::from_elem((nx, ny, nz), gamma);
        Self::new(alpha_0_field, gamma_field, REFERENCE_FREQUENCY_HZ)
    }
    /// With temperature dependence.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn with_temperature_dependence(
        mut self,
        temperature_field: Array3<f64>,
        coefficient: f64,
    ) -> KwaversResult<Self> {
        if temperature_field.dim() != self.alpha_0_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Temperature field dimension mismatch".to_owned(),
            ));
        }
        self.temperature_field = Some(temperature_field);
        self.temperature_coefficient = coefficient;
        Ok(self)
    }

    #[must_use]
    pub fn with_dispersion_correction(mut self, enabled: bool) -> Self {
        self.dispersion_correction = enabled;
        self
    }
}
