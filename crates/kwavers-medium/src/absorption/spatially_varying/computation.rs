use eunomia::Complex;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

use crate::parallel::{zip_mut_ref, zip_mut_three_refs, zip_mut_two_refs};

use super::SpatiallyVaryingAbsorption;
use kwavers_core::constants::numerical::TWO_PI;

impl SpatiallyVaryingAbsorption {
    #[must_use]
    pub fn absorption_at_point(&self, i: usize, j: usize, k: usize, frequency: f64) -> f64 {
        let alpha_0 = self.alpha_0_field[[i, j, k]];
        let gamma = self.gamma_field[[i, j, k]];

        let freq_ratio = frequency / self.f_ref;
        let mut alpha = alpha_0 * freq_ratio.powf(gamma);

        if let Some(ref temp_field) = self.temperature_field {
            let temp = temp_field[[i, j, k]];
            let delta_t = temp - self.reference_temperature;
            alpha *= self.temperature_coefficient.mul_add(delta_t, 1.0);
        }

        alpha
    }

    #[must_use]
    pub fn compute_absorption_field(&self, frequency: f64) -> Array3<f64> {
        let [nx, ny, nz] = self.alpha_0_field.shape();
        let mut alpha_field = Array3::zeros([nx, ny, nz]);

        let freq_ratio = frequency / self.f_ref;

        zip_mut_two_refs(
            &mut alpha_field,
            &self.alpha_0_field,
            &self.gamma_field,
            |alpha, &alpha_0, &gamma| {
                *alpha = alpha_0 * freq_ratio.powf(gamma);
            },
        );

        if let Some(ref temp_field) = self.temperature_field {
            let ref_temp = self.reference_temperature;
            let temp_coeff = self.temperature_coefficient;
            zip_mut_ref(&mut alpha_field, temp_field, |alpha, &temp| {
                let delta_t = temp - ref_temp;
                *alpha *= temp_coeff.mul_add(delta_t, 1.0);
            });
        }

        alpha_field
    }
    /// Apply frequency domain.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn apply_frequency_domain(
        &self,
        field: &mut Array3<Complex<f64>>,
        frequency: f64,
        dx: f64,
    ) -> KwaversResult<()> {
        if field.shape() != self.alpha_0_field.shape() {
            return Err(KwaversError::InvalidInput(
                "Field dimension mismatch".to_owned(),
            ));
        }

        let alpha_field = self.compute_absorption_field(frequency);

        zip_mut_ref(field, &alpha_field, |f, &alpha| {
            let attenuation = (-alpha * dx).exp();
            *f *= attenuation;
        });

        Ok(())
    }
    /// Apply directional.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn apply_directional(
        &self,
        field: &mut Array3<Complex<f64>>,
        frequency: f64,
        ds: f64,
        axis: usize,
    ) -> KwaversResult<()> {
        if axis > 2 {
            return Err(KwaversError::InvalidInput(format!(
                "Invalid axis {}, must be 0, 1, or 2",
                axis
            )));
        }

        let alpha_field = self.compute_absorption_field(frequency);

        zip_mut_ref(field, &alpha_field, |f, &alpha| {
            let attenuation = (-alpha * ds / 3.0_f64.sqrt()).exp();
            *f *= attenuation;
        });

        Ok(())
    }
    /// Phase velocity field.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn phase_velocity_field(
        &self,
        frequency: f64,
        c0_field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if !self.dispersion_correction {
            return Ok(c0_field.clone());
        }

        if c0_field.shape() != self.alpha_0_field.shape() {
            return Err(KwaversError::InvalidInput(
                "Sound speed field dimension mismatch".to_owned(),
            ));
        }

        let [nx, ny, nz] = c0_field.shape();
        let mut c_field = Array3::zeros([nx, ny, nz]);

        let omega = TWO_PI * frequency;

        zip_mut_three_refs(
            &mut c_field,
            c0_field,
            &self.alpha_0_field,
            &self.gamma_field,
            |c, &c0, &alpha_0, &gamma| {
                let tan_term = (std::f64::consts::PI * gamma / 2.0).tan();
                let dispersion_factor = (alpha_0 * tan_term).mul_add(omega.powf(gamma - 1.0), 1.0);
                *c = c0 / dispersion_factor;
            },
        );

        Ok(c_field)
    }
}
