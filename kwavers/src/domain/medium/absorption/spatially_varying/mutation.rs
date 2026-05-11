use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

use super::{AbsorptionStatistics, SpatiallyVaryingAbsorption};

impl SpatiallyVaryingAbsorption {
    /// Update temperature.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn update_temperature(&mut self, temperature_field: Array3<f64>) -> KwaversResult<()> {
        if temperature_field.dim() != self.alpha_0_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Temperature field dimension mismatch".to_owned(),
            ));
        }
        self.temperature_field = Some(temperature_field);
        Ok(())
    }
    /// Set region.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn set_region(
        &mut self,
        i_range: std::ops::Range<usize>,
        j_range: std::ops::Range<usize>,
        k_range: std::ops::Range<usize>,
        alpha_0: f64,
        gamma: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = self.alpha_0_field.dim();

        if i_range.end > nx || j_range.end > ny || k_range.end > nz {
            return Err(KwaversError::InvalidInput(
                "Region out of bounds".to_owned(),
            ));
        }

        for i in i_range {
            for j in j_range.clone() {
                for k in k_range.clone() {
                    self.alpha_0_field[[i, j, k]] = alpha_0;
                    self.gamma_field[[i, j, k]] = gamma;
                }
            }
        }

        Ok(())
    }

    pub fn add_spherical_inclusion(
        &mut self,
        center: (f64, f64, f64),
        radius: f64,
        alpha_0: f64,
        gamma: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        let (nx, ny, nz) = self.alpha_0_field.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;

                    let dist =
                        (z - center.2).mul_add(z - center.2, (y - center.1).mul_add(y - center.1, (x - center.0).powi(2)))
                            .sqrt();

                    if dist <= radius {
                        self.alpha_0_field[[i, j, k]] = alpha_0;
                        self.gamma_field[[i, j, k]] = gamma;
                    }
                }
            }
        }
    }

    pub fn add_gaussian_transition(
        &mut self,
        center: (f64, f64, f64),
        sigma: f64,
        alpha_0_target: f64,
        gamma_target: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) {
        let (nx, ny, nz) = self.alpha_0_field.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;

                    let dist_sq =
                        (z - center.2).mul_add(z - center.2, (y - center.1).mul_add(y - center.1, (x - center.0).powi(2)));

                    let weight = (-dist_sq / (2.0 * sigma * sigma)).exp();

                    let current_alpha = self.alpha_0_field[[i, j, k]];
                    let current_gamma = self.gamma_field[[i, j, k]];

                    self.alpha_0_field[[i, j, k]] =
                        current_alpha.mul_add(1.0 - weight, alpha_0_target * weight);
                    self.gamma_field[[i, j, k]] =
                        current_gamma.mul_add(1.0 - weight, gamma_target * weight);
                }
            }
        }
    }
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self
            .alpha_0_field
            .iter()
            .any(|&a| a < 0.0 || !a.is_finite())
        {
            return Err(KwaversError::InvalidInput(
                "alpha_0 field contains non-physical values".to_owned(),
            ));
        }

        if self
            .gamma_field
            .iter()
            .any(|&g| !(0.0..=3.0).contains(&g) || !g.is_finite())
        {
            return Err(KwaversError::InvalidInput(
                "gamma field contains non-physical values".to_owned(),
            ));
        }

        Ok(())
    }

    pub fn statistics(&self) -> AbsorptionStatistics {
        let alpha_min = self
            .alpha_0_field
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let alpha_max = self
            .alpha_0_field
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let alpha_mean = self.alpha_0_field.mean().unwrap_or(0.0);

        let gamma_min = self
            .gamma_field
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let gamma_max = self
            .gamma_field
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let gamma_mean = self.gamma_field.mean().unwrap_or(0.0);

        AbsorptionStatistics {
            alpha_0_min: alpha_min,
            alpha_0_max: alpha_max,
            alpha_0_mean: alpha_mean,
            gamma_min,
            gamma_max,
            gamma_mean,
        }
    }
}
