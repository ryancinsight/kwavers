//! Spatially-varying power law absorption model
//!
//! This module implements heterogeneous absorption where both the absorption
//! coefficient α₀ and power law exponent γ (or y) can vary spatially across
//! the computational domain.
//!
//! # Theory
//!
//! In realistic biological media, tissue properties vary continuously in space.
//! The absorption coefficient follows a power law:
//!
//! ```text
//! α(x, y, z, f) = α₀(x, y, z) · (f / f_ref)^γ(x, y, z)
//! ```
//!
//! where:
//! - α₀(x, y, z): spatially-varying absorption coefficient at reference frequency
//! - γ(x, y, z): spatially-varying power law exponent
//! - f: frequency [Hz]
//! - f_ref: reference frequency (typically 1 MHz)
//!
//! This model is essential for:
//! - Realistic tissue heterogeneity (Fullwave 2.5 model)
//! - Tumor/lesion modeling with different absorption properties
//! - Multi-tissue interfaces (e.g., fat-muscle-bone)
//! - Gradient tissues (e.g., skin layers, vessel walls)
//!
//! # References
//!
//! - Pinton et al. (2009): "A heterogeneous nonlinear attenuating full-wave
//!   model of ultrasound," IEEE Trans. UFFC, 56(3), 474-488.
//! - Fullwave 2.5: MATLAB ultrasound simulation toolbox
//! - Treeby & Cox (2010): "Modeling power law absorption and dispersion"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array3, Zip};
use num_complex::Complex;

/// Spatially-varying power law absorption model
///
/// Stores 3D fields for absorption coefficient α₀ and power law exponent γ,
/// allowing realistic heterogeneous tissue modeling.
#[derive(Debug, Clone)]
pub struct SpatiallyVaryingAbsorption {
    /// Absorption coefficient field α₀(x,y,z) at reference frequency [Np/m]
    /// Units: Nepers per meter at f_ref
    alpha_0_field: Array3<f64>,

    /// Power law exponent field γ(x,y,z) [dimensionless]
    /// Typical range: 1.0 (linear) to 2.0 (quadratic)
    /// Most soft tissues: γ ∈ [1.05, 1.5]
    gamma_field: Array3<f64>,

    /// Reference frequency [Hz]
    f_ref: f64,

    /// Enable Kramers-Kronig dispersion correction
    dispersion_correction: bool,

    /// Temperature field for temperature-dependent absorption (optional)
    temperature_field: Option<Array3<f64>>,

    /// Temperature coefficient β [1/K] for α(T) = α₀(1 + β(T - T₀))
    temperature_coefficient: f64,

    /// Reference temperature T₀ [K]
    reference_temperature: f64,
}

impl SpatiallyVaryingAbsorption {
    /// Create a new spatially-varying absorption model
    ///
    /// # Arguments
    ///
    /// * `alpha_0_field` - 3D array of absorption coefficients at reference frequency
    /// * `gamma_field` - 3D array of power law exponents
    /// * `f_ref` - Reference frequency in Hz (typically 1 MHz = 1e6)
    ///
    /// # Errors
    ///
    /// Returns error if field dimensions don't match or contain invalid values
    pub fn new(
        alpha_0_field: Array3<f64>,
        gamma_field: Array3<f64>,
        f_ref: f64,
    ) -> KwaversResult<Self> {
        // Validate dimensions match
        if alpha_0_field.dim() != gamma_field.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "Field dimension mismatch: alpha_0 {:?} vs gamma {:?}",
                alpha_0_field.dim(),
                gamma_field.dim()
            )));
        }

        // Validate physical values
        if alpha_0_field.iter().any(|&a| a < 0.0 || !a.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "alpha_0_field contains negative or non-finite values".to_string(),
            ));
        }

        if gamma_field
            .iter()
            .any(|&g| !(0.0..=3.0).contains(&g) || !g.is_finite())
        {
            return Err(KwaversError::InvalidInput(
                "gamma_field contains invalid values (must be in [0, 3])".to_string(),
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
            reference_temperature: 310.15, // 37°C
        })
    }

    /// Create from uniform values (for testing or homogeneous media)
    pub fn uniform(
        nx: usize,
        ny: usize,
        nz: usize,
        alpha_0: f64,
        gamma: f64,
    ) -> KwaversResult<Self> {
        let alpha_0_field = Array3::from_elem((nx, ny, nz), alpha_0);
        let gamma_field = Array3::from_elem((nx, ny, nz), gamma);
        Self::new(alpha_0_field, gamma_field, 1e6)
    }

    /// Enable temperature-dependent absorption
    pub fn with_temperature_dependence(
        mut self,
        temperature_field: Array3<f64>,
        coefficient: f64,
    ) -> KwaversResult<Self> {
        if temperature_field.dim() != self.alpha_0_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Temperature field dimension mismatch".to_string(),
            ));
        }
        self.temperature_field = Some(temperature_field);
        self.temperature_coefficient = coefficient;
        Ok(self)
    }

    /// Set dispersion correction flag
    pub fn with_dispersion_correction(mut self, enabled: bool) -> Self {
        self.dispersion_correction = enabled;
        self
    }

    /// Get absorption coefficient at a specific point and frequency
    ///
    /// # Arguments
    ///
    /// * `i, j, k` - Grid indices
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Absorption coefficient α(x, y, z, f) in Np/m
    #[must_use]
    pub fn absorption_at_point(&self, i: usize, j: usize, k: usize, frequency: f64) -> f64 {
        let alpha_0 = self.alpha_0_field[[i, j, k]];
        let gamma = self.gamma_field[[i, j, k]];

        // Power law: α(f) = α₀ * (f / f_ref)^γ
        let freq_ratio = frequency / self.f_ref;
        let mut alpha = alpha_0 * freq_ratio.powf(gamma);

        // Apply temperature correction if available
        if let Some(ref temp_field) = self.temperature_field {
            let temp = temp_field[[i, j, k]];
            let delta_t = temp - self.reference_temperature;
            alpha *= 1.0 + self.temperature_coefficient * delta_t;
        }

        alpha
    }

    /// Compute absorption field for entire domain at a given frequency
    pub fn compute_absorption_field(&self, frequency: f64) -> Array3<f64> {
        let (nx, ny, nz) = self.alpha_0_field.dim();
        let mut alpha_field = Array3::zeros((nx, ny, nz));

        let freq_ratio = frequency / self.f_ref;

        // Vectorized computation
        Zip::from(&mut alpha_field)
            .and(&self.alpha_0_field)
            .and(&self.gamma_field)
            .for_each(|alpha, &alpha_0, &gamma| {
                *alpha = alpha_0 * freq_ratio.powf(gamma);
            });

        // Apply temperature correction if available
        if let Some(ref temp_field) = self.temperature_field {
            Zip::from(&mut alpha_field)
                .and(temp_field)
                .for_each(|alpha, &temp| {
                    let delta_t = temp - self.reference_temperature;
                    *alpha *= 1.0 + self.temperature_coefficient * delta_t;
                });
        }

        alpha_field
    }

    /// Apply absorption to frequency-domain field
    ///
    /// Applies spatially-varying attenuation to a 3D complex field representing
    /// the frequency-domain representation of the acoustic field.
    ///
    /// # Arguments
    ///
    /// * `field` - Complex frequency-domain field to attenuate (modified in-place)
    /// * `frequency` - Frequency of this field component [Hz]
    /// * `dx` - Grid spacing in x [m]
    ///
    /// The attenuation is applied as: `field *= exp(-α * dx)`
    pub fn apply_frequency_domain(
        &self,
        field: &mut Array3<Complex<f64>>,
        frequency: f64,
        dx: f64,
    ) -> KwaversResult<()> {
        if field.dim() != self.alpha_0_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Field dimension mismatch".to_string(),
            ));
        }

        let alpha_field = self.compute_absorption_field(frequency);

        // Apply attenuation: field(x) *= exp(-α(x) * dx)
        Zip::from(field).and(&alpha_field).for_each(|f, &alpha| {
            let attenuation = (-alpha * dx).exp();
            *f *= attenuation;
        });

        Ok(())
    }

    /// Apply absorption along a specific direction (for directional splitting schemes)
    ///
    /// Used in operator splitting methods where absorption is applied separately
    /// along each spatial direction.
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

        Zip::from(field).and(&alpha_field).for_each(|f, &alpha| {
            // Fractional absorption for directional step
            let attenuation = (-alpha * ds / 3.0_f64.sqrt()).exp();
            *f *= attenuation;
        });

        Ok(())
    }

    /// Compute phase velocity field using Kramers-Kronig relations
    ///
    /// For a power-law absorbing medium, the dispersion relation relates
    /// absorption to phase velocity via Kramers-Kronig integrals.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    /// * `c0_field` - 3D array of baseline sound speeds [m/s]
    ///
    /// # Returns
    ///
    /// Phase velocity field c(x, y, z, f) [m/s]
    pub fn phase_velocity_field(
        &self,
        frequency: f64,
        c0_field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if !self.dispersion_correction {
            return Ok(c0_field.clone());
        }

        if c0_field.dim() != self.alpha_0_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Sound speed field dimension mismatch".to_string(),
            ));
        }

        let (nx, ny, nz) = c0_field.dim();
        let mut c_field = Array3::zeros((nx, ny, nz));

        let omega = 2.0 * std::f64::consts::PI * frequency;

        // Kramers-Kronig for power law: c(ω) = c₀ / (1 + α₀ * tan(πγ/2) * ω^(γ-1))
        Zip::from(&mut c_field)
            .and(c0_field)
            .and(&self.alpha_0_field)
            .and(&self.gamma_field)
            .for_each(|c, &c0, &alpha_0, &gamma| {
                let tan_term = (std::f64::consts::PI * gamma / 2.0).tan();
                let dispersion_factor = 1.0 + alpha_0 * tan_term * omega.powf(gamma - 1.0);
                *c = c0 / dispersion_factor;
            });

        Ok(c_field)
    }

    /// Get reference to alpha_0 field
    #[must_use]
    pub fn alpha_0_field(&self) -> &Array3<f64> {
        &self.alpha_0_field
    }

    /// Get reference to gamma field
    #[must_use]
    pub fn gamma_field(&self) -> &Array3<f64> {
        &self.gamma_field
    }

    /// Get mutable reference to alpha_0 field
    pub fn alpha_0_field_mut(&mut self) -> &mut Array3<f64> {
        &mut self.alpha_0_field
    }

    /// Get mutable reference to gamma field
    pub fn gamma_field_mut(&mut self) -> &mut Array3<f64> {
        &mut self.gamma_field
    }

    /// Update temperature field
    pub fn update_temperature(&mut self, temperature_field: Array3<f64>) -> KwaversResult<()> {
        if temperature_field.dim() != self.alpha_0_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Temperature field dimension mismatch".to_string(),
            ));
        }
        self.temperature_field = Some(temperature_field);
        Ok(())
    }

    /// Set a region to specific absorption properties
    ///
    /// Useful for creating tumors, lesions, or other localized features
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
                "Region out of bounds".to_string(),
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

    /// Create a spherical inclusion with different absorption properties
    ///
    /// Useful for modeling tumors, contrast agents, or lesions
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
                        ((x - center.0).powi(2) + (y - center.1).powi(2) + (z - center.2).powi(2))
                            .sqrt();

                    if dist <= radius {
                        self.alpha_0_field[[i, j, k]] = alpha_0;
                        self.gamma_field[[i, j, k]] = gamma;
                    }
                }
            }
        }
    }

    /// Apply smooth Gaussian transition between two absorption regions
    ///
    /// Creates a gradual transition useful for realistic tissue interfaces
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
                        (x - center.0).powi(2) + (y - center.1).powi(2) + (z - center.2).powi(2);

                    let weight = (-dist_sq / (2.0 * sigma * sigma)).exp();

                    let current_alpha = self.alpha_0_field[[i, j, k]];
                    let current_gamma = self.gamma_field[[i, j, k]];

                    self.alpha_0_field[[i, j, k]] =
                        current_alpha * (1.0 - weight) + alpha_0_target * weight;
                    self.gamma_field[[i, j, k]] =
                        current_gamma * (1.0 - weight) + gamma_target * weight;
                }
            }
        }
    }

    /// Validate that all field values are physical
    pub fn validate(&self) -> KwaversResult<()> {
        if self
            .alpha_0_field
            .iter()
            .any(|&a| a < 0.0 || !a.is_finite())
        {
            return Err(KwaversError::InvalidInput(
                "alpha_0 field contains non-physical values".to_string(),
            ));
        }

        if self
            .gamma_field
            .iter()
            .any(|&g| !(0.0..=3.0).contains(&g) || !g.is_finite())
        {
            return Err(KwaversError::InvalidInput(
                "gamma field contains non-physical values".to_string(),
            ));
        }

        Ok(())
    }

    /// Compute statistics of the absorption fields
    pub fn statistics(&self) -> AbsorptionStatistics {
        let alpha_min = self
            .alpha_0_field
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let alpha_max = self
            .alpha_0_field
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let alpha_mean = self.alpha_0_field.mean().unwrap_or(0.0);

        let gamma_min = self
            .gamma_field
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let gamma_max = self
            .gamma_field
            .iter()
            .cloned()
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

/// Statistics of absorption field properties
#[derive(Debug, Clone, Copy)]
pub struct AbsorptionStatistics {
    pub alpha_0_min: f64,
    pub alpha_0_max: f64,
    pub alpha_0_mean: f64,
    pub gamma_min: f64,
    pub gamma_max: f64,
    pub gamma_mean: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_absorption() -> KwaversResult<()> {
        let absorption = SpatiallyVaryingAbsorption::uniform(10, 10, 10, 0.5, 1.1)?;

        let alpha = absorption.absorption_at_point(5, 5, 5, 1e6);
        assert!((alpha - 0.5).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_frequency_dependence() -> KwaversResult<()> {
        let absorption = SpatiallyVaryingAbsorption::uniform(5, 5, 5, 1.0, 1.5)?;

        let alpha_1mhz = absorption.absorption_at_point(0, 0, 0, 1e6);
        let alpha_2mhz = absorption.absorption_at_point(0, 0, 0, 2e6);

        // α(2f) = α₀ * 2^1.5 ≈ α₀ * 2.828
        let expected_ratio = 2.0_f64.powf(1.5);
        let actual_ratio = alpha_2mhz / alpha_1mhz;

        assert!((actual_ratio - expected_ratio).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_spherical_inclusion() -> KwaversResult<()> {
        let mut absorption = SpatiallyVaryingAbsorption::uniform(20, 20, 20, 0.5, 1.0)?;

        absorption.add_spherical_inclusion((0.5, 0.5, 0.5), 0.3, 2.0, 1.5, 0.1, 0.1, 0.1);

        // Center should have inclusion properties
        let alpha_center = absorption.absorption_at_point(5, 5, 5, 1e6);
        assert!((alpha_center - 2.0).abs() < 1e-10);

        // Far away should have background properties
        let alpha_far = absorption.absorption_at_point(15, 15, 15, 1e6);
        assert!((alpha_far - 0.5).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_compute_absorption_field() -> KwaversResult<()> {
        let absorption = SpatiallyVaryingAbsorption::uniform(8, 8, 8, 0.75, 1.1)?;

        let field = absorption.compute_absorption_field(2e6);

        // All values should be α₀ * 2^1.1
        let expected = 0.75 * 2.0_f64.powf(1.1);
        for &val in field.iter() {
            assert!((val - expected).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_temperature_dependence() -> KwaversResult<()> {
        let absorption = SpatiallyVaryingAbsorption::uniform(5, 5, 5, 1.0, 1.0)?;

        let mut temp_field = Array3::from_elem((5, 5, 5), 310.15); // Reference temp
        temp_field[[2, 2, 2]] = 320.15; // 10K higher

        let absorption = absorption.with_temperature_dependence(temp_field, 0.01)?; // 1% per K

        let alpha_ref = absorption.absorption_at_point(0, 0, 0, 1e6);
        let alpha_hot = absorption.absorption_at_point(2, 2, 2, 1e6);

        // Hot spot should have 10% higher absorption
        let expected_ratio = 1.1;
        let actual_ratio = alpha_hot / alpha_ref;

        assert!((actual_ratio - expected_ratio).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_validation() -> KwaversResult<()> {
        let good = SpatiallyVaryingAbsorption::uniform(3, 3, 3, 0.5, 1.2)?;
        assert!(good.validate().is_ok());

        // Negative alpha_0 should fail
        let mut bad_alpha = Array3::from_elem((3, 3, 3), 0.5);
        bad_alpha[[1, 1, 1]] = -0.1;
        let bad =
            SpatiallyVaryingAbsorption::new(bad_alpha, Array3::from_elem((3, 3, 3), 1.0), 1e6);
        assert!(bad.is_err());

        Ok(())
    }

    #[test]
    fn test_statistics() -> KwaversResult<()> {
        let mut absorption = SpatiallyVaryingAbsorption::uniform(10, 10, 10, 1.0, 1.2)?;
        absorption.set_region(0..5, 0..5, 0..5, 0.5, 1.0)?;
        absorption.set_region(5..10, 5..10, 5..10, 2.0, 1.5)?;

        let stats = absorption.statistics();

        assert!((stats.alpha_0_min - 0.5).abs() < 1e-10);
        assert!((stats.alpha_0_max - 2.0).abs() < 1e-10);
        assert!((stats.gamma_min - 1.0).abs() < 1e-10);
        assert!((stats.gamma_max - 1.5).abs() < 1e-10);

        Ok(())
    }
}
