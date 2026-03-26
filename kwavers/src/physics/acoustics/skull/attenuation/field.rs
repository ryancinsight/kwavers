//! Full 3D attenuation field computation over numeric grids

use super::model::SkullAttenuation;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

impl SkullAttenuation {
    /// Compute attenuation field for skull with enhanced physics
    ///
    /// Returns attenuation factor (0-1) at each grid point, incorporating:
    /// - Frequency-dependent absorption
    /// - Rayleigh/geometric scattering
    /// - Optional temperature effects
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `skull_mask` - Binary mask (1 = bone, 0 = soft tissue)
    /// * `frequency` - Acoustic frequency (Hz)
    ///
    /// # Returns
    ///
    /// Attenuation factor array (exp(-α·d)) where d is path length
    pub fn compute_attenuation_field(
        &self,
        grid: &Grid,
        skull_mask: &Array3<f64>,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        self.compute_attenuation_field_with_temperature(grid, skull_mask, frequency, None)
    }

    /// Compute attenuation field with temperature dependence
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `skull_mask` - Binary mask (1 = bone, 0 = soft tissue)
    /// * `frequency` - Acoustic frequency (Hz)
    /// * `temperature` - Optional temperature field (°C)
    ///
    /// # Returns
    ///
    /// Attenuation factor array incorporating all physics
    pub fn compute_attenuation_field_with_temperature(
        &self,
        grid: &Grid,
        skull_mask: &Array3<f64>,
        frequency: f64,
        temperature: Option<&Array3<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        // Compute total attenuation coefficient
        let alpha_total = self.total_coefficient(frequency);

        let mut attenuation = Array3::ones(skull_mask.dim());

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if skull_mask[[i, j, k]] > 0.5 {
                        // Path length through skull bone layer
                        let path_length = grid.dx;

                        // Temperature correction if provided
                        let temp_factor = if let Some(temp_field) = temperature {
                            self.temperature_correction(temp_field[[i, j, k]])
                        } else {
                            1.0
                        };

                        // Total attenuation: exp(-α·d·T_correction)
                        attenuation[[i, j, k]] = (-alpha_total * path_length * temp_factor).exp();
                    }
                }
            }
        }

        Ok(attenuation)
    }

    /// Convert attenuation coefficient to dB/cm for clinical use
    ///
    /// Conversion: α_dB = α_Np · 8.686 (Np → dB conversion)
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency (Hz)
    ///
    /// # Returns
    ///
    /// Attenuation in dB/cm
    #[must_use]
    pub fn attenuation_db_per_cm(&self, frequency: f64) -> f64 {
        let alpha_np_per_m = self.total_coefficient(frequency);
        let np_to_db = 8.686;
        let m_to_cm = 0.01;
        alpha_np_per_m * np_to_db * m_to_cm
    }
}
