//! Blackbody radiation model for sonoluminescence
//!
//! Implements Planck's law for thermal radiation from hot bubble interior

use ndarray::{Array1, Array3};
use std::f64::consts::PI;

/// Physical constants
pub const PLANCK_CONSTANT: f64 = 6.62607015e-34; // J·s
pub const SPEED_OF_LIGHT: f64 = 2.99792458e8; // m/s
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W/(m²·K⁴)

/// Blackbody radiation model
#[derive(Debug, Clone)]
pub struct BlackbodyModel {
    /// Emissivity factor (0-1)
    pub emissivity: f64,
    /// Optical thickness correction
    pub optical_depth: f64,
}

impl Default for BlackbodyModel {
    fn default() -> Self {
        Self {
            emissivity: 0.1,    // Partial emissivity for bubble
            optical_depth: 0.1, // Optically thin approximation
        }
    }
}

impl BlackbodyModel {
    /// Calculate spectral radiance using Planck's law
    ///
    /// # Arguments
    /// * `wavelength` - Wavelength in meters
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Spectral radiance in W/(m²·sr·m)
    #[must_use]
    pub fn spectral_radiance(&self, wavelength: f64, temperature: f64) -> f64 {
        if temperature <= 0.0 || wavelength <= 0.0 {
            return 0.0;
        }

        let hc_over_lambda = PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelength;
        let hc_over_lambda_kt = hc_over_lambda / (BOLTZMANN_CONSTANT * temperature);

        // Use appropriate approximation based on x = hc/(λkT)
        let planck = if hc_over_lambda_kt > 50.0 {
            // Wien approximation for large x
            2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / wavelength.powi(5)
                * (-hc_over_lambda_kt).exp()
        } else if hc_over_lambda_kt < 0.01 {
            // Rayleigh-Jeans approximation for small x
            2.0 * SPEED_OF_LIGHT * BOLTZMANN_CONSTANT * temperature / wavelength.powi(4)
        } else {
            // Full Planck formula
            2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2)
                / wavelength.powi(5)
                / (hc_over_lambda_kt.exp() - 1.0)
        };

        // Apply emissivity and optical depth correction
        self.emissivity * planck * (1.0 - (-self.optical_depth).exp())
    }

    /// Calculate total radiated power using Stefan-Boltzmann law
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `surface_area` - Emitting surface area in m²
    ///
    /// # Returns
    /// Total radiated power in Watts
    #[must_use]
    pub fn total_power(&self, temperature: f64, surface_area: f64) -> f64 {
        self.emissivity * STEFAN_BOLTZMANN * surface_area * temperature.powi(4)
    }

    /// Calculate emission spectrum over wavelength range
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `wavelengths` - Array of wavelengths in meters
    ///
    /// # Returns
    /// Array of spectral radiances
    #[must_use]
    pub fn emission_spectrum(&self, temperature: f64, wavelengths: &Array1<f64>) -> Array1<f64> {
        wavelengths.mapv(|lambda| self.spectral_radiance(lambda, temperature))
    }

    /// Calculate peak wavelength using Wien's displacement law
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Peak wavelength in meters
    #[must_use]
    pub fn peak_wavelength(&self, temperature: f64) -> f64 {
        if temperature <= 0.0 {
            return 0.0;
        }
        2.897771955e-3 / temperature // Wien's displacement constant
    }

    /// Calculate color temperature from spectrum
    ///
    /// # Arguments
    /// * `spectrum` - Emission spectrum
    /// * `wavelengths` - Corresponding wavelengths
    ///
    /// # Returns
    /// Estimated color temperature in Kelvin
    #[must_use]
    pub fn color_temperature(&self, spectrum: &Array1<f64>, wavelengths: &Array1<f64>) -> f64 {
        // Find peak wavelength
        let max_idx = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx);

        let peak_wavelength = wavelengths[max_idx];

        // Use Wien's law to estimate temperature
        if peak_wavelength > 0.0 {
            2.897771955e-3 / peak_wavelength
        } else {
            0.0
        }
    }
}

/// Calculate blackbody emission field from temperature field
#[must_use]
pub fn calculate_blackbody_emission(
    temperature_field: &Array3<f64>,
    bubble_radius_field: &Array3<f64>,
    model: &BlackbodyModel,
) -> Array3<f64> {
    // Use zip and map for zero-copy iteration
    Array3::from_shape_fn(temperature_field.dim(), |(i, j, k)| {
        let temp = temperature_field[[i, j, k]];
        let radius = bubble_radius_field[[i, j, k]];

        if radius > 0.0 && temp > 0.0 {
            // Surface area of bubble
            let surface_area = 4.0 * PI * radius * radius;

            // Total power emitted
            let power = model.total_power(temp, surface_area);

            // Convert to power density (W/m³)
            let volume = 4.0 / 3.0 * PI * radius.powi(3);
            power / volume.max(1e-20)
        } else {
            0.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_planck_law_limits() {
        let model = BlackbodyModel::default();

        // Test Wien limit (short wavelength)
        let lambda_short = 100e-9; // 100 nm
        let temp = 10000.0; // 10,000 K
        let radiance_wien = model.spectral_radiance(lambda_short, temp);
        assert!(radiance_wien > 0.0);

        // Test Rayleigh-Jeans limit (long wavelength)
        let lambda_long = 1e-3; // 1 mm
        let radiance_rj = model.spectral_radiance(lambda_long, temp);
        assert!(radiance_rj > 0.0);
    }

    #[test]
    fn test_wien_displacement() {
        let model = BlackbodyModel::default();

        // Sun's surface temperature
        let t_sun = 5778.0;
        let peak = model.peak_wavelength(t_sun);
        assert_relative_eq!(peak, 501.5e-9, epsilon = 1e-9); // ~500 nm (green)

        // Sonoluminescence temperature
        let t_sono = 20000.0;
        let peak_sono = model.peak_wavelength(t_sono);
        assert_relative_eq!(peak_sono, 144.9e-9, epsilon = 1e-9); // ~145 nm (UV)
    }

    #[test]
    fn test_stefan_boltzmann() {
        let model = BlackbodyModel {
            emissivity: 1.0,
            optical_depth: 1.0,
        };

        let temp = 1000.0;
        let area = 1.0;
        let power = model.total_power(temp, area);

        assert_relative_eq!(power, STEFAN_BOLTZMANN * 1e12, epsilon = 1.0);
    }
}
