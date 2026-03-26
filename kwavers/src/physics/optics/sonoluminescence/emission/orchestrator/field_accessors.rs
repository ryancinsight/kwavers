//! Field accessor and statistics methods for emission data

use ndarray::Array3;

use crate::physics::optics::sonoluminescence::spectral::EmissionSpectrum;

use super::dynamics::IntegratedSonoluminescence;
use super::emission_calculator::SonoluminescenceEmission;
use crate::physics::optics::sonoluminescence::emission::statistics::SpectralStatistics;

impl IntegratedSonoluminescence {
    /// Get the current emission field
    #[must_use]
    pub fn emission_field(&self) -> &Array3<f64> {
        &self.emission.emission_field
    }

    /// Get the current temperature field
    #[must_use]
    pub fn temperature_field(&self) -> &Array3<f64> {
        &self.temperature_field
    }

    /// Get the current pressure field
    #[must_use]
    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.pressure_field
    }

    /// Get the current radius field
    #[must_use]
    pub fn radius_field(&self) -> &Array3<f64> {
        &self.radius_field
    }
}

impl SonoluminescenceEmission {
    /// Get total light output
    #[must_use]
    pub fn total_light_output(&self) -> f64 {
        self.emission_field.sum()
    }

    /// Get peak emission location
    #[must_use]
    pub fn peak_emission_location(&self) -> (usize, usize, usize) {
        let mut max_val = 0.0;
        let mut max_loc = (0, 0, 0);

        for ((i, j, k), &val) in self.emission_field.indexed_iter() {
            if val > max_val {
                max_val = val;
                max_loc = (i, j, k);
            }
        }

        max_loc
    }

    /// Estimate color temperature from peak emission
    #[must_use]
    pub fn estimate_color_temperature(&self, temperature_field: &Array3<f64>) -> f64 {
        let (i, j, k) = self.peak_emission_location();
        temperature_field[[i, j, k]]
    }

    /// Get spectral statistics from the spectral field
    #[must_use]
    pub fn get_spectral_statistics(&self) -> Option<SpectralStatistics> {
        self.spectral_field
            .as_ref()
            .map(|field| SpectralStatistics {
                mean_peak_wavelength: field.peak_wavelength.mean().unwrap_or(0.0),
                mean_color_temperature: field.color_temperature.mean().unwrap_or(0.0),
                max_total_intensity: field.total_intensity.iter().copied().fold(0.0, f64::max),
                peak_location: self.peak_emission_location(),
            })
    }

    /// Get spectrum at peak emission location
    #[must_use]
    pub fn get_peak_spectrum(&self) -> Option<EmissionSpectrum> {
        self.spectral_field.as_ref().map(|field| {
            let (i, j, k) = self.peak_emission_location();
            field.get_spectrum_at(i, j, k)
        })
    }
}
