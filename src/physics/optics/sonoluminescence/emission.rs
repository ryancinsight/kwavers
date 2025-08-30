//! Main sonoluminescence emission module
//!
//! Integrates blackbody, bremsstrahlung, and molecular emission models

use super::{
    blackbody::{calculate_blackbody_emission, BlackbodyModel},
    bremsstrahlung::{calculate_bremsstrahlung_emission, BremsstrahlungModel},
    spectral::{EmissionSpectrum, SpectralAnalyzer, SpectralRange},
};
use ndarray::{s, Array1, Array3, Array4};

/// Parameters for sonoluminescence emission
#[derive(Debug, Clone)]
pub struct EmissionParameters {
    /// Enable blackbody radiation
    pub use_blackbody: bool,
    /// Enable bremsstrahlung radiation
    pub use_bremsstrahlung: bool,
    /// Enable molecular line emission
    pub use_molecular_lines: bool,
    /// Ionization energy for gas (eV)
    pub ionization_energy: f64,
    /// Minimum temperature for light emission (K)
    pub min_temperature: f64,
    /// Opacity correction factor
    pub opacity_factor: f64,
}

impl Default for EmissionParameters {
    fn default() -> Self {
        Self {
            use_blackbody: true,
            use_bremsstrahlung: true,
            use_molecular_lines: false, // Not implemented yet
            ionization_energy: 15.76,   // eV for argon
            min_temperature: 2000.0,    // K
            opacity_factor: 1.0,        // Optically thin
        }
    }
}

/// Spectral field using Struct-of-Arrays for better performance
#[derive(Debug, Debug)]
pub struct SpectralField {
    /// Wavelength grid (shared for all spatial points)
    pub wavelengths: Array1<f64>,
    /// Spectral intensities: dimensions (nx, ny, nz, n_wavelengths)
    pub intensities: Array4<f64>,
    /// Peak wavelength at each point: dimensions (nx, ny, nz)
    pub peak_wavelength: Array3<f64>,
    /// Total intensity at each point: dimensions (nx, ny, nz)
    pub total_intensity: Array3<f64>,
    /// Color temperature at each point: dimensions (nx, ny, nz)
    pub color_temperature: Array3<f64>,
}

impl SpectralField {
    /// Create new spectral field
    pub fn new(grid_shape: (usize, usize, usize), wavelengths: Array1<f64>) -> Self {
        let n_wavelengths = wavelengths.len();
        let shape_4d = (grid_shape.0, grid_shape.1, grid_shape.2, n_wavelengths);

        Self {
            wavelengths,
            intensities: Array4::zeros(shape_4d),
            peak_wavelength: Array3::zeros(grid_shape),
            total_intensity: Array3::zeros(grid_shape),
            color_temperature: Array3::zeros(grid_shape),
        }
    }

    /// Update derived quantities (peak wavelength, total intensity, etc.)
    pub fn update_derived_quantities(&mut self) {
        let shape = (
            self.intensities.shape()[0],
            self.intensities.shape()[1],
            self.intensities.shape()[2],
        );

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    // Get spectrum at this point
                    let spectrum = self.intensities.slice(s![i, j, k, ..]);

                    // Total intensity
                    self.total_intensity[[i, j, k]] = spectrum.sum();

                    // Peak wavelength
                    if let Some(max_idx) = spectrum
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                    {
                        self.peak_wavelength[[i, j, k]] = self.wavelengths[max_idx];
                    }

                    // Color temperature (simplified Wien's law)
                    if self.peak_wavelength[[i, j, k]] > 0.0 {
                        self.color_temperature[[i, j, k]] =
                            2.898e-3 / self.peak_wavelength[[i, j, k]];
                    }
                }
            }
        }
    }

    /// Get spectrum at a specific point
    pub fn get_spectrum_at(&self, i: usize, j: usize, k: usize) -> EmissionSpectrum {
        let intensities = self.intensities.slice(s![i, j, k, ..]).to_owned();
        EmissionSpectrum::new(self.wavelengths.clone(), intensities, 0.0)
    }
}

/// Main sonoluminescence emission calculator
#[derive(Debug, Debug)]
pub struct SonoluminescenceEmission {
    /// Emission parameters
    pub params: EmissionParameters,
    /// Blackbody model
    pub blackbody: BlackbodyModel,
    /// Bremsstrahlung model
    pub bremsstrahlung: BremsstrahlungModel,
    /// Spectral analyzer
    pub analyzer: SpectralAnalyzer,
    /// Total emission field (W/m³)
    pub emission_field: Array3<f64>,
    /// Spectral emission field (Struct-of-Arrays)
    pub spectral_field: Option<SpectralField>,
}

impl SonoluminescenceEmission {
    /// Create new emission calculator
    pub fn new(grid_shape: (usize, usize, usize), params: EmissionParameters) -> Self {
        Self {
            params,
            blackbody: BlackbodyModel::default(),
            bremsstrahlung: BremsstrahlungModel::default(),
            analyzer: SpectralAnalyzer::new(SpectralRange::default()),
            emission_field: Array3::zeros(grid_shape),
            spectral_field: None,
        }
    }

    /// Calculate total light emission from bubble fields
    pub fn calculate_emission(
        &mut self,
        temperature_field: &Array3<f64>,
        pressure_field: &Array3<f64>,
        radius_field: &Array3<f64>,
        time: f64,
    ) {
        // Reset emission field
        self.emission_field.fill(0.0);

        // Calculate blackbody emission
        if self.params.use_blackbody {
            let bb_emission =
                calculate_blackbody_emission(temperature_field, radius_field, &self.blackbody);
            self.emission_field = &self.emission_field + &bb_emission;
        }

        // Calculate bremsstrahlung emission
        if self.params.use_bremsstrahlung {
            let br_emission = calculate_bremsstrahlung_emission(
                temperature_field,
                pressure_field,
                radius_field,
                &self.bremsstrahlung,
                self.params.ionization_energy,
            );
            self.emission_field = &self.emission_field + &br_emission;
        }

        // Apply minimum temperature cutoff
        for ((i, j, k), emission) in self.emission_field.indexed_iter_mut() {
            if temperature_field[[i, j, k]] < self.params.min_temperature {
                *emission = 0.0;
            } else {
                *emission *= self.params.opacity_factor;
            }
        }
    }

    /// Calculate spectral emission at a specific point
    pub fn calculate_spectrum_at_point(
        &self,
        temperature: f64,
        pressure: f64,
        radius: f64,
    ) -> EmissionSpectrum {
        let wavelengths = self.analyzer.range.wavelengths();
        let mut intensities = Array1::zeros(wavelengths.len());

        if temperature < self.params.min_temperature || radius <= 0.0 {
            return EmissionSpectrum::new(wavelengths, intensities, 0.0);
        }

        // Blackbody contribution
        if self.params.use_blackbody {
            let bb_spectrum = self.blackbody.emission_spectrum(temperature, &wavelengths);
            intensities = intensities + bb_spectrum;
        }

        // Bremsstrahlung contribution
        if self.params.use_bremsstrahlung && temperature > 5000.0 {
            // Calculate ionization
            let x_ion = self.bremsstrahlung.saha_ionization(
                temperature,
                pressure,
                self.params.ionization_energy,
            );

            let n_total = pressure / (1.380649e-23 * temperature);
            let n_electron = x_ion * n_total;
            let n_ion = n_electron;

            let br_spectrum = self.bremsstrahlung.emission_spectrum(
                temperature,
                n_electron,
                n_ion,
                2.0 * radius, // Path length through bubble
                &wavelengths,
            );
            intensities = intensities + br_spectrum;
        }

        // Apply opacity correction
        intensities *= self.params.opacity_factor;

        EmissionSpectrum::new(wavelengths, intensities, 0.0)
    }

    /// Calculate full spectral field
    pub fn calculate_spectral_field(
        &mut self,
        temperature_field: &Array3<f64>,
        pressure_field: &Array3<f64>,
        radius_field: &Array3<f64>,
        time: f64,
    ) {
        let shape = temperature_field.dim();
        let wavelengths = self.analyzer.range.wavelengths();
        let mut spectral_field = SpectralField::new(shape, wavelengths);

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let mut spectrum = self.calculate_spectrum_at_point(
                        temperature_field[[i, j, k]],
                        pressure_field[[i, j, k]],
                        radius_field[[i, j, k]],
                    );
                    spectrum.time = time;
                    spectrum.position = Some((i, j, k));
                    // Assign spectrum intensities to the 4D array
                    for (idx, &intensity) in spectrum.intensities.iter().enumerate() {
                        spectral_field.intensities[[i, j, k, idx]] = intensity;
                    }
                }
            }
        }

        spectral_field.update_derived_quantities();
        self.spectral_field = Some(spectral_field);
    }

    /// Get total light output
    pub fn total_light_output(&self) -> f64 {
        self.emission_field.sum()
    }

    /// Get peak emission location
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
    pub fn estimate_color_temperature(&self, temperature_field: &Array3<f64>) -> f64 {
        let (i, j, k) = self.peak_emission_location();
        temperature_field[[i, j, k]]
    }

    /// Get spectral statistics from the spectral field
    pub fn get_spectral_statistics(&self) -> Option<SpectralStatistics> {
        self.spectral_field
            .as_ref()
            .map(|field| SpectralStatistics {
                mean_peak_wavelength: field.peak_wavelength.mean().unwrap_or(0.0),
                mean_color_temperature: field.color_temperature.mean().unwrap_or(0.0),
                max_total_intensity: field.total_intensity.iter().cloned().fold(0.0, f64::max),
                peak_location: self.peak_emission_location(),
            })
    }

    /// Get spectrum at peak emission location
    pub fn get_peak_spectrum(&self) -> Option<EmissionSpectrum> {
        self.spectral_field.as_ref().map(|field| {
            let (i, j, k) = self.peak_emission_location();
            field.get_spectrum_at(i, j, k)
        })
    }
}

/// Spectral statistics
#[derive(Debug, Clone)]
pub struct SpectralStatistics {
    pub mean_peak_wavelength: f64,
    pub mean_color_temperature: f64,
    pub max_total_intensity: f64,
    pub peak_location: (usize, usize, usize),
}

/// Calculate sonoluminescence pulse characteristics
#[derive(Debug, Clone)]
pub struct SonoluminescencePulse {
    /// Peak intensity (W/m³)
    pub peak_intensity: f64,
    /// Pulse duration (s)
    pub duration: f64,
    /// Total energy (J)
    pub total_energy: f64,
    /// Peak temperature (K)
    pub peak_temperature: f64,
    /// Peak wavelength (m)
    pub peak_wavelength: f64,
    /// Color temperature (K)
    pub color_temperature: f64,
}

impl SonoluminescencePulse {
    /// Analyze emission time series to extract pulse characteristics
    pub fn from_time_series(
        times: &Array1<f64>,
        intensities: &Array1<f64>,
        temperatures: &Array1<f64>,
        spectra: &[EmissionSpectrum],
    ) -> Option<Self> {
        if times.len() < 2 || intensities.len() != times.len() {
            return None;
        }

        // Find peak intensity
        let (peak_idx, &peak_intensity) = intensities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        // Find FWHM duration
        let half_max = peak_intensity / 2.0;
        let mut start_idx = peak_idx;
        let mut end_idx = peak_idx;

        for i in (0..peak_idx).rev() {
            if intensities[i] < half_max {
                start_idx = i;
                break;
            }
        }

        for i in peak_idx..intensities.len() {
            if intensities[i] < half_max {
                end_idx = i;
                break;
            }
        }

        let duration = times[end_idx] - times[start_idx];

        // Calculate total energy (integrate intensity over time)
        let mut total_energy = 0.0;
        for i in 1..times.len() {
            let dt = times[i] - times[i - 1];
            let avg_intensity = 0.5 * (intensities[i] + intensities[i - 1]);
            total_energy += avg_intensity * dt;
        }

        // Get peak temperature
        let peak_temperature = temperatures[peak_idx];

        // Get spectral characteristics at peak
        let (peak_wavelength, color_temperature) = if peak_idx < spectra.len() {
            let spectrum = &spectra[peak_idx];
            let peak_wl = spectrum.peak_wavelength();
            let color_temperature = if peak_wl > 0.0 {
                2.897771955e-3 / peak_wl
            } else {
                0.0
            };
            (peak_wl, color_temperature)
        } else {
            (0.0, 0.0)
        };

        Some(Self {
            peak_intensity,
            duration,
            total_energy,
            peak_temperature,
            peak_wavelength,
            color_temperature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emission_calculation() {
        let shape = (10, 10, 10);
        let mut emission = SonoluminescenceEmission::new(shape, EmissionParameters::default());

        // Create test fields
        let mut temp_field = Array3::zeros(shape);
        let pressure_field = Array3::from_elem(shape, 101325.0); // 1 atm
        let radius_field = Array3::from_elem(shape, 5e-6); // 5 μm

        // Set high temperature at center
        temp_field[[5, 5, 5]] = 20000.0; // 20,000 K

        // Calculate emission
        emission.calculate_emission(&temp_field, &pressure_field, &radius_field, 0.0);

        // Check that emission occurred at hot spot
        assert!(emission.emission_field[[5, 5, 5]] > 0.0);
        assert_eq!(emission.peak_emission_location(), (5, 5, 5));
    }

    #[test]
    fn test_spectrum_calculation() {
        let emission = SonoluminescenceEmission::new((1, 1, 1), EmissionParameters::default());

        let spectrum = emission.calculate_spectrum_at_point(
            10000.0,  // 10,000 K
            101325.0, // 1 atm
            5e-6,     // 5 μm radius
        );

        // Should have emission
        assert!(spectrum.total_intensity() > 0.0);

        // Peak should be in UV for this temperature
        let peak = spectrum.peak_wavelength();
        assert!(peak > 100e-9 && peak < 400e-9);
    }
}
