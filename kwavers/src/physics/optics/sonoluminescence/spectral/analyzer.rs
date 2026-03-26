use super::range::SpectralRange;
use super::spectrum::EmissionSpectrum;
use crate::core::constants::optical::WIEN_CONSTANT;
use ndarray::Array1;

/// Spectral analyzer for sonoluminescence
#[derive(Debug)]
pub struct SpectralAnalyzer {
    /// Spectral range
    pub range: SpectralRange,
    /// Time history of spectra
    pub spectra_history: Vec<EmissionSpectrum>,
    /// Maximum number of spectra to store
    pub max_history: usize,
}

impl SpectralAnalyzer {
    /// Create new spectral analyzer
    #[must_use]
    pub fn new(range: SpectralRange) -> Self {
        Self {
            range,
            spectra_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Add spectrum to history
    pub fn add_spectrum(&mut self, spectrum: EmissionSpectrum) {
        self.spectra_history.push(spectrum);

        // Limit history size
        if self.spectra_history.len() > self.max_history {
            self.spectra_history.remove(0);
        }
    }

    /// Get time evolution of peak wavelength
    #[must_use]
    pub fn peak_wavelength_evolution(&self) -> (Array1<f64>, Array1<f64>) {
        let n = self.spectra_history.len();
        let mut times = Array1::zeros(n);
        let mut peaks = Array1::zeros(n);

        for (i, spectrum) in self.spectra_history.iter().enumerate() {
            times[i] = spectrum.time;
            peaks[i] = spectrum.peak_wavelength();
        }

        (times, peaks)
    }

    /// Get time evolution of total intensity
    #[must_use]
    pub fn intensity_evolution(&self) -> (Array1<f64>, Array1<f64>) {
        let n = self.spectra_history.len();
        let mut times = Array1::zeros(n);
        let mut intensities = Array1::zeros(n);

        for (i, spectrum) in self.spectra_history.iter().enumerate() {
            times[i] = spectrum.time;
            intensities[i] = spectrum.total_intensity();
        }

        (times, intensities)
    }

    /// Calculate time-averaged spectrum
    #[must_use]
    pub fn time_averaged_spectrum(&self) -> Option<EmissionSpectrum> {
        if self.spectra_history.is_empty() {
            return None;
        }

        let wavelengths = self.spectra_history[0].wavelengths.clone();
        let mut avg_intensities = Array1::zeros(wavelengths.len());

        for spectrum in &self.spectra_history {
            avg_intensities += &spectrum.intensities;
        }

        avg_intensities /= self.spectra_history.len() as f64;

        Some(EmissionSpectrum::new(
            wavelengths,
            avg_intensities,
            self.spectra_history.last().unwrap().time,
        ))
    }

    /// Fit blackbody temperature to spectrum
    #[must_use]
    pub fn fit_blackbody_temperature(&self, spectrum: &EmissionSpectrum) -> f64 {
        // Use Wien's displacement law on peak wavelength
        let peak = spectrum.peak_wavelength();
        if peak > 0.0 {
            WIEN_CONSTANT / peak // Wien's displacement law
        } else {
            0.0
        }
    }
}
