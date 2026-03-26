use crate::core::constants::optical::WIEN_CONSTANT;
use ndarray::Array1;

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
    #[must_use]
    pub fn from_time_series(
        times: &Array1<f64>,
        intensities: &Array1<f64>,
        temperatures: &Array1<f64>,
        spectra: &[crate::physics::optics::sonoluminescence::spectral::EmissionSpectrum],
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
                WIEN_CONSTANT / peak_wl
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
