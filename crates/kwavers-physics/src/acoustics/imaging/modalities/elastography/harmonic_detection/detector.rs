//! Harmonic detection and analysis processor

use super::config::HarmonicDetectionConfig;
use super::types::HarmonicDisplacementField;
use kwavers_core::error::KwaversResult;
use leto::{
    /* s -- no leto equivalent */,
    Array4,
};

/// Harmonic detection and analysis processor
pub struct HarmonicDetector {
    /// Configuration
    pub(crate) config: HarmonicDetectionConfig,
}

impl std::fmt::Debug for HarmonicDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarmonicDetector")
            .field("config", &self.config)
            .finish()
    }
}

impl HarmonicDetector {
    /// Create new harmonic detector
    #[must_use]
    pub fn new(config: HarmonicDetectionConfig) -> Self {
        Self { config }
    }

    /// Analyze displacement time series for harmonic content
    ///
    /// # Arguments
    ///
    /// * `displacement_time_series` - 4D array: [nx, ny, nz, n_time_points]
    /// * `sampling_frequency` - Sampling frequency (Hz)
    ///
    /// # Returns
    ///
    /// Harmonic displacement field with all frequency components
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn analyze_harmonics(
        &self,
        displacement_time_series: &Array4<f64>,
        sampling_frequency: f64,
    ) -> KwaversResult<HarmonicDisplacementField> {
        let (nx, ny, nz, n_times) = displacement_time_series.dim();

        let mut harmonic_field =
            HarmonicDisplacementField::new(nx, ny, nz, self.config.n_harmonics, n_times);

        // Set time and frequency vectors
        for t in 0..n_times {
            harmonic_field.time[t] = t as f64 / sampling_frequency;
        }

        let df = sampling_frequency / self.config.fft_window_size as f64;
        for f in 0..harmonic_field.frequency.len() {
            harmonic_field.frequency[f] = f as f64 * df;
        }

        // Analyze each spatial point
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let time_series = displacement_time_series.slice(s![i, j, k, ..]);
                    let harmonics =
                        self.analyze_single_point(&time_series.to_vec(), sampling_frequency)?;

                    // Store results
                    harmonic_field.fundamental_magnitude[[i, j, k]] =
                        harmonics.fundamental_magnitude;
                    harmonic_field.fundamental_phase[[i, j, k]] = harmonics.fundamental_phase;

                    for h in 0..self.config.n_harmonics {
                        if h < harmonics.harmonic_magnitudes.len() {
                            harmonic_field.harmonic_magnitudes[h][[i, j, k]] =
                                harmonics.harmonic_magnitudes[h];
                            harmonic_field.harmonic_phases[h][[i, j, k]] =
                                harmonics.harmonic_phases[h];
                            harmonic_field.harmonic_snrs[h][[i, j, k]] = harmonics.harmonic_snrs[h];
                        }
                    }
                }
            }
        }

        // Compute nonlinearity parameter
        harmonic_field.compute_nonlinearity_parameter(&self.config);

        Ok(harmonic_field)
    }
}
