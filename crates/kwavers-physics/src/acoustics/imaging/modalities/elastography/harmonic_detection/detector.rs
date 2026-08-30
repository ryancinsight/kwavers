//! Harmonic detection and analysis processor

use super::config::HarmonicDetectionConfig;
use super::spectral::SpectralWorkspace;
use super::types::HarmonicDisplacementField;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::Array4;

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

    /// Analyze one complete displacement record at every spatial point.
    ///
    /// # Arguments
    ///
    /// * `displacement_time_series` - 4D array: [nx, ny, nz, n_time_points]
    /// * `sampling_frequency` - Sampling frequency (Hz)
    ///
    /// # Returns
    ///
    /// Harmonic displacement field with all requested frequency components.
    /// The time extent is the FFT size: each point receives one symmetric Hann
    /// window and one transform, with no implicit segmentation or overlap.
    /// Returned phases use the principal interval [-π, π]. Signal-to-noise
    /// ratios are reported from the full normalized spectrum's neighboring
    /// bins and never suppress an output harmonic. The Hann coefficients and
    /// FFT buffers allocate once per call and are reused across every spatial
    /// point.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::InvalidValue`] when the time extent is shorter
    /// than two samples, no harmonics are requested, the sampling frequency is
    /// not finite and positive, or the fundamental frequency is not finite and
    /// positive. The fundamental and highest requested harmonic must also lie
    /// at or below the Nyquist frequency.
    pub fn analyze_harmonics(
        &self,
        displacement_time_series: &Array4<f64>,
        sampling_frequency: f64,
    ) -> KwaversResult<HarmonicDisplacementField> {
        let [nx, ny, nz, n_times] = displacement_time_series.shape();
        self.validate_analysis(n_times, sampling_frequency)?;

        let mut harmonic_field =
            HarmonicDisplacementField::new(nx, ny, nz, self.config.n_harmonics - 1, n_times);

        // Set time and frequency vectors
        for (index, time) in harmonic_field.time.iter_mut().enumerate() {
            *time = index as f64 / sampling_frequency;
        }

        let frequency_resolution = sampling_frequency / n_times as f64;
        for (index, frequency) in harmonic_field.frequency.iter_mut().enumerate() {
            *frequency = index as f64 * frequency_resolution;
        }

        let mut workspace = SpectralWorkspace::new(n_times);
        if let Some(displacement) = displacement_time_series.as_slice() {
            let mut points = displacement.chunks_exact(n_times);
            for (point_index, samples) in points.by_ref().enumerate() {
                let spectrum = workspace.transform(samples.iter().copied());
                self.store_point(
                    &mut harmonic_field,
                    point_index,
                    spectrum,
                    sampling_frequency,
                    n_times,
                );
            }
            debug_assert!(points.remainder().is_empty());
        } else {
            let mut point_index = 0;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let samples =
                            (0..n_times).map(|time| displacement_time_series[[i, j, k, time]]);
                        let spectrum = workspace.transform(samples);
                        self.store_point(
                            &mut harmonic_field,
                            point_index,
                            spectrum,
                            sampling_frequency,
                            n_times,
                        );
                        point_index += 1;
                    }
                }
            }
        }

        // Compute nonlinearity parameter
        harmonic_field.compute_nonlinearity_parameter();

        Ok(harmonic_field)
    }

    fn validate_analysis(&self, sample_count: usize, sampling_frequency: f64) -> KwaversResult<()> {
        if sample_count < 2 {
            return Err(invalid_value(
                "displacement_time_series time extent",
                sample_count as f64,
                "must contain at least two samples",
            ));
        }
        if self.config.n_harmonics == 0 {
            return Err(invalid_value(
                "HarmonicDetectionConfig.n_harmonics",
                0.0,
                "must be greater than zero",
            ));
        }
        if !sampling_frequency.is_finite() || sampling_frequency <= 0.0 {
            return Err(invalid_value(
                "sampling_frequency",
                sampling_frequency,
                "must be finite and greater than zero",
            ));
        }
        if !self.config.fundamental_frequency.is_finite()
            || self.config.fundamental_frequency <= 0.0
        {
            return Err(invalid_value(
                "HarmonicDetectionConfig.fundamental_frequency",
                self.config.fundamental_frequency,
                "must be finite and greater than zero",
            ));
        }
        let nyquist_frequency = sampling_frequency / 2.0;
        if self.config.fundamental_frequency > nyquist_frequency {
            return Err(invalid_value(
                "HarmonicDetectionConfig.fundamental_frequency",
                self.config.fundamental_frequency,
                "must not exceed the Nyquist frequency",
            ));
        }
        let highest_requested_frequency =
            self.config.n_harmonics as f64 * self.config.fundamental_frequency;
        if !highest_requested_frequency.is_finite()
            || highest_requested_frequency > nyquist_frequency
        {
            return Err(invalid_value(
                "HarmonicDetectionConfig.n_harmonics",
                self.config.n_harmonics as f64,
                "all requested harmonics must lie at or below the Nyquist frequency",
            ));
        }
        Ok(())
    }

    fn store_point(
        &self,
        field: &mut HarmonicDisplacementField,
        point_index: usize,
        spectrum: &[apollo::Complex64],
        sampling_frequency: f64,
        sample_count: usize,
    ) {
        let (magnitude, phase, _) = Self::harmonic_component(
            spectrum,
            sampling_frequency,
            self.config.fundamental_frequency,
            1,
            sample_count,
        );
        field
            .fundamental_magnitude
            .as_slice_mut()
            .expect("invariant: harmonic output is C-contiguous")[point_index] = magnitude;
        field
            .fundamental_phase
            .as_slice_mut()
            .expect("invariant: harmonic output is C-contiguous")[point_index] = phase;

        for (output_index, ((magnitudes, phases), snrs)) in field
            .harmonic_magnitudes
            .iter_mut()
            .zip(&mut field.harmonic_phases)
            .zip(&mut field.harmonic_snrs)
            .enumerate()
        {
            let (magnitude, phase, snr) = Self::harmonic_component(
                spectrum,
                sampling_frequency,
                self.config.fundamental_frequency,
                output_index + 2,
                sample_count,
            );
            magnitudes
                .as_slice_mut()
                .expect("invariant: harmonic output is C-contiguous")[point_index] = magnitude;
            phases
                .as_slice_mut()
                .expect("invariant: harmonic output is C-contiguous")[point_index] = phase;
            snrs.as_slice_mut()
                .expect("invariant: harmonic output is C-contiguous")[point_index] = snr;
        }
    }
}

fn invalid_value(parameter: &str, value: f64, reason: &str) -> KwaversError {
    ValidationError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        reason: reason.to_owned(),
    }
    .into()
}
