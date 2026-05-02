//! `DelayAndSumPAM` — time-domain DAS beamformer for passive acoustic mapping.
//!
//! ## Algorithm (Gyöngy & Coussios 2010)
//!
//! ```text
//! For each candidate source position r_s:
//!   τᵢ = ||r_s − rᵢ|| / c        (propagation delay to sensor i)
//!   sᵢ'(t) = sᵢ(t + τᵢ)          (time-shifted signal)
//!   P(r_s, t) = Σᵢ wᵢ · sᵢ'(t)   (coherent sum with apodization wᵢ)
//!   I(r_s) = ∫ |P|² dt            (intensity)
//! ```

use super::types::{ApodizationType, CavitationEvent, DelayAndSumConfig};
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::fft_1d_array;
use ndarray::{Array1, Array2};

/// Delay-and-Sum PAM processor.
#[derive(Debug)]
pub struct DelayAndSumPAM {
    pub(super) config: DelayAndSumConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

impl DelayAndSumPAM {
    /// Create a new DAS PAM processor.
    ///
    /// Requires at least 3 sensors for 3D localization.
    pub fn new(sensor_positions: Vec<[f64; 3]>, config: DelayAndSumConfig) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();
        if num_sensors < 3 {
            return Err(KwaversError::InvalidInput(
                "Need at least 3 sensors for PAM".to_string(),
            ));
        }
        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_string(),
            ));
        }
        if config.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sampling frequency must be positive".to_string(),
            ));
        }
        Ok(Self {
            config,
            sensor_positions,
            num_sensors,
        })
    }

    /// Beamform passive acoustic data to produce a cavitation intensity map.
    ///
    /// # Arguments
    /// * `passive_data` — recorded signals `[sensors × samples]`
    /// * `grid_points`  — candidate source positions `[points × 3]`
    pub fn beamform(
        &self,
        passive_data: &Array2<f64>,
        grid_points: &Array2<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let (num_sensors_data, _) = passive_data.dim();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }

        let num_grid_points = grid_points.nrows();
        let mut intensity_map = Array1::<f64>::zeros(num_grid_points);
        let apodization_weights = self.compute_apodization_weights();

        for (grid_idx, grid_point) in grid_points.rows().into_iter().enumerate() {
            let candidate_pos = [grid_point[0], grid_point[1], grid_point[2]];
            let delays_samples = self.compute_delays(&candidate_pos)?;
            let intensity =
                self.delay_and_sum_at_point(passive_data, &delays_samples, &apodization_weights)?;
            intensity_map[grid_idx] = intensity;
        }

        Ok(intensity_map)
    }

    /// Detect cavitation events above threshold in a precomputed intensity map.
    pub fn detect_events(
        &self,
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
        time: f64,
    ) -> KwaversResult<Vec<CavitationEvent>> {
        if intensity_map.len() != grid_points.nrows() {
            return Err(KwaversError::InvalidInput(
                "Intensity map and grid points size mismatch".to_string(),
            ));
        }

        let threshold = self.noise_threshold(intensity_map);
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points.row(idx);
                let position = [grid_point[0], grid_point[1], grid_point[2]];
                let coherence = self.coherence_factor(intensity, threshold);
                events.push(CavitationEvent {
                    position,
                    intensity,
                    time,
                    coherence,
                    peak_frequency: None,
                });
            }
        }

        events.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(events)
    }

    /// Detect events and estimate peak frequency from raw sensor data.
    pub fn detect_events_with_data(
        &self,
        passive_data: &Array2<f64>,
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
        time: f64,
    ) -> KwaversResult<Vec<CavitationEvent>> {
        let (num_sensors_data, _) = passive_data.dim();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }
        if intensity_map.len() != grid_points.nrows() {
            return Err(KwaversError::InvalidInput(
                "Intensity map and grid points size mismatch".to_string(),
            ));
        }

        let threshold = self.noise_threshold(intensity_map);
        let apodization_weights = self.compute_apodization_weights();
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points.row(idx);
                let position = [grid_point[0], grid_point[1], grid_point[2]];
                let coherence = self.coherence_factor(intensity, threshold);
                let delays_samples = self.compute_delays(&position)?;
                let peak_frequency = self
                    .beamformed_signal_at_point(passive_data, &delays_samples, &apodization_weights)
                    .ok()
                    .and_then(|signal| self.estimate_peak_frequency(&signal));

                events.push(CavitationEvent {
                    position,
                    intensity,
                    time,
                    coherence,
                    peak_frequency,
                });
            }
        }

        events.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(events)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Compute propagation delays (samples) from each sensor to `source_pos`.
    pub(super) fn compute_delays(&self, source_pos: &[f64; 3]) -> KwaversResult<Vec<f64>> {
        let mut delays = Vec::with_capacity(self.num_sensors);
        for sensor_pos in &self.sensor_positions {
            let dx = source_pos[0] - sensor_pos[0];
            let dy = source_pos[1] - sensor_pos[1];
            let dz = source_pos[2] - sensor_pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            delays.push(distance / self.config.sound_speed * self.config.sampling_frequency);
        }
        Ok(delays)
    }

    /// Compute intensity at a single grid point via DAS.
    fn delay_and_sum_at_point(
        &self,
        passive_data: &Array2<f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> KwaversResult<f64> {
        let summed_signal =
            self.beamformed_signal_at_point(passive_data, delays_samples, apodization)?;
        let intensity: f64 = summed_signal.iter().map(|&x| x * x).sum();
        Ok(intensity / summed_signal.len().max(1) as f64)
    }

    fn beamformed_signal_at_point(
        &self,
        passive_data: &Array2<f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> KwaversResult<Vec<f64>> {
        let num_samples = passive_data.ncols();
        let window_size = self.config.window_size.min(num_samples).max(1);
        let mut summed_signal = vec![0.0; window_size];

        for (sensor_idx, &delay) in delays_samples.iter().enumerate() {
            let delay_idx = delay.round() as isize;
            let weight = apodization[sensor_idx];
            for (t, value) in summed_signal.iter_mut().enumerate().take(window_size) {
                let sample_idx = t as isize + delay_idx;
                if sample_idx >= 0 && (sample_idx as usize) < num_samples {
                    *value += weight * passive_data[[sensor_idx, sample_idx as usize]];
                }
            }
        }

        Ok(summed_signal)
    }

    fn estimate_peak_frequency(&self, signal: &[f64]) -> Option<f64> {
        let n = signal.len();
        if n < 2
            || !self.config.sampling_frequency.is_finite()
            || self.config.sampling_frequency <= 0.0
        {
            return None;
        }

        let complex_data = fft_1d_array(&Array1::from_vec(signal.to_vec()));
        let half = n / 2;
        let mut max_mag = 0.0f64;
        let mut max_idx: Option<usize> = None;

        for (idx, value) in complex_data.iter().take(half).enumerate().skip(1) {
            let mag = value.re * value.re + value.im * value.im;
            if mag > max_mag {
                max_mag = mag;
                max_idx = Some(idx);
            }
        }

        max_idx.map(|idx| (idx as f64 * self.config.sampling_frequency) / n as f64)
    }

    /// Compute apodization weights for sidelobe suppression.
    pub(super) fn compute_apodization_weights(&self) -> Vec<f64> {
        let n = self.num_sensors;
        match self.config.apodization {
            ApodizationType::None => vec![1.0; n],
            ApodizationType::Hamming => (0..n)
                .map(|i| {
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                })
                .collect(),
            ApodizationType::Hanning => (0..n)
                .map(|i| {
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
                })
                .collect(),
            ApodizationType::Blackman => (0..n)
                .map(|i| {
                    let alpha = 0.16;
                    let a0 = (1.0 - alpha) / 2.0;
                    let a1 = 0.5;
                    let a2 = alpha / 2.0;
                    let n_term = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
                    a0 - a1 * n_term.cos() + a2 * (2.0 * n_term).cos()
                })
                .collect(),
        }
    }

    fn noise_threshold(&self, intensity_map: &Array1<f64>) -> f64 {
        let mut sorted = intensity_map.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = sorted[sorted.len() / 4]; // lower quartile
        noise_floor * self.config.detection_threshold
    }

    fn coherence_factor(&self, intensity: f64, noise_floor: f64) -> f64 {
        if self.config.coherence_weighting {
            (intensity / (intensity + noise_floor)).min(1.0)
        } else {
            1.0
        }
    }
}
