//! Delay-and-Sum Passive Acoustic Mapping
//!
//! This module implements time-domain delay-and-sum beamforming for passive acoustic
//! mapping, the foundational algorithm for detecting and localizing cavitation events
//! during therapeutic ultrasound procedures.
//!
//! # Algorithm
//!
//! Delay-and-Sum (DAS) PAM coherently combines signals from multiple passive sensors
//! by compensating for propagation delays:
//!
//! ```text
//! For each candidate source position r_s:
//!   1. Compute delays: τᵢ = ||r_s - rᵢ|| / c for sensor i at position rᵢ
//!   2. Time-shift signals: sᵢ'(t) = sᵢ(t + τᵢ)
//!   3. Sum coherently: P(r_s, t) = Σᵢ sᵢ'(t)
//!   4. Compute intensity: I(r_s) = ∫ |P(r_s, t)|² dt
//! ```
//!
//! # References
//!
//! - Gyöngy & Coussios (2010). "Passive Spatial Mapping of Inertial Cavitation"
//!   *IEEE UFFC*, 57(1), 48-56. DOI: 10.1109/TUFFC.2010.1386
//! - Jensen et al. (2012). "Spatiotemporal Monitoring of High-Intensity Focused Ultrasound"
//!   *Ultrasound in Medicine & Biology*, 38(11), 1938-1950

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

/// Configuration for delay-and-sum PAM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayAndSumConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,

    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,

    /// Detection threshold (multiple of noise floor)
    pub detection_threshold: f64,

    /// Temporal window size (samples)
    pub window_size: usize,

    /// Apodization window type
    pub apodization: ApodizationType,

    /// Enable coherence factor weighting
    pub coherence_weighting: bool,
}

/// Apodization window types for sidelobe suppression
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ApodizationType {
    /// No apodization (uniform weighting)
    None,

    /// Hamming window
    Hamming,

    /// Hanning window
    Hanning,

    /// Blackman window
    Blackman,
}

impl Default for DelayAndSumConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1540.0,      // Soft tissue
            sampling_frequency: 5e6,  // 5 MHz
            detection_threshold: 3.0, // 3x noise floor
            window_size: 512,
            apodization: ApodizationType::Hamming,
            coherence_weighting: true,
        }
    }
}

/// Delay-and-Sum PAM processor
#[derive(Debug)]
pub struct DelayAndSumPAM {
    config: DelayAndSumConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

/// Detected cavitation event
#[derive(Debug, Clone)]
pub struct CavitationEvent {
    /// 3D position (m)
    pub position: [f64; 3],

    /// Intensity (arbitrary units)
    pub intensity: f64,

    /// Time of occurrence (s)
    pub time: f64,

    /// Coherence factor (0-1)
    pub coherence: f64,

    /// Frequency content (Hz)
    pub peak_frequency: Option<f64>,
}

impl DelayAndSumPAM {
    /// Create a new delay-and-sum PAM processor
    ///
    /// # Arguments
    ///
    /// * `sensor_positions` - Array positions in meters [[x, y, z], ...]
    /// * `config` - PAM configuration
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

    /// Beamform passive acoustic data to create cavitation map
    ///
    /// # Arguments
    ///
    /// * `passive_data` - Recorded signals [sensors × samples]
    /// * `grid_points` - Candidate source positions [points × 3]
    ///
    /// # Returns
    ///
    /// Intensity map at each grid point
    pub fn beamform(
        &self,
        passive_data: &Array2<f64>,
        grid_points: &Array2<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let (num_sensors_data, _num_samples) = passive_data.dim();

        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }

        let num_grid_points = grid_points.nrows();
        let mut intensity_map = Array1::<f64>::zeros(num_grid_points);

        // Compute apodization weights
        let apodization_weights = self.compute_apodization_weights();

        // Beamform to each grid point
        for (grid_idx, grid_point) in grid_points.rows().into_iter().enumerate() {
            let candidate_pos = [grid_point[0], grid_point[1], grid_point[2]];

            // Compute delays for this candidate position
            let delays_samples = self.compute_delays(&candidate_pos)?;

            // Apply delay-and-sum with temporal integration
            let intensity =
                self.delay_and_sum_at_point(passive_data, &delays_samples, &apodization_weights)?;

            intensity_map[grid_idx] = intensity;
        }

        Ok(intensity_map)
    }

    /// Detect cavitation events above threshold
    ///
    /// # Arguments
    ///
    /// * `intensity_map` - Beamformed intensity values
    /// * `grid_points` - Corresponding grid positions
    /// * `time` - Time of this frame (s)
    ///
    /// # Returns
    ///
    /// Vector of detected events
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

        // Compute noise floor (median of lower 50% intensities)
        let mut sorted_intensities = intensity_map.to_vec();
        sorted_intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_idx = sorted_intensities.len() / 2;
        let noise_floor = sorted_intensities[median_idx / 2];

        let threshold = noise_floor * self.config.detection_threshold;

        // Find local maxima above threshold
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points.row(idx);
                let position = [grid_point[0], grid_point[1], grid_point[2]];

                // Compute coherence factor (simplified)
                let coherence = if self.config.coherence_weighting {
                    (intensity / (intensity + noise_floor)).min(1.0)
                } else {
                    1.0
                };

                events.push(CavitationEvent {
                    position,
                    intensity,
                    time,
                    coherence,
                    peak_frequency: None, // Could be computed from spectral analysis
                });
            }
        }

        // Sort by intensity (descending)
        events.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(events)
    }

    /// Detect cavitation events and estimate peak frequency from raw sensor data.
    pub fn detect_events_with_data(
        &self,
        passive_data: &Array2<f64>,
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
        time: f64,
    ) -> KwaversResult<Vec<CavitationEvent>> {
        let (num_sensors_data, _num_samples) = passive_data.dim();
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

        // Compute noise floor (median of lower 50% intensities)
        let mut sorted_intensities = intensity_map.to_vec();
        sorted_intensities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_idx = sorted_intensities.len() / 2;
        let noise_floor = sorted_intensities[median_idx / 2];

        let threshold = noise_floor * self.config.detection_threshold;
        let apodization_weights = self.compute_apodization_weights();

        // Find local maxima above threshold
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points.row(idx);
                let position = [grid_point[0], grid_point[1], grid_point[2]];

                // Compute coherence factor (simplified)
                let coherence = if self.config.coherence_weighting {
                    (intensity / (intensity + noise_floor)).min(1.0)
                } else {
                    1.0
                };

                let delays_samples = self.compute_delays(&position)?;
                let peak_frequency = self
                    .beamformed_signal_at_point(
                        passive_data,
                        &delays_samples,
                        &apodization_weights,
                    )
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

        // Sort by intensity (descending)
        events.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(events)
    }

    /// Compute propagation delays from candidate source to sensors
    fn compute_delays(&self, source_pos: &[f64; 3]) -> KwaversResult<Vec<f64>> {
        let mut delays = Vec::with_capacity(self.num_sensors);

        for sensor_pos in &self.sensor_positions {
            let dx = source_pos[0] - sensor_pos[0];
            let dy = source_pos[1] - sensor_pos[1];
            let dz = source_pos[2] - sensor_pos[2];

            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let delay_time = distance / self.config.sound_speed;
            let delay_samples = delay_time * self.config.sampling_frequency;

            delays.push(delay_samples);
        }

        Ok(delays)
    }

    /// Apply delay-and-sum at a single point
    fn delay_and_sum_at_point(
        &self,
        passive_data: &Array2<f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> KwaversResult<f64> {
        let summed_signal =
            self.beamformed_signal_at_point(passive_data, delays_samples, apodization)?;

        // Compute intensity (energy over window)
        let intensity: f64 = summed_signal.iter().map(|&x| x * x).sum();
        let normalized_intensity = intensity / summed_signal.len().max(1) as f64;

        Ok(normalized_intensity)
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

        // Apply delays and sum
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
        if n < 2 {
            return None;
        }

        if !self.config.sampling_frequency.is_finite() || self.config.sampling_frequency <= 0.0 {
            return None;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut complex_data: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut complex_data);

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

    /// Compute apodization weights for sidelobe suppression
    fn compute_apodization_weights(&self) -> Vec<f64> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pam_creation() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let config = DelayAndSumConfig::default();
        let pam = DelayAndSumPAM::new(sensors, config);
        assert!(pam.is_ok());
    }

    #[test]
    fn test_insufficient_sensors() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
        let config = DelayAndSumConfig::default();
        assert!(DelayAndSumPAM::new(sensors, config).is_err());
    }

    #[test]
    fn test_delay_computation() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let config = DelayAndSumConfig::default();
        let pam = DelayAndSumPAM::new(sensors, config).unwrap();

        // Source at origin should have zero delay to first sensor
        let source_pos = [0.0, 0.0, 0.0];
        let delays = pam.compute_delays(&source_pos).unwrap();

        assert_eq!(delays.len(), 3);
        assert_relative_eq!(delays[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_apodization_weights() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let config = DelayAndSumConfig {
            apodization: ApodizationType::None,
            ..Default::default()
        };
        let pam = DelayAndSumPAM::new(sensors, config).unwrap();

        let weights = pam.compute_apodization_weights();
        assert_eq!(weights.len(), 3);
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_beamform_basic() {
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.01, 0.01, 0.0],
        ];
        let config = DelayAndSumConfig::default();
        let pam = DelayAndSumPAM::new(sensors, config).unwrap();

        // Create synthetic passive data (4 sensors × 1000 samples)
        let passive_data = Array2::<f64>::from_shape_fn((4, 1000), |(i, t)| {
            (2.0 * std::f64::consts::PI * t as f64 / 100.0 + i as f64).sin()
        });

        // Create grid points
        let grid_points = Array2::<f64>::from_shape_fn((5, 3), |(i, j)| match j {
            0 => (i as f64 - 2.0) * 0.005,
            1 => (i as f64 - 2.0) * 0.005,
            2 => 0.02,
            _ => 0.0,
        });

        let intensity_map = pam.beamform(&passive_data, &grid_points).unwrap();

        assert_eq!(intensity_map.len(), 5);
        assert!(intensity_map.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_event_detection() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let config = DelayAndSumConfig {
            detection_threshold: 2.0,
            ..Default::default()
        };
        let pam = DelayAndSumPAM::new(sensors, config).unwrap();

        let intensity_map = Array1::from_vec(vec![0.5, 0.8, 5.0, 1.0, 0.3]);
        let grid_points = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.0, 0.02, 0.01, 0.0, 0.02, 0.005, 0.005, 0.02, 0.0, 0.01, 0.02, -0.01, 0.0,
                0.02,
            ],
        )
        .unwrap();

        let events = pam
            .detect_events(&intensity_map, &grid_points, 0.0)
            .unwrap();

        // Should detect at least the point with intensity 5.0
        assert!(!events.is_empty());
        assert!(events[0].intensity > 2.0);
    }

    #[test]
    fn test_event_detection_with_peak_frequency() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let config = DelayAndSumConfig {
            sampling_frequency: 10e6,
            window_size: 256,
            detection_threshold: 0.5,
            ..Default::default()
        };
        let pam = DelayAndSumPAM::new(sensors, config).unwrap();

        let freq = 1e6;
        let num_samples = 256;
        let mut passive_data = Array2::zeros((3, num_samples));
        for t in 0..num_samples {
            let time = t as f64 / pam.config.sampling_frequency;
            let sample = (2.0 * std::f64::consts::PI * freq * time).sin();
            for sensor in 0..3 {
                passive_data[[sensor, t]] = sample;
            }
        }

        let intensity_map = Array1::from_vec(vec![2.0]);
        let grid_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();

        let events = pam
            .detect_events_with_data(&passive_data, &intensity_map, &grid_points, 0.0)
            .unwrap();

        assert!(!events.is_empty());
        let peak = events[0].peak_frequency.expect("peak frequency should be available");
        let resolution = pam.config.sampling_frequency / num_samples as f64;
        assert!((peak - freq).abs() <= resolution);
    }
}
