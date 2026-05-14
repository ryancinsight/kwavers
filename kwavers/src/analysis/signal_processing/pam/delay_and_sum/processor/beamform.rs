//! Beamforming: delay computation, apodization, and DAS accumulation.

use ndarray::{Array1, Array2, ArrayView2};

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::fft_1d_array;

use super::super::types::ApodizationType;
use super::DelayAndSumPAM;

impl DelayAndSumPAM {
    /// Beamform passive acoustic data to produce a cavitation intensity map.
    ///
    /// # Arguments
    /// * `passive_data` — recorded signals `[sensors × samples]`.
    /// * `grid_points`  — candidate source positions `[points × 3]`.
    ///
    /// # Errors
    /// Propagates input-validation and interpolation errors.
    pub fn beamform(
        &self,
        passive_data: &Array2<f64>,
        grid_points: &Array2<f64>,
    ) -> KwaversResult<Array1<f64>> {
        self.beamform_view(passive_data.view(), grid_points.view())
    }

    /// Beamform without copying input matrices.
    ///
    /// `passive_data` shape: `[sensor, time]`; `grid_points` shape:
    /// `[candidate, xyz]`. Validation is performed before indexing so Rust
    /// and PyO3 callers share the same rejection contract.
    ///
    /// # Errors
    /// Returns `Err` on shape mismatches or non-finite values.
    pub fn beamform_view(
        &self,
        passive_data: ArrayView2<'_, f64>,
        grid_points: ArrayView2<'_, f64>,
    ) -> KwaversResult<Array1<f64>> {
        self.validate_beamform_inputs(passive_data, grid_points)?;

        let num_grid_points = grid_points.nrows();
        let mut intensity_map = Array1::<f64>::zeros(num_grid_points);
        let apodization_weights = self.compute_apodization_weights();

        for (grid_idx, grid_point) in grid_points.rows().into_iter().enumerate() {
            let candidate_pos = [grid_point[0], grid_point[1], grid_point[2]];
            let delays_samples = self.compute_delays(&candidate_pos)?;
            let intensity = self.delay_and_sum_at_point_view(
                passive_data,
                &delays_samples,
                &apodization_weights,
            )?;
            intensity_map[grid_idx] = intensity;
        }

        Ok(intensity_map)
    }

    pub(super) fn validate_beamform_inputs(
        &self,
        passive_data: ArrayView2<'_, f64>,
        grid_points: ArrayView2<'_, f64>,
    ) -> KwaversResult<()> {
        let (num_sensors_data, _) = passive_data.dim();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }
        if passive_data.ncols() == 0 {
            return Err(KwaversError::InvalidInput(
                "Passive data must contain at least one time sample".to_owned(),
            ));
        }
        if grid_points.ncols() != 3 {
            return Err(KwaversError::InvalidInput(format!(
                "Grid points must have shape [points x 3], got {} columns",
                grid_points.ncols()
            )));
        }
        if grid_points.nrows() == 0 {
            return Err(KwaversError::InvalidInput(
                "Grid points must contain at least one candidate".to_owned(),
            ));
        }
        if !passive_data.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Passive data must contain only finite values".to_owned(),
            ));
        }
        if !grid_points.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Grid points must contain only finite coordinates".to_owned(),
            ));
        }
        Ok(())
    }

    /// Compute propagation delays (samples) from each sensor to `source_pos`.
    ///
    /// `delay_i = ||r_s − r_i|| / c · fs`
    pub(crate) fn compute_delays(&self, source_pos: &[f64; 3]) -> KwaversResult<Vec<f64>> {
        let mut delays = Vec::with_capacity(self.num_sensors);
        for sensor_pos in &self.sensor_positions {
            let dx = source_pos[0] - sensor_pos[0];
            let dy = source_pos[1] - sensor_pos[1];
            let dz = source_pos[2] - sensor_pos[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
            delays.push(distance / self.config.sound_speed * self.config.sampling_frequency);
        }
        Ok(delays)
    }

    fn delay_and_sum_at_point_view(
        &self,
        passive_data: ArrayView2<'_, f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> KwaversResult<f64> {
        let summed_signal =
            self.beamformed_signal_at_point_view(passive_data, delays_samples, apodization)?;
        let intensity: f64 = summed_signal.iter().map(|&x| x * x).sum();
        Ok(intensity / summed_signal.len().max(1) as f64)
    }

    pub(super) fn beamformed_signal_at_point(
        &self,
        passive_data: &Array2<f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> KwaversResult<Vec<f64>> {
        self.beamformed_signal_at_point_view(passive_data.view(), delays_samples, apodization)
    }

    fn beamformed_signal_at_point_view(
        &self,
        passive_data: ArrayView2<'_, f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> KwaversResult<Vec<f64>> {
        let num_samples = passive_data.ncols();
        let window_size = self.config.window_size.min(num_samples).max(1);
        let mut summed_signal = vec![0.0; window_size];

        for (sensor_idx, &delay) in delays_samples.iter().enumerate() {
            let weight = apodization[sensor_idx];
            for (t, value) in summed_signal.iter_mut().enumerate().take(window_size) {
                let sample_pos = t as f64 + delay;
                if (0.0..=(num_samples - 1) as f64).contains(&sample_pos) {
                    let lo = sample_pos.floor() as usize;
                    let hi = (lo + 1).min(num_samples - 1);
                    let frac = sample_pos - lo as f64;
                    let interpolated = (1.0 - frac).mul_add(
                        passive_data[[sensor_idx, lo]],
                        frac * passive_data[[sensor_idx, hi]],
                    );
                    *value += weight * interpolated;
                }
            }
        }

        Ok(summed_signal)
    }

    /// Estimate the dominant spectral frequency of a beamformed signal via FFT.
    pub(super) fn estimate_peak_frequency(&self, signal: &[f64]) -> Option<f64> {
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
            let mag = value.re.mul_add(value.re, value.im * value.im);
            if mag > max_mag {
                max_mag = mag;
                max_idx = Some(idx);
            }
        }

        max_idx.map(|idx| (idx as f64 * self.config.sampling_frequency) / n as f64)
    }

    /// Compute apodization weights for sidelobe suppression.
    pub(crate) fn compute_apodization_weights(&self) -> Vec<f64> {
        let n = self.num_sensors;
        match self.config.apodization {
            ApodizationType::None => vec![1.0; n],
            ApodizationType::Hamming => (0..n)
                .map(|i| {
                    0.46f64.mul_add(
                        -(2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos(),
                        0.54,
                    )
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

    pub(super) fn noise_threshold(&self, intensity_map: &Array1<f64>) -> f64 {
        let mut sorted = intensity_map.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = sorted[sorted.len() / 4]; // lower quartile
        noise_floor * self.config.detection_threshold
    }

    pub(super) fn coherence_factor(&self, intensity: f64, noise_floor: f64) -> f64 {
        if self.config.coherence_weighting {
            (intensity / (intensity + noise_floor)).min(1.0)
        } else {
            1.0
        }
    }
}
