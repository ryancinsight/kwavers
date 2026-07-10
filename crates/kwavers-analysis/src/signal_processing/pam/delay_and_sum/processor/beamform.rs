//! Beamforming: delay computation, apodization, and DAS accumulation.

use leto::Array1;
use leto::{
    Array2,
    ArrayView2,
};

use apollo::fft_1d_array;
use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::types::PamImagingMode;
use super::DelayAndSumPAM;

impl DelayAndSumPAM {
    /// Beamform passive acoustic data to produce a cavitation intensity map
    /// using the classical delay-and-sum imaging condition.
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

    /// Delay-and-sum beamform without copying input matrices.
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
        self.beamform_with_mode_view(passive_data, grid_points, PamImagingMode::DelayAndSum)
    }

    /// Beamform with an explicit [`PamImagingMode`] (delay-and-sum or the
    /// sign-preserving delay-multiply-and-sum). DMAS sharpens the mainlobe and
    /// suppresses sidelobes by replacing the coherent sum with the pairwise
    /// signal correlation (Matrone et al. 2015).
    ///
    /// # Errors
    /// Propagates input-validation and interpolation errors.
    pub fn beamform_with_mode(
        &self,
        passive_data: &Array2<f64>,
        grid_points: &Array2<f64>,
        mode: PamImagingMode,
    ) -> KwaversResult<Array1<f64>> {
        self.beamform_with_mode_view(passive_data.view(), grid_points.view(), mode)
    }

    /// Mode-selected beamforming without copying input matrices.
    ///
    /// # Errors
    /// Returns `Err` on shape mismatches or non-finite values.
    pub fn beamform_with_mode_view(
        &self,
        passive_data: ArrayView2<'_, f64>,
        grid_points: ArrayView2<'_, f64>,
        mode: PamImagingMode,
    ) -> KwaversResult<Array1<f64>> {
        self.validate_beamform_inputs(passive_data, grid_points)?;

        let num_grid_points = grid_points.shape()[0];
        let mut intensity_map = Array1::<f64>::zeros([num_grid_points]);
        let apodization_weights = self.compute_apodization_weights();

        for (grid_idx, grid_point) in grid_points
            .rows()
            .expect("invariant: rank-2 array has a row axis")
            .enumerate()
        {
            let candidate_pos = [grid_point[0], grid_point[1], grid_point[2]];
            let delays_samples = self.compute_delays(&candidate_pos)?;
            intensity_map[grid_idx] = match mode {
                PamImagingMode::DelayAndSum => self.delay_and_sum_at_point_view(
                    passive_data,
                    &delays_samples,
                    &apodization_weights,
                )?,
                PamImagingMode::DelayMultiplyAndSum => {
                    self.dmas_at_point_view(passive_data, &delays_samples, &apodization_weights)
                }
            };
        }

        Ok(intensity_map)
    }

    /// Delay-and-sum beamformed time series at every candidate grid point.
    ///
    /// Returns an `[n_points × window]` matrix whose row `p` is the coherent
    /// (apodized, fractional-delay-aligned) sum of the receiver traces steered
    /// to grid point `p`. Unlike [`Self::beamform`], which collapses each point
    /// to a scalar intensity, this preserves the per-point time series — the
    /// input for *spectral* passive acoustic mapping, where the broadband
    /// delay-and-sum localizes the source (full bandwidth → fine range
    /// resolution) and a subsequent per-point spectral analysis attributes the
    /// energy to cavitation bands (subharmonic / ultraharmonic).
    ///
    /// # Errors
    /// Propagates input-validation and interpolation errors.
    pub fn beamform_signals_view(
        &self,
        passive_data: ArrayView2<'_, f64>,
        grid_points: ArrayView2<'_, f64>,
    ) -> KwaversResult<Array2<f64>> {
        self.validate_beamform_inputs(passive_data, grid_points)?;
        let apodization = self.compute_apodization_weights();
        let window = self.config.window_size.min(passive_data.shape()[1]).max(1);
        let mut signals = Array2::<f64>::zeros((grid_points.shape()[0], window));
        for (grid_idx, grid_point) in grid_points
            .rows()
            .expect("invariant: rank-2 array has a row axis")
            .enumerate()
        {
            let candidate_pos = [grid_point[0], grid_point[1], grid_point[2]];
            let delays_samples = self.compute_delays(&candidate_pos)?;
            let series =
                self.beamformed_signal_at_point_view(passive_data, &delays_samples, &apodization)?;
            for (t, &value) in series.iter().enumerate() {
                signals[[grid_idx, t]] = value;
            }
        }
        Ok(signals)
    }

    /// Delay-and-sum beamformed time series using an externally supplied
    /// propagation-delay matrix instead of the internal homogeneous-speed model.
    ///
    /// `delays_samples` is `[n_points × n_sensors]`: entry `(p, i)` is the
    /// propagation delay (in samples) from candidate grid point `p` to sensor
    /// `i`. This is the hook for **aberration-corrected** passive acoustic
    /// mapping: the caller supplies heterogeneous-medium travel times (e.g. from
    /// an eikonal solve) so the coherent sum remains aligned through speed
    /// contrasts where the straight-line/constant-speed model would lose
    /// coherence (critical at the higher cavitation bands). Returns an
    /// `[n_points × window]` matrix of per-point beamformed series.
    ///
    /// # Errors
    /// Returns `Err` on shape mismatches (sensor count, delay-matrix columns) or
    /// non-finite values.
    pub fn beamform_signals_with_delays(
        &self,
        passive_data: ArrayView2<'_, f64>,
        delays_samples: ArrayView2<'_, f64>,
    ) -> KwaversResult<Array2<f64>> {
        let [num_sensors_data, num_samples] = passive_data.shape();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }
        if num_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "Passive data must contain at least one time sample".to_owned(),
            ));
        }
        let [num_points, delay_cols] = delays_samples.shape();
        if delay_cols != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Delay matrix has {delay_cols} columns but PAM configured for {} sensors",
                self.num_sensors
            )));
        }
        if num_points == 0 {
            return Err(KwaversError::InvalidInput(
                "Delay matrix must contain at least one candidate point".to_owned(),
            ));
        }
        if !passive_data.iter().all(|v| v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Passive data must contain only finite values".to_owned(),
            ));
        }
        if !delays_samples.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(KwaversError::InvalidInput(
                "Delay matrix must contain only finite, non-negative delays".to_owned(),
            ));
        }

        let apodization = self.compute_apodization_weights();
        let window = self.config.window_size.min(num_samples).max(1);
        let mut signals = Array2::<f64>::zeros((num_points, window));
        for (grid_idx, delay_row) in delays_samples
            .rows()
            .expect("invariant: rank-2 array has a row axis")
            .enumerate()
        {
            let delays: Vec<f64> = delay_row.iter().copied().collect();
            let series =
                self.beamformed_signal_at_point_view(passive_data, &delays, &apodization)?;
            for (t, &value) in series.iter().enumerate() {
                signals[[grid_idx, t]] = value;
            }
        }
        Ok(signals)
    }

    pub(super) fn validate_beamform_inputs(
        &self,
        passive_data: ArrayView2<'_, f64>,
        grid_points: ArrayView2<'_, f64>,
    ) -> KwaversResult<()> {
        let [num_sensors_data, _] = passive_data.shape();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }
        if passive_data.shape()[1] == 0 {
            return Err(KwaversError::InvalidInput(
                "Passive data must contain at least one time sample".to_owned(),
            ));
        }
        if grid_points.shape()[1] != 3 {
            return Err(KwaversError::InvalidInput(format!(
                "Grid points must have shape [points x 3], got {} columns",
                grid_points.shape()[1]
            )));
        }
        if grid_points.shape()[0] == 0 {
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
        let num_samples = passive_data.shape()[1];
        let window_size = self.config.window_size.min(num_samples).max(1);
        let mut summed_signal = vec![0.0; window_size];

        for (sensor_idx, &delay) in delays_samples.iter().enumerate() {
            let weight = apodization[sensor_idx];
            for (t, value) in summed_signal.iter_mut().enumerate().take(window_size) {
                let sample_pos = t as f64 + delay;
                if let Some(interpolated) =
                    Self::interpolate(passive_data, sensor_idx, sample_pos, num_samples)
                {
                    *value += weight * interpolated;
                }
            }
        }

        Ok(summed_signal)
    }

    /// Fractional-delay linear interpolation of `sensor_idx`'s trace at
    /// continuous sample position `sample_pos`. Returns `None` when the position
    /// falls outside the recorded window.
    fn interpolate(
        passive_data: ArrayView2<'_, f64>,
        sensor_idx: usize,
        sample_pos: f64,
        num_samples: usize,
    ) -> Option<f64> {
        if (0.0..=(num_samples - 1) as f64).contains(&sample_pos) {
            let lo = sample_pos.floor() as usize;
            let hi = (lo + 1).min(num_samples - 1);
            let frac = sample_pos - lo as f64;
            Some((1.0 - frac).mul_add(
                passive_data[[sensor_idx, lo]],
                frac * passive_data[[sensor_idx, hi]],
            ))
        } else {
            None
        }
    }

    /// Sign-preserving delay-multiply-and-sum intensity at one candidate point.
    ///
    /// ## Algorithm (Matrone et al. 2015, IEEE TMI 34(4))
    ///
    /// Let `cᵢ(t) = wᵢ · sᵢ(t + τᵢ)` be the apodized, time-aligned sensor sample
    /// and `ŝᵢ = sign(cᵢ)·√|cᵢ|` its sign-preserving square root (which keeps the
    /// product `ŝᵢŝⱼ = sign(cᵢcⱼ)√(|cᵢ||cⱼ|)` dimensionally consistent with a
    /// pressure). The DMAS sample is the sum of all distinct pairwise products,
    /// evaluated in `O(N)` via the closed form
    ///
    /// ```text
    /// y(t) = Σ_{i<j} ŝᵢ(t) ŝⱼ(t) = ½ [ (Σᵢ ŝᵢ)² − Σᵢ ŝᵢ² ].
    /// ```
    ///
    /// The pixel intensity is the windowed energy `⟨y²⟩`. Because each term is a
    /// product of two channels, energy that is coherent across the aperture
    /// (the true source) is reinforced while incoherent off-focus energy — which
    /// has random relative sign between channels — averages toward zero, giving a
    /// narrower mainlobe and lower sidelobes than delay-and-sum.
    fn dmas_at_point_view(
        &self,
        passive_data: ArrayView2<'_, f64>,
        delays_samples: &[f64],
        apodization: &[f64],
    ) -> f64 {
        use crate::signal_processing::beamforming::time_domain::dmas::dmas_combine;

        let num_samples = passive_data.shape()[1];
        let window_size = self.config.window_size.min(num_samples).max(1);
        let mut energy = 0.0_f64;
        // Reused buffer for the apodized, time-aligned aperture column at each t.
        let mut column = Vec::with_capacity(delays_samples.len());

        for t in 0..window_size {
            column.clear();
            for (sensor_idx, &delay) in delays_samples.iter().enumerate() {
                let sample_pos = t as f64 + delay;
                if let Some(interpolated) =
                    Self::interpolate(passive_data, sensor_idx, sample_pos, num_samples)
                {
                    column.push(apodization[sensor_idx] * interpolated);
                }
            }
            // y = Σ_{i<j} ŝᵢŝⱼ (canonical SSOT combiner); intensity accumulates y².
            let dmas_sample = dmas_combine(&column);
            energy += dmas_sample * dmas_sample;
        }

        energy / window_size as f64
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

        let complex_data = fft_1d_array(
            &leto::Array1::from_vec([signal.len()], signal.to_vec()).expect("fft_1d_array signal"),
        );
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
        self.config.apodization.weights(self.num_sensors)
    }

    pub(super) fn noise_threshold(&self, intensity_map: &Array1<f64>) -> f64 {
        let mut sorted: Vec<f64> = intensity_map.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
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
