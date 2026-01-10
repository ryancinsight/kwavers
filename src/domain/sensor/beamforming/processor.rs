//! Beamforming processor

use super::config::BeamformingConfig;
use crate::core::error::KwaversResult;
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2, Array3};

/// Beamforming processor for array algorithms
#[derive(Debug)]
pub struct BeamformingProcessor {
    pub config: BeamformingConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

impl BeamformingProcessor {
    /// Create new beamforming processor
    #[must_use]
    pub fn new(config: BeamformingConfig, sensor_positions: Vec<[f64; 3]>) -> Self {
        let num_sensors = sensor_positions.len();
        Self {
            config,
            sensor_positions,
            num_sensors,
        }
    }
    /// Get number of sensors
    #[must_use]
    pub fn num_sensors(&self) -> usize {
        self.num_sensors
    }

    /// Get sensor positions
    #[must_use]
    pub fn sensor_positions(&self) -> &[[f64; 3]] {
        &self.sensor_positions
    }

    /// Compute eigendecomposition of a matrix
    pub fn eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> KwaversResult<(ndarray::Array1<f64>, Array2<f64>)> {
        LinearAlgebra::eigendecomposition(matrix)
    }

    /// Compute matrix inverse
    pub fn matrix_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        LinearAlgebra::matrix_inverse(matrix)
    }

    /// Compute geometric **propagation delays / time-of-flight (TOF)** (seconds) from each sensor
    /// to a **candidate focal point** using the configured medium sound speed.
    ///
    /// # Field jargon / semantics
    /// - This returns **absolute** TOF values: `τ_i(p) = ||x_i - p|| / c`.
    /// - For time-domain DAS alignment you typically need **relative delays**:
    ///   `Δτ_i(p) = τ_i(p) - τ_ref(p)` (e.g., reference sensor, array centroid, or earliest/ latest).
    ///
    /// # Important
    /// This function does **not** choose a delay reference; it returns absolute TOFs.
    /// Reference selection is a separate policy decision and must be explicit in the caller.
    #[must_use]
    pub fn compute_delays(&self, focal_point: [f64; 3]) -> Vec<f64> {
        let c = self.config.sound_speed;
        self.sensor_positions
            .iter()
            .map(|&pos| {
                let dx = pos[0] - focal_point[0];
                let dy = pos[1] - focal_point[1];
                let dz = pos[2] - focal_point[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                distance / c
            })
            .collect()
    }

    /// Perform **time-domain delay-and-sum (DAS)** using precomputed delays and element weights.
    ///
    /// # Field jargon
    /// - Also known as **conventional beamforming** / **shift-and-sum**.
    /// - This implementation performs **relative-delay alignment to the latest arrival**:
    ///   it subtracts `max(τ)` so that the **latest** channel aligns to `t = 0`.
    ///
    /// # Inputs / invariants
    /// - `sensor_data` shape is expected to be `(n_elements, 1, n_samples)`.
    /// - `weights` length must equal `n_elements`.
    /// - `delays` length must equal `n_elements` and be expressed in seconds.
    ///
    /// # Interpretation note (critical for localization scoring)
    /// If your raw data encodes a transient event at **absolute** arrival times (impulses at `τ_i(p)`),
    /// and you score `∑_t y(t)^2`, then the **choice of delay reference matters**.
    /// For grid-search localization of transients, many pipelines use a fixed reference sensor
    /// (or earliest-arrival) rather than latest-arrival normalization.
    pub fn delay_and_sum_with(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        delays: &[f64],
        weights: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        let (n_elements, _channels, n_samples) = sensor_data.dim();

        if delays.len() != n_elements || weights.len() != n_elements {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Invalid delays/weights: delays={}, weights={}, n_elements={}",
                delays.len(),
                weights.len(),
                n_elements
            )));
        }

        // Delay reference policy (explicit):
        // Align by relative delays so contributions add coherently.
        //
        // Current convention: **latest-arrival alignment** (reference = max delay).
        // That is, we shift each channel by: Δτ_i = τ_ref - τ_i where τ_ref = max_i τ_i.
        //
        // This is a valid beamforming convention, but it is not the only one.
        // For some localization objectives you may want earliest-arrival alignment or a fixed sensor reference.
        let max_delay = delays.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut output = Array3::zeros((1, 1, n_samples));
        for (elem_idx, &delay) in delays.iter().enumerate() {
            let relative_delay = max_delay - delay;
            let delay_samples = (relative_delay * sample_rate).round().max(0.0) as usize;
            if delay_samples < n_samples {
                for it in delay_samples..n_samples {
                    output[[0, 0, it - delay_samples]] +=
                        sensor_data[[elem_idx, 0, it]] * weights[elem_idx];
                }
            }
        }

        Ok(output)
    }

    /// MVDR **adaptive weighting** with diagonal loading using a **uniform steering vector**.
    ///
    /// # Field jargon (be precise)
    /// - MVDR is also called the **Capon beamformer**.
    /// - A true **steered MVDR/Capon spatial spectrum** uses a *look-dependent* steering vector `a(p)` and
    ///   evaluates `P(p) = 1 / (a(p)^H R^{-1} a(p))` (or forms a steered beamformed output).
    ///
    /// # What this function is (and is not)
    /// - This function computes MVDR weights from the sample covariance `R`, but uses a **uniform**
    ///   steering vector (no dependence on look direction / focal point).
    /// - Therefore, it is **not** a steered Capon spatial spectrum and is **not** suitable as-is
    ///   for MVDR/Capon grid-search localization (it lacks point-dependent steering).
    ///
    /// Implementation details:
    /// - Computes a real-valued sample covariance from `(n_elements, 1, n_samples)`.
    /// - Applies diagonal loading `δI` for robustness.
    /// - Applies the resulting weights across time to produce `(1, 1, n_samples)`.
    pub fn mvdr_unsteered_weights_time_series(
        &self,
        sensor_data: &Array3<f64>,
        diagonal_loading: f64,
    ) -> KwaversResult<Array3<f64>> {
        let shape = sensor_data.shape();
        let (n_elements, _channels, n_samples) = (shape[0], shape[1], shape[2]);

        if n_elements < 2 {
            // Fall back to trivial DAS with unit weights
            let weights = vec![1.0; n_elements];
            let delays = vec![0.0; n_elements];
            return self.delay_and_sum_with(
                sensor_data,
                self.config.sampling_frequency,
                &delays,
                &weights,
            );
        }

        // Compute sample covariance R = (1/N) Σ x(n)x^T(n)
        let mut covariance = Array2::zeros((n_elements, n_elements));
        for t in 0..n_samples {
            for i in 0..n_elements {
                for j in 0..n_elements {
                    covariance[[i, j]] += sensor_data[[i, 0, t]] * sensor_data[[j, 0, t]];
                }
            }
        }
        covariance /= n_samples as f64;

        // Diagonal loading: R' = R + δI
        for i in 0..n_elements {
            covariance[[i, i]] += diagonal_loading;
        }

        // Invert covariance
        let inv_cov = self.matrix_inverse(&covariance)?;

        // Uniform steering normalized to unity gain
        let a = Array1::from_vec(vec![1.0 / (n_elements as f64).sqrt(); n_elements]);
        let inv_cov_a = inv_cov.dot(&a);
        let denominator = a.dot(&inv_cov_a);

        if denominator.abs() < 1e-12 {
            // Fallback on numerical issues
            let weights = vec![1.0; n_elements];
            let delays = vec![0.0; n_elements];
            return self.delay_and_sum_with(
                sensor_data,
                self.config.sampling_frequency,
                &delays,
                &weights,
            );
        }

        let weights = inv_cov_a.mapv(|x| x / denominator);

        // Apply weights across time
        let mut output = Array3::<f64>::zeros((1, 1, n_samples));
        for t in 0..n_samples {
            let mut beamformed_value = 0.0;
            for i in 0..n_elements {
                beamformed_value += weights[i] * sensor_data[[i, 0, t]];
            }
            output[[0, 0, t]] = beamformed_value;
        }

        Ok(output)
    }

    /// Deprecated alias for `mvdr_unsteered_weights_time_series`.
    ///
    /// # Deprecation rationale
    /// This function name previously implied a **Capon spatial spectrum**, but it is **not steered**
    /// (it uses a uniform steering vector) and therefore cannot be used as a point-dependent
    /// `P_Capon(p)` scorer for localization.
    #[deprecated(
        since = "2.14.0",
        note = "renamed to mvdr_unsteered_weights_time_series; this is not a steered Capon spatial spectrum"
    )]
    pub fn capon_with_uniform(
        &self,
        sensor_data: &Array3<f64>,
        diagonal_loading: f64,
    ) -> KwaversResult<Array3<f64>> {
        self.mvdr_unsteered_weights_time_series(sensor_data, diagonal_loading)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_processor(n_elements: usize) -> BeamformingProcessor {
        let cfg = BeamformingConfig::default();
        let positions = (0..n_elements).map(|_| [0.0, 0.0, 0.0]).collect();
        BeamformingProcessor::new(cfg, positions)
    }

    #[test]
    fn delay_and_sum_equal_delays_sums_channels() {
        let bf = make_processor(2);

        let sensor0 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sensor1 = [10.0, 20.0, 30.0, 40.0, 50.0];
        let n_samples = sensor0.len();

        let mut data = Array3::<f64>::zeros((2, 1, n_samples));
        for t in 0..n_samples {
            data[[0, 0, t]] = sensor0[t];
            data[[1, 0, t]] = sensor1[t];
        }

        // Equal delays and unit weights should reduce to simple sum
        let delays = vec![0.0; 2];
        let weights = vec![1.0; 2];
        let out = bf
            .delay_and_sum_with(&data, 1.0, &delays, &weights)
            .expect("DAS should succeed");

        for t in 0..n_samples {
            assert!((out[[0, 0, t]] - (sensor0[t] + sensor1[t])).abs() < 1e-12);
        }
    }

    #[test]
    fn capon_uniform_constant_signal_expected_gain() {
        let bf = make_processor(2);

        let n_samples = 8;
        let mut data = Array3::<f64>::zeros((2, 1, n_samples));
        for t in 0..n_samples {
            data[[0, 0, t]] = 1.0;
            data[[1, 0, t]] = 1.0;
        }

        // With diagonal loading 0.1 and constant equal sensors, MVDR weights
        // reduce to approximately uniform 1/sqrt(2), yielding ~sqrt(2) output.
        let out = bf
            .mvdr_unsteered_weights_time_series(&data, 0.1)
            .expect("MVDR should succeed");

        for t in 0..n_samples {
            assert!((out[[0, 0, t]] - std::f64::consts::SQRT_2).abs() < 1e-6);
        }

        // Backward-compat alias: old name should behave identically.
        #[allow(deprecated)]
        let out_alias = bf
            .capon_with_uniform(&data, 0.1)
            .expect("Capon-uniform alias should succeed");
        for t in 0..n_samples {
            assert!((out_alias[[0, 0, t]] - out[[0, 0, t]]).abs() < 1e-12);
        }
    }
}
