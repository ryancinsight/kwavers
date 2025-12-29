//! Beamforming processor

use super::config::BeamformingConfig;
use crate::error::KwaversResult;
use crate::utils::linear_algebra::LinearAlgebra;
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

    /// Compute geometric propagation delays (seconds) from sensors to a focal point
    /// using the configured medium sound speed.
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

    /// Perform delay-and-sum using precomputed delays and element weights.
    ///
    /// - `sensor_data` shape is expected to be `(n_elements, 1, n_samples)`.
    /// - `weights` length must equal `n_elements`.
    /// - `delays` length must equal `n_elements` and be expressed in seconds.
    pub fn delay_and_sum_with(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        delays: &[f64],
        weights: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        let (n_elements, _channels, n_samples) = sensor_data.dim();

        if delays.len() != n_elements || weights.len() != n_elements {
            return Err(crate::error::KwaversError::InvalidInput(format!(
                "Invalid delays/weights: delays={}, weights={}, n_elements={}",
                delays.len(),
                weights.len(),
                n_elements
            )));
        }

        // Align by relative delays so contributions add coherently
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

    /// Capon (MVDR) beamforming with diagonal loading using a uniform steering vector.
    ///
    /// This computes the sample covariance, applies diagonal loading for robustness,
    /// obtains MVDR weights, and applies them across the time series.
    pub fn capon_with_uniform(
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
            .capon_with_uniform(&data, 0.1)
            .expect("Capon should succeed");

        for t in 0..n_samples {
            assert!((out[[0, 0, t]] - std::f64::consts::SQRT_2).abs() < 1e-6);
        }
    }
}
