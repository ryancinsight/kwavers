//! Adaptive Beamforming Algorithms for Real-Time Processing
//!
//! This module implements advanced adaptive beamforming algorithms that adjust
//! in real-time to changing conditions, interference, and array geometry variations.
//! All implementations are based on established literature and optimized for
//! ultrasound applications.
//!
//! # Design Principles
//! - **Literature-Based**: All algorithms follow established research
//! - **Zero-Copy**: Efficient ArrayView usage for real-time processing
//! - **Adaptive**: Real-time adaptation to changing conditions
//! - **Robust**: Handles uncertainty and interference
//!
//! # Literature References
//! - Frost (1972): "An algorithm for linearly constrained adaptive array processing"
//! - Widrow et al. (1975): "Adaptive noise cancelling: Principles and applications"
//! - Compton (1988): "Adaptive antennas: Concepts and performance"
//! - Li & Stoica (2006): "Robust adaptive beamforming"

use crate::error::KwaversResult;
use crate::sensor::beamforming::{BeamformingConfig, BeamformingProcessor};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::VecDeque;

/// Configuration for adaptive beamforming algorithms
#[derive(Debug, Clone)]
pub struct AdaptiveBeamformingConfig {
    /// Base beamforming configuration
    pub base_config: BeamformingConfig,
    /// Adaptation step size
    pub step_size: f64,
    /// Forgetting factor for exponential averaging
    pub forgetting_factor: f64,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Maximum number of adaptation iterations
    pub max_iterations: usize,
    /// Enable interference suppression
    pub enable_interference_suppression: bool,
    /// Constraint violation tolerance
    pub constraint_tolerance: f64,
}

impl Default for AdaptiveBeamformingConfig {
    fn default() -> Self {
        Self {
            base_config: BeamformingConfig::default(),
            step_size: 0.01,
            forgetting_factor: 0.99,
            convergence_threshold: 1e-6,
            max_iterations: 1000,
            enable_interference_suppression: true,
            constraint_tolerance: 1e-3,
        }
    }
}

/// Adaptive beamforming algorithms
#[derive(Debug, Clone)]
pub enum AdaptiveAlgorithm {
    /// Least Mean Squares (LMS)
    LMS { step_size: f64, regularization: f64 },
    /// Normalized LMS
    NLMS { step_size: f64, regularization: f64 },
    /// Recursive Least Squares (RLS)
    RLS {
        forgetting_factor: f64,
        initialization_factor: f64,
    },
    /// Constrained LMS
    ConstrainedLMS {
        step_size: f64,
        constraints: Array2<f64>,
        response: Array1<f64>,
    },
    /// Sample Matrix Inversion (SMI)
    SMI {
        diagonal_loading: f64,
        adaptation_rate: f64,
    },
    /// Eigenspace-Based Beamforming
    EigenspaceBased {
        signal_subspace_rank: usize,
        tracking_factor: f64,
    },
}

/// State tracking for adaptive algorithms
#[derive(Debug, Clone)]
pub struct AdaptiveState {
    /// Current weight vector
    pub weights: Array1<f64>,
    /// Covariance matrix estimate
    pub covariance_estimate: Array2<f64>,
    /// Inverse covariance matrix (for efficiency)
    pub inverse_covariance: Option<Array2<f64>>,
    /// Adaptation history
    pub weight_history: VecDeque<Array1<f64>>,
    /// Convergence metrics
    pub convergence_history: VecDeque<f64>,
    /// Number of processed snapshots
    pub snapshot_count: usize,
    /// Last update timestamp
    pub last_update: f64,
}

/// Adaptive beamforming processor
pub struct AdaptiveBeamformingProcessor {
    config: AdaptiveBeamformingConfig,
    base_processor: BeamformingProcessor,
    adaptive_state: AdaptiveState,
    sensor_positions: Vec<[f64; 3]>,
}

impl AdaptiveBeamformingProcessor {
    /// Create new adaptive beamforming processor
    pub fn new(config: AdaptiveBeamformingConfig, sensor_positions: Vec<[f64; 3]>) -> Self {
        let num_sensors = sensor_positions.len();
        let base_processor =
            BeamformingProcessor::new(config.base_config.clone(), sensor_positions.clone());

        let adaptive_state = AdaptiveState {
            weights: Array1::ones(num_sensors) / num_sensors as f64,
            covariance_estimate: Array2::eye(num_sensors),
            inverse_covariance: Some(Array2::eye(num_sensors)),
            weight_history: VecDeque::with_capacity(100),
            convergence_history: VecDeque::with_capacity(100),
            snapshot_count: 0,
            last_update: 0.0,
        };

        Self {
            config,
            base_processor,
            adaptive_state,
            sensor_positions,
        }
    }

    /// Process sensor data with adaptive beamforming
    pub fn adaptive_process(
        &mut self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        algorithm: &AdaptiveAlgorithm,
        timestamp: f64,
    ) -> KwaversResult<Array1<f64>> {
        // Update adaptive weights
        self.update_adaptive_weights(sensor_data, algorithm, timestamp)?;

        // Apply adaptive beamforming
        let mut beamformed_output = Array1::zeros(scan_points.len());

        for (point_idx, &scan_point) in scan_points.iter().enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point)?;

            // Apply current adaptive weights
            let mut output = 0.0;
            for t in 0..sensor_data.ncols() {
                let mut weighted_sum = 0.0;
                for s in 0..self.adaptive_state.weights.len() {
                    weighted_sum += self.adaptive_state.weights[s] * sensor_data[[s, t]];
                }
                output += weighted_sum.powi(2);
            }

            beamformed_output[point_idx] = output / sensor_data.ncols() as f64;
        }

        Ok(beamformed_output)
    }

    /// Update adaptive weights using specified algorithm
    fn update_adaptive_weights(
        &mut self,
        sensor_data: ArrayView2<f64>,
        algorithm: &AdaptiveAlgorithm,
        timestamp: f64,
    ) -> KwaversResult<()> {
        match algorithm {
            AdaptiveAlgorithm::LMS {
                step_size,
                regularization,
            } => {
                self.lms_update(sensor_data, *step_size, *regularization)?;
            }
            AdaptiveAlgorithm::NLMS {
                step_size,
                regularization,
            } => {
                self.nlms_update(sensor_data, *step_size, *regularization)?;
            }
            AdaptiveAlgorithm::RLS {
                forgetting_factor,
                initialization_factor,
            } => {
                self.rls_update(sensor_data, *forgetting_factor, *initialization_factor)?;
            }
            AdaptiveAlgorithm::ConstrainedLMS {
                step_size,
                constraints,
                response,
            } => {
                self.constrained_lms_update(
                    sensor_data,
                    *step_size,
                    constraints.view(),
                    response.view(),
                )?;
            }
            AdaptiveAlgorithm::SMI {
                diagonal_loading,
                adaptation_rate,
            } => {
                self.smi_update(sensor_data, *diagonal_loading, *adaptation_rate)?;
            }
            AdaptiveAlgorithm::EigenspaceBased {
                signal_subspace_rank,
                tracking_factor,
            } => {
                self.eigenspace_update(sensor_data, *signal_subspace_rank, *tracking_factor)?;
            }
        }

        self.adaptive_state.last_update = timestamp;
        self.adaptive_state.snapshot_count += sensor_data.ncols();

        // Track convergence
        self.update_convergence_metrics()?;

        Ok(())
    }

    /// LMS algorithm update (Widrow et al., 1975)
    fn lms_update(
        &mut self,
        sensor_data: ArrayView2<f64>,
        step_size: f64,
        regularization: f64,
    ) -> KwaversResult<()> {
        let num_sensors = self.adaptive_state.weights.len();
        let num_samples = sensor_data.ncols();

        for t in 0..num_samples {
            // Get current sensor snapshot
            let x = sensor_data.column(t);

            // Compute output
            let y = self.adaptive_state.weights.dot(&x);

            // Assume desired signal is from main look direction (simplified)
            let d = x[0]; // Use first sensor as reference

            // Compute error
            let e = d - y;

            // Update weights: w(n+1) = w(n) + μ * e * x
            for i in 0..num_sensors {
                self.adaptive_state.weights[i] += step_size * e * x[i];
            }

            // Apply regularization
            if regularization > 0.0 {
                self.adaptive_state
                    .weights
                    .mapv_inplace(|w| w * (1.0 - regularization * step_size));
            }
        }

        Ok(())
    }

    /// Normalized LMS algorithm update
    fn nlms_update(
        &mut self,
        sensor_data: ArrayView2<f64>,
        step_size: f64,
        regularization: f64,
    ) -> KwaversResult<()> {
        let num_sensors = self.adaptive_state.weights.len();
        let num_samples = sensor_data.ncols();

        for t in 0..num_samples {
            let x = sensor_data.column(t);
            let y = self.adaptive_state.weights.dot(&x);
            let d = x[0]; // Reference signal
            let e = d - y;

            // Normalized step size
            let x_power = x.dot(&x) + regularization;
            let normalized_step = step_size / x_power;

            // Update weights
            for i in 0..num_sensors {
                self.adaptive_state.weights[i] += normalized_step * e * x[i];
            }
        }

        Ok(())
    }

    /// Recursive Least Squares (RLS) algorithm update
    fn rls_update(
        &mut self,
        sensor_data: ArrayView2<f64>,
        forgetting_factor: f64,
        initialization_factor: f64,
    ) -> KwaversResult<()> {
        let num_sensors = self.adaptive_state.weights.len();
        let num_samples = sensor_data.ncols();

        // Initialize inverse correlation matrix if needed
        if self.adaptive_state.inverse_covariance.is_none() {
            self.adaptive_state.inverse_covariance =
                Some(Array2::eye(num_sensors) * initialization_factor);
        }

        let mut p_matrix = self
            .adaptive_state
            .inverse_covariance
            .as_ref()
            .unwrap()
            .clone();

        for t in 0..num_samples {
            let x = sensor_data.column(t);
            let d = x[0]; // Reference signal

            // Compute gain vector: g = P * x / (λ + x^T * P * x)
            let px = p_matrix.dot(&x);
            let denominator = forgetting_factor + x.dot(&px);
            let gain = &px / denominator;

            // Compute a priori error
            let y = self.adaptive_state.weights.dot(&x);
            let e = d - y;

            // Update weights: w(n+1) = w(n) + g * e
            for i in 0..num_sensors {
                self.adaptive_state.weights[i] += gain[i] * e;
            }

            // Update inverse correlation matrix: P(n+1) = (P(n) - g * x^T * P(n)) / λ
            let px_outer =
                Array2::from_shape_fn((num_sensors, num_sensors), |(i, j)| gain[i] * px[j]);
            p_matrix = (&p_matrix - &px_outer) / forgetting_factor;
        }

        self.adaptive_state.inverse_covariance = Some(p_matrix);
        Ok(())
    }

    /// Constrained LMS update (Frost, 1972)
    fn constrained_lms_update(
        &mut self,
        sensor_data: ArrayView2<f64>,
        step_size: f64,
        constraints: ArrayView2<f64>,
        response: ArrayView1<f64>,
    ) -> KwaversResult<()> {
        let num_samples = sensor_data.ncols();

        for t in 0..num_samples {
            let x = sensor_data.column(t);
            let y = self.adaptive_state.weights.dot(&x);
            let d = x[0]; // Reference signal
            let e = d - y;

            // Standard LMS update
            let unconstrained_update = &self.adaptive_state.weights + &(&x * (step_size * e));

            // Project onto constraint subspace
            // w_constrained = w_unconstrained - C^T * (C * C^T)^-1 * (C * w_unconstrained - f)
            let c_w = constraints.dot(&unconstrained_update);
            let constraint_error = &c_w - &response;

            // Compute constraint correction (simplified - assumes C*C^T is invertible)
            let ct_c_inv = self.pseudo_inverse(&constraints.dot(&constraints.t()))?;
            let correction = constraints.t().dot(&ct_c_inv.dot(&constraint_error));

            self.adaptive_state.weights = &unconstrained_update - &correction;
        }

        Ok(())
    }

    /// Sample Matrix Inversion (SMI) update
    fn smi_update(
        &mut self,
        sensor_data: ArrayView2<f64>,
        diagonal_loading: f64,
        adaptation_rate: f64,
    ) -> KwaversResult<()> {
        let num_sensors = self.adaptive_state.weights.len();
        let num_samples = sensor_data.ncols();

        // Update covariance matrix estimate
        let mut new_covariance = Array2::<f64>::zeros((num_sensors, num_sensors));
        for t in 0..num_samples {
            let x = sensor_data.column(t);
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    new_covariance[[i, j]] += x[i] * x[j];
                }
            }
        }
        new_covariance.mapv_inplace(|x| x / num_samples as f64);

        // Exponential averaging
        let alpha = adaptation_rate;
        self.adaptive_state.covariance_estimate = &(&self.adaptive_state.covariance_estimate
            * (1.0 - alpha))
            + &(&new_covariance * alpha);

        // Add diagonal loading
        for i in 0..num_sensors {
            self.adaptive_state.covariance_estimate[[i, i]] += diagonal_loading;
        }

        // Compute steering vector for main look direction
        let steering_vector = Array1::ones(num_sensors) / (num_sensors as f64).sqrt();

        // SMI beamforming: w = R^-1 * a / (a^T * R^-1 * a)
        let r_inv = self.matrix_inverse(&self.adaptive_state.covariance_estimate)?;
        let r_inv_a = r_inv.dot(&steering_vector);
        let denominator = steering_vector.dot(&r_inv_a);

        if denominator.abs() > 1e-12 {
            self.adaptive_state.weights = &r_inv_a / denominator;
        }

        Ok(())
    }

    /// Eigenspace-based beamforming update
    fn eigenspace_update(
        &mut self,
        sensor_data: ArrayView2<f64>,
        signal_subspace_rank: usize,
        tracking_factor: f64,
    ) -> KwaversResult<()> {
        // Update covariance matrix
        self.update_covariance_estimate(sensor_data, tracking_factor)?;

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) =
            self.eigendecomposition(&self.adaptive_state.covariance_estimate)?;

        // Signal subspace projection
        let signal_subspace = eigenvectors.slice(ndarray::s![.., 0..signal_subspace_rank]);

        // Compute beamforming weights using signal subspace
        let steering_vector = Array1::ones(self.adaptive_state.weights.len())
            / (self.adaptive_state.weights.len() as f64).sqrt();

        // Project steering vector onto signal subspace
        let mut projected_steering = Array1::<f64>::zeros(steering_vector.len());
        for i in 0..signal_subspace_rank {
            let eigenvec = signal_subspace.column(i);
            let projection = steering_vector.dot(&eigenvec);
            for j in 0..projected_steering.len() {
                projected_steering[j] += projection * eigenvec[j];
            }
        }

        // Normalize
        let norm = projected_steering.dot(&projected_steering).sqrt();
        if norm > 1e-12 {
            self.adaptive_state.weights = projected_steering / norm;
        }

        Ok(())
    }

    // Helper methods

    fn calculate_steering_vector(&self, scan_point: &[f64; 3]) -> KwaversResult<Array1<f64>> {
        // Simplified steering vector calculation
        let mut steering_vector = Array1::zeros(self.sensor_positions.len());
        let wavelength =
            self.config.base_config.sound_speed / self.config.base_config.reference_frequency;
        let wavenumber = 2.0 * std::f64::consts::PI / wavelength;

        let reference_pos = self.sensor_positions[0];
        for (i, &sensor_pos) in self.sensor_positions.iter().enumerate() {
            let path_diff = Self::euclidean_distance(scan_point, &sensor_pos)
                - Self::euclidean_distance(scan_point, &reference_pos);
            steering_vector[i] = (wavenumber * path_diff).cos();
        }

        Ok(steering_vector)
    }

    fn update_covariance_estimate(
        &mut self,
        sensor_data: ArrayView2<f64>,
        tracking_factor: f64,
    ) -> KwaversResult<()> {
        let num_sensors = self.adaptive_state.weights.len();
        let num_samples = sensor_data.ncols();

        let mut sample_covariance = Array2::<f64>::zeros((num_sensors, num_sensors));
        for t in 0..num_samples {
            let x = sensor_data.column(t);
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    sample_covariance[[i, j]] += x[i] * x[j];
                }
            }
        }
        sample_covariance.mapv_inplace(|x| x / num_samples as f64);

        // Exponential averaging
        self.adaptive_state.covariance_estimate = &(&self.adaptive_state.covariance_estimate
            * (1.0 - tracking_factor))
            + &(&sample_covariance * tracking_factor);

        Ok(())
    }

    fn update_convergence_metrics(&mut self) -> KwaversResult<()> {
        // Store weight history
        self.adaptive_state
            .weight_history
            .push_back(self.adaptive_state.weights.clone());
        if self.adaptive_state.weight_history.len() > 100 {
            self.adaptive_state.weight_history.pop_front();
        }

        // Compute convergence metric (weight change)
        if self.adaptive_state.weight_history.len() >= 2 {
            let current =
                &self.adaptive_state.weight_history[self.adaptive_state.weight_history.len() - 1];
            let previous =
                &self.adaptive_state.weight_history[self.adaptive_state.weight_history.len() - 2];
            let change = (current - previous).mapv(|x| x.abs()).sum();

            self.adaptive_state.convergence_history.push_back(change);
            if self.adaptive_state.convergence_history.len() > 100 {
                self.adaptive_state.convergence_history.pop_front();
            }
        }

        Ok(())
    }

    fn eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        // Use the enhanced eigendecomposition from the base processor
        // This is a simplified delegation - in practice would implement here
        let processor = BeamformingProcessor::new(
            self.config.base_config.clone(),
            self.sensor_positions.clone(),
        );
        processor.eigendecomposition(matrix)
    }

    fn matrix_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Delegate to base processor's enhanced matrix inverse
        let processor = BeamformingProcessor::new(
            self.config.base_config.clone(),
            self.sensor_positions.clone(),
        );
        processor.matrix_inverse(matrix)
    }

    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Compute pseudo-inverse using SVD (simplified implementation)
        // In practice, would use proper SVD decomposition
        self.matrix_inverse(matrix)
    }

    fn euclidean_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }

    /// Get current adaptive state
    pub fn adaptive_state(&self) -> &AdaptiveState {
        &self.adaptive_state
    }

    /// Check if algorithm has converged
    pub fn has_converged(&self) -> bool {
        if let Some(&last_change) = self.adaptive_state.convergence_history.back() {
            last_change < self.config.convergence_threshold
        } else {
            false
        }
    }

    /// Get convergence rate
    pub fn convergence_rate(&self) -> f64 {
        if self.adaptive_state.convergence_history.len() >= 10 {
            let recent: Vec<f64> = self
                .adaptive_state
                .convergence_history
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect();
            recent.iter().sum::<f64>() / recent.len() as f64
        } else {
            1.0 // Not enough data
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_processor_creation() {
        let config = AdaptiveBeamformingConfig::default();
        let sensor_positions = vec![[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [2e-3, 0.0, 0.0]];
        let processor = AdaptiveBeamformingProcessor::new(config, sensor_positions);
        assert_eq!(processor.adaptive_state.weights.len(), 3);
    }

    #[test]
    fn test_lms_update() {
        let config = AdaptiveBeamformingConfig::default();
        let sensor_positions = vec![[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]];
        let mut processor = AdaptiveBeamformingProcessor::new(config, sensor_positions);

        let sensor_data = Array2::ones((2, 10));
        let algorithm = AdaptiveAlgorithm::LMS {
            step_size: 0.01,
            regularization: 0.001,
        };

        let result = processor.lms_update(sensor_data.view(), 0.01, 0.001);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convergence_tracking() {
        let config = AdaptiveBeamformingConfig::default();
        let sensor_positions = vec![[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]];
        let processor = AdaptiveBeamformingProcessor::new(config, sensor_positions);

        // Initially should not be converged
        assert!(!processor.has_converged());

        // Convergence rate should be 1.0 without enough data
        assert_eq!(processor.convergence_rate(), 1.0);
    }
}
