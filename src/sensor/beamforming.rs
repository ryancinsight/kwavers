//! Beamforming Algorithms for Ultrasound Arrays
//!
//! This module implements state-of-the-art beamforming algorithms for ultrasound
//! imaging and passive acoustic mapping, following established literature and
//! designed for large-scale array processing.
//!
//! # Design Principles
//! - **Literature-Based**: All algorithms follow established papers
//! - **Zero-Copy**: Efficient ArrayView usage throughout
//! - **Sparse Operations**: Designed for large arrays with sparse matrices
//! - **Modular Design**: Plugin-compatible architecture
//!
//! # Literature References
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Li et al. (2003): "Robust Capon beamforming"
//! - Schmidt (1986): "Multiple emitter location and signal parameter estimation"
//! - Capon (1969): "High-resolution frequency-wavenumber spectrum analysis"
//! - Frost (1972): "An algorithm for linearly constrained adaptive array processing"

use crate::error::KwaversResult;
use crate::utils::sparse_matrix::{CompressedSparseRowMatrix, BeamformingMatrixOperations};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Solve, Inverse, Eig};
use std::f64::consts::PI;

/// Configuration for beamforming operations
#[derive(Debug, Clone)]
pub struct BeamformingConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Reference frequency for array design (Hz)
    pub reference_frequency: f64,
    /// Diagonal loading factor for regularization
    pub diagonal_loading: f64,
    /// Number of snapshots for covariance estimation
    pub num_snapshots: usize,
    /// Spatial smoothing factor
    pub spatial_smoothing: Option<usize>,
}

impl Default for BeamformingConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1540.0, // Typical soft tissue value
            sampling_frequency: 40e6, // 40 MHz
            reference_frequency: 5e6, // 5 MHz
            diagonal_loading: 0.01, // 1% diagonal loading
            num_snapshots: 100,
            spatial_smoothing: None,
        }
    }
}

/// Beamforming algorithm types with literature-based implementations
#[derive(Debug, Clone)]
pub enum BeamformingAlgorithm {
    /// Delay-and-Sum (conventional beamforming)
    DelaySum,
    /// Minimum Variance Distortionless Response (MVDR/Capon)
    MVDR { 
        diagonal_loading: f64,
        spatial_smoothing: bool,
    },
    /// MUltiple SIgnal Classification
    MUSIC { 
        signal_subspace_dimension: usize,
        spatial_smoothing: bool,
    },
    /// Capon Beamforming with Regularization
    RobustCapon {
        diagonal_loading: f64,
        uncertainty_set_size: f64,
    },
    /// Linearly Constrained Minimum Variance (LCMV)
    LCMV {
        constraint_matrix: Array2<f64>,
        response_vector: Array1<f64>,
    },
    /// Generalized Sidelobe Canceller (GSC)
    GSC {
        main_beam_weight: f64,
        adaptation_step_size: f64,
    },
    /// Compressive Beamforming
    Compressive {
        sparsity_parameter: f64,
        dictionary_size: usize,
    },
}

/// Steering vector calculation methods
#[derive(Debug, Clone)]
pub enum SteeringVectorMethod {
    /// Far-field plane wave assumption
    PlaneWave,
    /// Near-field spherical wave
    SphericalWave,
    /// Focused beam
    Focused { focal_point: [f64; 3] },
}

/// Beamforming processor for array algorithms
#[derive(Debug)]
pub struct BeamformingProcessor {
    pub config: BeamformingConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

impl BeamformingProcessor {
    /// Create new beamforming processor
    pub fn new(config: BeamformingConfig, sensor_positions: Vec<[f64; 3]>) -> Self {
        let num_sensors = sensor_positions.len();
        Self {
            config,
            sensor_positions,
            num_sensors,
        }
    }

    /// Process sensor data with specified beamforming algorithm
    pub fn process(
        &self,
        sensor_data: ArrayView2<f64>, // [sensors x time_samples]
        scan_points: &[[f64; 3]],
        algorithm: &BeamformingAlgorithm,
    ) -> KwaversResult<Array1<f64>> {
        match algorithm {
            BeamformingAlgorithm::DelaySum => {
                self.delay_sum_beamforming(sensor_data, scan_points)
            }
            BeamformingAlgorithm::MVDR { diagonal_loading, spatial_smoothing } => {
                self.mvdr_beamforming(sensor_data, scan_points, *diagonal_loading, *spatial_smoothing)
            }
            BeamformingAlgorithm::MUSIC { signal_subspace_dimension, spatial_smoothing } => {
                self.music_beamforming(sensor_data, scan_points, *signal_subspace_dimension, *spatial_smoothing)
            }
            BeamformingAlgorithm::RobustCapon { diagonal_loading, uncertainty_set_size } => {
                self.capon_beamforming(sensor_data, scan_points, *diagonal_loading, *uncertainty_set_size)
            }
            BeamformingAlgorithm::LCMV { constraint_matrix, response_vector } => {
                self.lcmv_beamforming(sensor_data, scan_points, constraint_matrix.view(), response_vector.view())
            }
            BeamformingAlgorithm::GSC { main_beam_weight, adaptation_step_size } => {
                self.gsc_beamforming(sensor_data, scan_points, *main_beam_weight, *adaptation_step_size)
            }
            BeamformingAlgorithm::Compressive { sparsity_parameter, dictionary_size } => {
                self.compressive_beamforming(sensor_data, scan_points, *sparsity_parameter, *dictionary_size)
            }
        }
    }

    /// Delay-and-Sum beamforming (Van Veen & Buckley, 1988)
    pub fn delay_sum_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
    ) -> KwaversResult<Array1<f64>> {
        let mut beamformed_output = Array1::zeros(scan_points.len());
        let dt = 1.0 / self.config.sampling_frequency;
        
        for (point_idx, &scan_point) in scan_points.iter().enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave)?;
            let delays = self.calculate_time_delays(&scan_point)?;
            
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for (sensor_idx, &delay) in delays.iter().enumerate() {
                let delay_samples = (delay / dt).round() as isize;
                let weight = steering_vector[sensor_idx];
                
                // Sum over time samples with delay compensation
                for t in 0..sensor_data.ncols() {
                    let delayed_t = (t as isize - delay_samples).max(0) as usize;
                    if delayed_t < sensor_data.ncols() {
                        sum += weight * sensor_data[[sensor_idx, delayed_t]];
                        weight_sum += weight.abs();
                    }
                }
            }
            
            beamformed_output[point_idx] = if weight_sum > 1e-12 { sum / weight_sum } else { 0.0 };
        }
        
        Ok(beamformed_output)
    }

    /// MVDR/Capon beamforming (Li et al., 2003)
    pub fn mvdr_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        diagonal_loading: f64,
        spatial_smoothing: bool,
    ) -> KwaversResult<Array1<f64>> {
        // Estimate covariance matrix
        let covariance = self.estimate_covariance_matrix(sensor_data, spatial_smoothing)?;
        
        // Apply diagonal loading for regularization
        let mut regularized_cov = covariance.clone();
        for i in 0..self.num_sensors {
            regularized_cov[[i, i]] += diagonal_loading * regularized_cov[[i, i]];
        }
        
        // Compute MVDR weights for each scan point
        let mut beamformed_output = Array1::zeros(scan_points.len());
        
        for (point_idx, &scan_point) in scan_points.iter().enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave)?;
            
            // Solve: R^-1 * a for MVDR weights
            let weights = self.solve_linear_system(&regularized_cov, &steering_vector)?;
            
            // Normalize weights: w = R^-1 * a / (a^H * R^-1 * a)
            let denominator = steering_vector.dot(&weights);
            let normalized_weights = if denominator.abs() > 1e-12 {
                &weights / denominator
            } else {
                weights
            };
            
            // Apply weights to sensor data
            let mut output = 0.0;
            for t in 0..sensor_data.ncols() {
                let mut weighted_sum = 0.0;
                for s in 0..self.num_sensors {
                    weighted_sum += normalized_weights[s] * sensor_data[[s, t]];
                }
                output += weighted_sum.powi(2);
            }
            
            beamformed_output[point_idx] = output / sensor_data.ncols() as f64;
        }
        
        Ok(beamformed_output)
    }

    /// MUSIC algorithm (Schmidt, 1986)
    pub fn music_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        signal_subspace_dimension: usize,
        spatial_smoothing: bool,
    ) -> KwaversResult<Array1<f64>> {
        // Estimate covariance matrix
        let covariance = self.estimate_covariance_matrix(sensor_data, spatial_smoothing)?;
        
        // Eigendecomposition of covariance matrix
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&covariance)?;
        
        // Determine noise subspace (smallest eigenvalues)
        let noise_subspace_start = signal_subspace_dimension;
        let noise_subspace = eigenvectors.slice(ndarray::s![.., noise_subspace_start..]);
        
        // Compute MUSIC spectrum for each scan point
        let mut music_spectrum = Array1::zeros(scan_points.len());
        
        for (point_idx, &scan_point) in scan_points.iter().enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave)?;
            
            // MUSIC pseudospectrum: 1 / (a^H * P_n * P_n^H * a)
            let mut denominator = 0.0;
            for col in noise_subspace.columns() {
                let projection = steering_vector.dot(&col.to_owned());
                denominator += projection.powi(2);
            }
            
            music_spectrum[point_idx] = if denominator > 1e-12 { 1.0 / denominator } else { 0.0 };
        }
        
        Ok(music_spectrum)
    }

    /// Capon Beamforming with Regularization (Li et al., 2003)
    pub fn capon_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        diagonal_loading: f64,
        uncertainty_set_size: f64,
    ) -> KwaversResult<Array1<f64>> {
        // Estimate covariance matrix
        let covariance = self.estimate_covariance_matrix(sensor_data, false)?;
        
        let mut beamformed_output = Array1::zeros(scan_points.len());
        
        for (point_idx, &scan_point) in scan_points.iter().enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave)?;
            
            // Capon formulation with uncertainty set
            let identity = Array2::<f64>::eye(self.num_sensors);
            let uncertainty_matrix = &identity * uncertainty_set_size;
            
            // Modified covariance: R + ε * I + δ * (I - aa^H/||a||^2)
            let mut regularized_cov = covariance.clone();
            for i in 0..self.num_sensors {
                                  regularized_cov[[i, i]] += diagonal_loading;
            }
            
            // Add uncertainty constraint
            let norm_sq = steering_vector.dot(&steering_vector);
            if norm_sq > 1e-12 {
                for i in 0..self.num_sensors {
                    for j in 0..self.num_sensors {
                        let projection_term = steering_vector[i] * steering_vector[j] / norm_sq;
                        regularized_cov[[i, j]] += uncertainty_set_size * (if i == j { 1.0 } else { 0.0 } - projection_term);
                    }
                }
            }
            
            // Solve for regularized weights
            let weights = self.solve_linear_system(&regularized_cov, &steering_vector)?;
            let denominator = steering_vector.dot(&weights);
            let normalized_weights = if denominator.abs() > 1e-12 {
                &weights / denominator
            } else {
                weights
            };
            
            // Apply weights to compute output power
            let mut output = 0.0;
            for t in 0..sensor_data.ncols() {
                let mut weighted_sum = 0.0;
                for s in 0..self.num_sensors {
                    weighted_sum += normalized_weights[s] * sensor_data[[s, t]];
                }
                output += weighted_sum.powi(2);
            }
            
            beamformed_output[point_idx] = output / sensor_data.ncols() as f64;
        }
        
        Ok(beamformed_output)
    }

    /// Linearly Constrained Minimum Variance (LCMV) beamforming (Frost, 1972)
    pub fn lcmv_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        constraint_matrix: ArrayView2<f64>,
        response_vector: ArrayView1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        // Estimate covariance matrix
        let covariance = self.estimate_covariance_matrix(sensor_data, false)?;
        
        // LCMV solution: w = R^-1 * C * (C^H * R^-1 * C)^-1 * f
        let r_inv_c = self.matrix_multiply(&self.matrix_inverse(&covariance)?, &constraint_matrix.to_owned())?;
        let c_h_r_inv_c = constraint_matrix.t().to_owned().dot(&r_inv_c);
        let c_h_r_inv_c_inv = self.matrix_inverse(&c_h_r_inv_c)?;
        
        let mut beamformed_output = Array1::zeros(scan_points.len());
        
        for (point_idx, &_scan_point) in scan_points.iter().enumerate() {
            // For simplicity, using the same constraint for all points
            // In practice, would vary constraints based on scan point
            let intermediate = c_h_r_inv_c_inv.dot(&response_vector);
            let weights = r_inv_c.dot(&intermediate);
            
            // Apply weights to sensor data
            let mut output = 0.0;
            for t in 0..sensor_data.ncols() {
                let mut weighted_sum = 0.0;
                for s in 0..self.num_sensors {
                    weighted_sum += weights[s] * sensor_data[[s, t]];
                }
                output += weighted_sum.powi(2);
            }
            
            beamformed_output[point_idx] = output / sensor_data.ncols() as f64;
        }
        
        Ok(beamformed_output)
    }

    /// Generalized Sidelobe Canceller (GSC) beamforming
    pub fn gsc_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        main_beam_weight: f64,
        adaptation_step_size: f64,
    ) -> KwaversResult<Array1<f64>> {
        let mut beamformed_output = Array1::zeros(scan_points.len());
        
        for (point_idx, &scan_point) in scan_points.iter().enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave)?;
            
            // Fixed beamformer (main beam)
            let fixed_weights = &steering_vector * main_beam_weight;
            
            // Blocking matrix (orthogonal to steering vector)
            let blocking_matrix = self.construct_blocking_matrix(&steering_vector)?;
            
            // Adaptive weights using Normalized LMS (NLMS) algorithm
            // Based on Haykin (2002): "Adaptive Filter Theory"
            let mut adaptive_weights = Array1::<f64>::zeros(self.num_sensors - 1);
            let step_size = 0.01; // NLMS step size
            let regularization = 1e-6; // Regularization parameter
            
            let mut output = 0.0;
            for t in 0..sensor_data.ncols() {
                // Fixed beamformer output
                let mut fixed_output = 0.0;
                for s in 0..self.num_sensors {
                    fixed_output += fixed_weights[s] * sensor_data[[s, t]];
                }
                
                // Blocked signals
                let mut blocked_signals = Array1::zeros(self.num_sensors - 1);
                for i in 0..(self.num_sensors - 1) {
                    for s in 0..self.num_sensors {
                        blocked_signals[i] += blocking_matrix[[i, s]] * sensor_data[[s, t]];
                    }
                }
                
                // Adaptive cancellation
                let adaptive_output = adaptive_weights.dot(&blocked_signals);
                let gsc_output: f64 = fixed_output - adaptive_output;
                
                // NLMS adaptation with normalization
                let signal_power = blocked_signals.dot(&blocked_signals) + regularization;
                let normalized_step = step_size / signal_power;
                
                for i in 0..(self.num_sensors - 1) {
                    adaptive_weights[i] += normalized_step * gsc_output * blocked_signals[i];
                }
                
                output += gsc_output.powi(2);
            }
            
            beamformed_output[point_idx] = output / sensor_data.ncols() as f64;
        }
        
        Ok(beamformed_output)
    }

    /// Compressive beamforming using sparse reconstruction
    pub fn compressive_beamforming(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        sparsity_parameter: f64,
        dictionary_size: usize,
    ) -> KwaversResult<Array1<f64>> {
        // Build dictionary matrix (steering vectors for all potential source locations)
        let dictionary = self.build_dictionary_matrix(scan_points, dictionary_size)?;
        
        // Sparse reconstruction for each time sample
        let mut beamformed_output = Array1::zeros(scan_points.len());
        
        for t in 0..sensor_data.ncols() {
            let measurement_vector = sensor_data.column(t);
            
            // Solve sparse reconstruction: min ||x||_1 subject to ||Ax - y||_2 < ε
            let sparse_solution = self.solve_sparse_reconstruction(
                &dictionary,
                measurement_vector,
                sparsity_parameter,
            )?;
            
            // Accumulate sparse solution
            for (i, &value) in sparse_solution.iter().enumerate() {
                if i < beamformed_output.len() {
                    beamformed_output[i] += value.abs();
                }
            }
        }
        
        // Normalize by number of time samples
        beamformed_output.mapv_inplace(|x| x / sensor_data.ncols() as f64);
        
        Ok(beamformed_output)
    }

    // Helper methods for beamforming operations

    /// Calculate steering vector for given scan point
    fn calculate_steering_vector(
        &self,
        scan_point: &[f64; 3],
        method: SteeringVectorMethod,
    ) -> KwaversResult<Array1<f64>> {
        let mut steering_vector = Array1::zeros(self.num_sensors);
        let wavelength = self.config.sound_speed / self.config.reference_frequency;
        let wavenumber = 2.0 * PI / wavelength;
        
        match method {
            SteeringVectorMethod::PlaneWave => {
                // Far-field plane wave assumption
                let reference_pos = self.sensor_positions[0];
                for (i, &sensor_pos) in self.sensor_positions.iter().enumerate() {
                    let path_diff = Self::euclidean_distance(scan_point, &sensor_pos) 
                                  - Self::euclidean_distance(scan_point, &reference_pos);
                    steering_vector[i] = (wavenumber * path_diff).cos();
                }
            }
            SteeringVectorMethod::SphericalWave => {
                // Near-field spherical wave
                for (i, &sensor_pos) in self.sensor_positions.iter().enumerate() {
                    let distance = Self::euclidean_distance(scan_point, &sensor_pos);
                    let phase = wavenumber * distance;
                    steering_vector[i] = phase.cos() / distance;
                }
            }
            SteeringVectorMethod::Focused { focal_point } => {
                // Focused beam steering
                for (i, &sensor_pos) in self.sensor_positions.iter().enumerate() {
                    let focus_distance = Self::euclidean_distance(&focal_point, &sensor_pos);
                    let scan_distance = Self::euclidean_distance(scan_point, &sensor_pos);
                    let path_diff = scan_distance - focus_distance;
                    steering_vector[i] = (wavenumber * path_diff).cos();
                }
            }
        }
        
        Ok(steering_vector)
    }

    /// Calculate time delays for delay-and-sum beamforming
    fn calculate_time_delays(&self, scan_point: &[f64; 3]) -> KwaversResult<Array1<f64>> {
        let mut delays = Array1::zeros(self.num_sensors);
        let reference_distance = Self::euclidean_distance(scan_point, &self.sensor_positions[0]);
        
        for (i, &sensor_pos) in self.sensor_positions.iter().enumerate() {
            let distance = Self::euclidean_distance(scan_point, &sensor_pos);
            delays[i] = (distance - reference_distance) / self.config.sound_speed;
        }
        
        Ok(delays)
    }

    /// Estimate covariance matrix from sensor data
    fn estimate_covariance_matrix(
        &self,
        sensor_data: ArrayView2<f64>,
        spatial_smoothing: bool,
    ) -> KwaversResult<Array2<f64>> {
        let num_snapshots = sensor_data.ncols().min(self.config.num_snapshots);
        let mut covariance = Array2::zeros((self.num_sensors, self.num_sensors));
        
        if spatial_smoothing {
            // Forward-backward spatial smoothing for estimation
            let smoothing_factor = self.config.spatial_smoothing.unwrap_or(1);
            let effective_sensors = self.num_sensors - smoothing_factor + 1;
            
            for snapshot in 0..num_snapshots {
                for sub_array in 0..smoothing_factor {
                    let start_sensor = sub_array;
                    let end_sensor = start_sensor + effective_sensors;
                    
                    for i in start_sensor..end_sensor {
                        for j in start_sensor..end_sensor {
                            let forward = sensor_data[[i, snapshot]] * sensor_data[[j, snapshot]];
                            let backward = sensor_data[[i, snapshot]] * sensor_data[[j, snapshot]];
                            covariance[[i - start_sensor, j - start_sensor]] += (forward + backward) / 2.0;
                        }
                    }
                }
            }
            
            covariance.mapv_inplace(|x| x / (num_snapshots * smoothing_factor) as f64);
        } else {
            // Standard covariance estimation
            for snapshot in 0..num_snapshots {
                for i in 0..self.num_sensors {
                    for j in 0..self.num_sensors {
                        covariance[[i, j]] += sensor_data[[i, snapshot]] * sensor_data[[j, snapshot]];
                    }
                }
            }
            
            covariance.mapv_inplace(|x| x / num_snapshots as f64);
        }
        
        Ok(covariance)
    }

    /// Eigendecomposition using robust LAPACK-based algorithms
    /// Replaces the naive power iteration method for better numerical stability
    pub fn eigendecomposition(&self, matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        // Use ndarray-linalg for robust eigendecomposition
        let (eigenvalues, eigenvectors) = matrix.eig()
            .map_err(|e| crate::error::KwaversError::Numerical(
                crate::error::NumericalError::Instability {
                    operation: "eigendecomposition".to_string(),
                    condition: format!("LAPACK eigendecomposition failed: {}", e),
                }
            ))?;
        
        // Convert complex eigenvalues to real (assuming real symmetric matrix)
        let real_eigenvalues = Array1::from_vec(
            eigenvalues.iter()
                .map(|&c| c.re) // Take real part
                .collect()
        );
        
        // Convert complex eigenvectors to real
        let real_eigenvectors = Array2::from_shape_vec(
            eigenvectors.dim(),
            eigenvectors.iter()
                .map(|&c| c.re) // Take real part
                .collect()
        ).map_err(|e| crate::error::KwaversError::Numerical(
            crate::error::NumericalError::Instability {
                operation: "eigendecomposition".to_string(),
                condition: format!("Eigenvector conversion failed: {}", e),
            }
        ))?;
        
        // Sort eigenvalues and eigenvectors in descending order
        let n = real_eigenvalues.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| real_eigenvalues[b].partial_cmp(&real_eigenvalues[a]).unwrap());
        
        let sorted_eigenvalues = Array1::from_vec(indices.iter().map(|&i| real_eigenvalues[i]).collect());
        let mut sorted_eigenvectors = Array2::zeros((n, n));
        for (j, &i) in indices.iter().enumerate() {
            for row in 0..n {
                sorted_eigenvectors[[row, j]] = real_eigenvectors[[row, i]];
            }
        }
        
        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }

    /// Solve linear system Ax = b using robust LAPACK-based algorithms
    /// Replaces naive Gaussian elimination for better numerical stability
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        // Use ndarray-linalg for robust linear system solving
        let x = a.solve_into(b.clone())
            .map_err(|e| crate::error::KwaversError::Numerical(
                crate::error::NumericalError::Instability {
                    operation: "linear_system_solve".to_string(),
                    condition: format!("LAPACK solver failed: {}", e),
                }
            ))?;
        
        Ok(x)
    }

    /// Matrix inverse using robust LAPACK-based algorithms
    /// Replaces naive Gauss-Jordan elimination for better numerical stability
    pub fn matrix_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Use ndarray-linalg for robust matrix inversion
        let inverse = matrix.inv()
            .map_err(|e| crate::error::KwaversError::Numerical(
                crate::error::NumericalError::Instability {
                    operation: "matrix_inverse".to_string(),
                    condition: format!("LAPACK matrix inversion failed: {}", e),
                }
            ))?;
        
        Ok(inverse)
    }

    /// Matrix multiplication helper
    pub fn matrix_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        Ok(a.dot(b))
    }

    /// Construct blocking matrix orthogonal to steering vector using Gram-Schmidt
    fn construct_blocking_matrix(&self, steering_vector: &Array1<f64>) -> KwaversResult<Array2<f64>> {
        let n = self.num_sensors;
        let mut blocking_matrix = Array2::zeros((n - 1, n));
        
        // Normalize steering vector
        let norm = steering_vector.dot(steering_vector).sqrt();
        if norm < 1e-12 {
            return Err(crate::error::KwaversError::Numerical(
                crate::error::NumericalError::DivisionByZero {
                    operation: "construct_blocking_matrix".to_string(),
                    location: "steering_vector_normalization".to_string(),
                }
            ));
        }
        let normalized_steering = steering_vector / norm;
        
        // Generate orthonormal basis using Gram-Schmidt process
        let mut basis_vectors = Vec::new();
        
        // Start with standard basis vectors
        for i in 0..n {
            let mut candidate = Array1::zeros(n);
            candidate[i] = 1.0;
            
            // Project out steering vector
            let projection = candidate.dot(&normalized_steering);
            candidate = &candidate - &(&normalized_steering * projection);
            
            // Project out previous basis vectors
            for basis_vec in &basis_vectors {
                let projection = candidate.dot(basis_vec);
                candidate = &candidate - &(basis_vec * projection);
            }
            
            // Normalize
            let norm = candidate.dot(&candidate).sqrt();
            if norm > 1e-12 {
                candidate.mapv_inplace(|x| x / norm);
                basis_vectors.push(candidate);
                
                if basis_vectors.len() >= n - 1 {
                    break;
                }
            }
        }
        
        // Fill blocking matrix with orthogonal vectors
        for (i, basis_vec) in basis_vectors.iter().enumerate().take(n - 1) {
            for j in 0..n {
                blocking_matrix[[i, j]] = basis_vec[j];
            }
        }
        
        Ok(blocking_matrix)
    }

    /// Build dictionary matrix for compressive beamforming
    fn build_dictionary_matrix(&self, scan_points: &[[f64; 3]], dictionary_size: usize) -> KwaversResult<Array2<f64>> {
        let actual_size = dictionary_size.min(scan_points.len());
        let mut dictionary = Array2::zeros((self.num_sensors, actual_size));
        
        for (col, &scan_point) in scan_points.iter().take(actual_size).enumerate() {
            let steering_vector = self.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave)?;
            for row in 0..self.num_sensors {
                dictionary[[row, col]] = steering_vector[row];
            }
        }
        
        Ok(dictionary)
    }

    /// Solve sparse reconstruction using ISTA (Iterative Soft-Thresholding Algorithm)
    /// Based on Beck & Teboulle (2009): "Iterative Shrinkage-Thresholding Algorithm"
    fn solve_sparse_reconstruction(
        &self,
        dictionary: &Array2<f64>,
        measurement: ndarray::ArrayView1<f64>,
        sparsity_parameter: f64,
    ) -> KwaversResult<Array1<f64>> {
        use crate::constants::tolerance::CONVERGENCE;
        
        // ISTA algorithm parameters based on literature
        let max_iterations = 1000;
        let tolerance = CONVERGENCE;
        
        // Compute Lipschitz constant L = ||A^T A||_2 for step size
        let ata = dictionary.t().dot(dictionary);
        let lipschitz_constant = self.estimate_spectral_norm(&ata);
        let step_size = 0.99 / lipschitz_constant; // Conservative step size
        
        let mut solution = Array1::zeros(dictionary.ncols());
        let mut prev_solution = solution.clone();
        
        for iteration in 0..max_iterations {
            prev_solution.assign(&solution);
            
            // Gradient descent step: x = x - t * A^T(Ax - b)
            let residual = &dictionary.dot(&solution) - &measurement.to_owned();
            let gradient = dictionary.t().dot(&residual);
            solution = &solution - &(&gradient * step_size);
            
            // Proximal operator: soft thresholding
            let threshold = sparsity_parameter * step_size;
            solution.mapv_inplace(|x| self.soft_threshold(x, threshold));
            
            // Check convergence
            let change = (&solution - &prev_solution).mapv(|x| x.abs()).sum();
            if change < tolerance {
                break;
            }
            
            // Progress logging for long runs
            if iteration % 100 == 0 && iteration > 0 {
                let objective = self.compute_lasso_objective(dictionary, &measurement.to_owned(), &solution, sparsity_parameter);
                log::debug!("ISTA iteration {}: objective = {:.6e}, change = {:.6e}", iteration, objective, change);
            }
        }
        
        Ok(solution)
    }
    
    /// Soft thresholding operator for LASSO
    fn soft_threshold(&self, x: f64, threshold: f64) -> f64 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }
    
    /// Estimate spectral norm using power iteration
    fn estimate_spectral_norm(&self, matrix: &Array2<f64>) -> f64 {
        let n = matrix.ncols();
        let mut v = Array1::ones(n) / (n as f64).sqrt();
        let max_iter = 50;
        let tolerance = 1e-6;
        
        for _ in 0..max_iter {
            let av = matrix.dot(&v);
            let norm = av.dot(&av).sqrt();
            let new_v = &av / norm;
            
            let change = (&new_v - &v).mapv(|x| x.abs()).sum();
            v = new_v;
            
            if change < tolerance {
                break;
            }
        }
        
        let av = matrix.dot(&v);
        av.dot(&av).sqrt()
    }
    
    /// Compute LASSO objective: ||Ax - b||^2 / 2 + λ||x||_1
    fn compute_lasso_objective(&self, a: &Array2<f64>, b: &Array1<f64>, x: &Array1<f64>, lambda: f64) -> f64 {
        let residual = &a.dot(x) - b;
        let data_term = residual.dot(&residual) / 2.0;
        let regularization_term = lambda * x.mapv(|xi| xi.abs()).sum();
        data_term + regularization_term
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beamforming_processor_creation() {
        let config = BeamformingConfig::default();
        let sensor_positions = vec![[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [2e-3, 0.0, 0.0]];
        let processor = BeamformingProcessor::new(config, sensor_positions);
        assert_eq!(processor.num_sensors, 3);
    }

    #[test]
    fn test_steering_vector_calculation() {
        let config = BeamformingConfig::default();
        let sensor_positions = vec![[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]];
        let processor = BeamformingProcessor::new(config, sensor_positions);
        
        let scan_point = [0.0, 0.0, 10e-3];
        let steering_vector = processor.calculate_steering_vector(&scan_point, SteeringVectorMethod::PlaneWave).unwrap();
        
        assert_eq!(steering_vector.len(), 2);
        assert!(steering_vector.iter().all(|&x| x.abs() <= 1.0));
    }

    #[test]
    fn test_delay_sum_beamforming() {
        let config = BeamformingConfig::default();
        let sensor_positions = vec![[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]];
        let processor = BeamformingProcessor::new(config, sensor_positions);
        
        let sensor_data = Array2::ones((2, 100));
        let scan_points = vec![[0.0, 0.0, 10e-3]];
        
        let result = processor.delay_sum_beamforming(sensor_data.view(), &scan_points).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0] > 0.0);
    }
}