//! Calibration procedures for flexible array geometries
//!
//! This module provides methods for calibrating and tracking flexible
//! transducer arrays, including self-calibration and external tracking integration.

use crate::error::KwaversResult;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3};

/// Calibration data storage
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Time-dependent geometry snapshots
    pub geometry_history: Vec<GeometrySnapshot>,
    /// Calibration quality metrics
    pub quality_metrics: QualityMetrics,
    /// Reference configuration
    pub reference_geometry: Option<Array2<f64>>,
}

/// Geometry snapshot at a specific time
#[derive(Debug, Clone)]
pub struct GeometrySnapshot {
    /// Timestamp
    pub timestamp: f64,
    /// Element positions [n_elements x 3]
    pub positions: Array2<f64>,
    /// Confidence scores per element
    pub confidence: Array1<f64>,
}

/// Calibration quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Position uncertainty (meters)
    pub position_uncertainty: f64,
    /// Orientation uncertainty (radians)
    pub orientation_uncertainty: f64,
    /// Overall calibration confidence [0, 1]
    pub confidence: f64,
}

/// Calibration manager for flexible arrays
#[derive(Debug)]
pub struct CalibrationManager {
    /// Stored calibration data
    pub data: CalibrationData,
    /// Kalman filter state for tracking
    kalman_state: Option<KalmanState>,
    /// Last calibration timestamp
    last_calibration_time: f64,
}

/// Kalman filter state for position tracking
#[derive(Debug, Clone)]
struct KalmanState {
    /// State estimate (positions and velocities)
    state: DVector<f64>,
    /// Error covariance matrix
    covariance: DMatrix<f64>,
    /// Process noise covariance
    process_noise: DMatrix<f64>,
    /// Measurement noise covariance
    measurement_noise: DMatrix<f64>,
}

impl Default for CalibrationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationManager {
    /// Create a new calibration manager
    pub fn new() -> Self {
        Self {
            data: CalibrationData {
                geometry_history: Vec::new(),
                quality_metrics: QualityMetrics {
                    position_uncertainty: 1e-3,
                    orientation_uncertainty: 1e-2,
                    confidence: 0.0,
                },
                reference_geometry: None,
            },
            kalman_state: None,
            last_calibration_time: 0.0,
        }
    }

    /// Perform self-calibration using acoustic reflections
    pub fn self_calibrate(
        &mut self,
        pressure_field: &Array3<f64>,
        known_reflectors: &[[f64; 3]],
        frequency: f64,
        sound_speed: f64,
    ) -> KwaversResult<Array2<f64>> {
        let wavelength = sound_speed / frequency;
        let (nx, ny, nz) = pressure_field.dim();

        // Extract peak locations from pressure field
        let peaks = self.extract_peaks(pressure_field, wavelength)?;

        // Match peaks to known reflectors
        let correspondences = self.match_reflectors(&peaks, known_reflectors)?;

        // Estimate element positions using matched correspondences
        let positions = self.estimate_positions(&correspondences, known_reflectors)?;

        // Update calibration data
        self.data.reference_geometry = Some(positions.clone());
        self.update_quality_metrics(&positions, &correspondences);

        Ok(positions)
    }

    /// Triangulate position from multiple measurements using least-squares
    pub fn triangulate_position(
        &self,
        measurements: &[f64],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<[f64; 3]> {
        // Proper least-squares triangulation using overdetermined system
        // Based on: Fang, B.T. (1990) "Simple solutions for hyperbolic and related position fixes"
        // IEEE Transactions on Aerospace and Electronic Systems, 26(5), 748-753

        let n = reflectors.len();
        if n < 4 || measurements.len() < n {
            return Err(crate::error::KwaversError::InvalidInput(
                "Insufficient measurements for triangulation".to_string(),
            ));
        }

        // Build the system matrix A and vector b for Ax = b
        // Using Time Difference of Arrival (TDOA) formulation
        let mut a_matrix = DMatrix::zeros(n - 1, 3);
        let mut b_vector = DVector::zeros(n - 1);

        // Reference reflector (first one)
        let ref_pos = &reflectors[0];
        let ref_dist = measurements[0];

        for i in 1..n {
            let pos = &reflectors[i];
            let dist_diff = measurements[i] - ref_dist;

            // Coefficient matrix row
            a_matrix[(i - 1, 0)] = 2.0 * (pos[0] - ref_pos[0]);
            a_matrix[(i - 1, 1)] = 2.0 * (pos[1] - ref_pos[1]);
            a_matrix[(i - 1, 2)] = 2.0 * (pos[2] - ref_pos[2]);

            // Right-hand side
            b_vector[i - 1] = dist_diff * dist_diff
                - (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
                + (ref_pos[0] * ref_pos[0] + ref_pos[1] * ref_pos[1] + ref_pos[2] * ref_pos[2]);
        }

        // Solve using least-squares (A^T A x = A^T b)
        let at_a = a_matrix.transpose() * &a_matrix;
        let at_b = a_matrix.transpose() * b_vector;

        // Use LU decomposition for numerical stability
        let decomp = at_a.lu();
        let solution = decomp
            .solve(&at_b)
            .ok_or(crate::error::KwaversError::Numerical(
                crate::error::NumericalError::SolverFailed {
                    method: "LU decomposition".to_string(),
                    reason: "Singular matrix in triangulation".to_string(),
                },
            ))?;

        Ok([solution[0], solution[1], solution[2]])
    }

    /// Process external tracking data with proper Kalman filtering
    pub fn process_external_tracking(
        &mut self,
        tracking_data: &Array2<f64>,
        measurement_noise: f64,
        timestamp: f64,
    ) -> KwaversResult<Array2<f64>> {
        let dt = timestamp - self.last_calibration_time;
        let num_elements = tracking_data.nrows();

        // Initialize Kalman filter if needed
        if self.kalman_state.is_none() {
            self.initialize_kalman_filter(num_elements, measurement_noise)?;
        }

        // Apply Kalman filtering
        let filtered_positions = self.kalman_filter_update(tracking_data, dt)?;

        // Store result
        self.data.geometry_history.push(GeometrySnapshot {
            timestamp,
            positions: filtered_positions.clone(),
            confidence: Array1::ones(num_elements) * (1.0 / (1.0 + measurement_noise)),
        });

        self.last_calibration_time = timestamp;

        Ok(filtered_positions)
    }

    /// Initialize Kalman filter state
    fn initialize_kalman_filter(
        &mut self,
        num_elements: usize,
        measurement_noise: f64,
    ) -> KwaversResult<()> {
        // State vector: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, ...]
        // For each element: position (3) + velocity (3) = 6 states
        let state_dim = num_elements * 6;
        let meas_dim = num_elements * 3;

        // Initialize state (positions at zero, velocities at zero)
        let state = DVector::zeros(state_dim);

        // Initialize covariance (large initial uncertainty)
        let mut covariance = DMatrix::identity(state_dim, state_dim);
        covariance *= 1.0; // 1 meter initial position uncertainty

        // Process noise (acceleration model)
        let mut process_noise = DMatrix::zeros(state_dim, state_dim);
        let accel_variance = 0.01; // m²/s⁴
        for i in 0..num_elements {
            let base = i * 6;
            // Position process noise
            for j in 0..3 {
                process_noise[(base + j, base + j)] = accel_variance * 0.25; // dt⁴/4
                process_noise[(base + j, base + j + 3)] = accel_variance * 0.5; // dt³/2
                process_noise[(base + j + 3, base + j)] = accel_variance * 0.5; // dt³/2
                process_noise[(base + j + 3, base + j + 3)] = accel_variance; // dt²
            }
        }

        // Measurement noise
        let measurement_noise =
            DMatrix::identity(meas_dim, meas_dim) * measurement_noise * measurement_noise;

        self.kalman_state = Some(KalmanState {
            state,
            covariance,
            process_noise,
            measurement_noise,
        });

        Ok(())
    }

    /// Kalman filter update step
    fn kalman_filter_update(
        &mut self,
        measurements: &Array2<f64>,
        dt: f64,
    ) -> KwaversResult<Array2<f64>> {
        let kalman = self
            .kalman_state
            .as_mut()
            .ok_or(crate::error::KwaversError::InvalidInput(
                "Kalman filter not initialized".to_string(),
            ))?;

        let num_elements = measurements.nrows();
        let state_dim = kalman.state.len();
        let meas_dim = num_elements * 3;

        // State transition matrix (constant velocity model)
        let mut f_matrix = DMatrix::identity(state_dim, state_dim);
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                f_matrix[(base + j, base + j + 3)] = dt; // Position += velocity * dt
            }
        }

        // Measurement matrix (observe positions only)
        let mut h_matrix = DMatrix::zeros(meas_dim, state_dim);
        for i in 0..num_elements {
            for j in 0..3 {
                h_matrix[(i * 3 + j, i * 6 + j)] = 1.0;
            }
        }

        // Prediction step
        kalman.state = &f_matrix * &kalman.state;
        kalman.covariance = &f_matrix * &kalman.covariance * f_matrix.transpose()
            + &kalman.process_noise * (dt * dt * dt * dt);

        // Update step
        let z = DVector::from_iterator(meas_dim, measurements.iter().cloned());

        let y = z - &h_matrix * &kalman.state; // Innovation
        let s = &h_matrix * &kalman.covariance * h_matrix.transpose() + &kalman.measurement_noise; // Innovation covariance
        let k = &kalman.covariance
            * h_matrix.transpose()
            * s.try_inverse()
                .ok_or(crate::error::KwaversError::Numerical(
                    crate::error::NumericalError::SolverFailed {
                        method: "Matrix inversion".to_string(),
                        reason: "Singular innovation covariance".to_string(),
                    },
                ))?; // Kalman gain

        kalman.state = &kalman.state + &k * y;
        let i_kh = DMatrix::identity(state_dim, state_dim) - &k * &h_matrix;
        kalman.covariance = i_kh * &kalman.covariance;

        // Extract filtered positions
        let mut filtered = Array2::zeros((num_elements, 3));
        for i in 0..num_elements {
            for j in 0..3 {
                filtered[[i, j]] = kalman.state[i * 6 + j];
            }
        }

        Ok(filtered)
    }

    /// Extract peak locations from pressure field
    fn extract_peaks(
        &self,
        pressure_field: &Array3<f64>,
        wavelength: f64,
    ) -> KwaversResult<Vec<[f64; 3]>> {
        let (nx, ny, nz) = pressure_field.dim();
        let mut peaks = Vec::new();

        // Peak detection threshold (based on RMS of field)
        let rms: f64 =
            pressure_field.iter().map(|&x| x * x).sum::<f64>().sqrt() / (nx * ny * nz) as f64;
        let threshold = 3.0 * rms; // 3-sigma threshold

        // Search for local maxima with minimum separation
        let min_separation = (wavelength / 2.0) as usize;

        for i in min_separation..(nx - min_separation) {
            for j in min_separation..(ny - min_separation) {
                for k in min_separation..(nz - min_separation) {
                    let val = pressure_field[[i, j, k]].abs();

                    if val > threshold {
                        // Check if local maximum
                        let mut is_max = true;
                        for di in 0..=2 {
                            for dj in 0..=2 {
                                for dk in 0..=2 {
                                    if di == 1 && dj == 1 && dk == 1 {
                                        continue;
                                    }
                                    let neighbor =
                                        pressure_field[[i + di - 1, j + dj - 1, k + dk - 1]].abs();
                                    if neighbor > val {
                                        is_max = false;
                                        break;
                                    }
                                }
                                if !is_max {
                                    break;
                                }
                            }
                            if !is_max {
                                break;
                            }
                        }

                        if is_max {
                            peaks.push([i as f64, j as f64, k as f64]);
                        }
                    }
                }
            }
        }

        Ok(peaks)
    }

    /// Match detected peaks to known reflectors
    fn match_reflectors(
        &self,
        peaks: &[[f64; 3]],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<Vec<(usize, usize)>> {
        let mut correspondences = Vec::new();

        // Hungarian algorithm for optimal assignment
        // For now, use nearest neighbor matching
        for (i, peak) in peaks.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_match = 0;

            for (j, reflector) in reflectors.iter().enumerate() {
                let dist = ((peak[0] - reflector[0]).powi(2)
                    + (peak[1] - reflector[1]).powi(2)
                    + (peak[2] - reflector[2]).powi(2))
                .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_match = j;
                }
            }

            correspondences.push((i, best_match));
        }

        Ok(correspondences)
    }

    /// Estimate element positions from correspondences
    fn estimate_positions(
        &self,
        correspondences: &[(usize, usize)],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<Array2<f64>> {
        let num_elements = correspondences.len();
        let mut positions = Array2::zeros((num_elements, 3));

        // Use correspondence geometry to estimate positions
        for (i, &(_, reflector_idx)) in correspondences.iter().enumerate() {
            if reflector_idx < reflectors.len() {
                // Initial estimate: use reflector position
                // In a full implementation, this would use the acoustic path geometry
                positions[[i, 0]] = reflectors[reflector_idx][0];
                positions[[i, 1]] = reflectors[reflector_idx][1];
                positions[[i, 2]] = reflectors[reflector_idx][2];
            }
        }

        Ok(positions)
    }

    /// Update quality metrics based on calibration results
    fn update_quality_metrics(
        &mut self,
        positions: &Array2<f64>,
        correspondences: &[(usize, usize)],
    ) {
        // Calculate position uncertainty from correspondence quality
        let num_correspondences = correspondences.len();
        let num_elements = positions.nrows();

        // Estimate uncertainty based on correspondence ratio
        let correspondence_ratio = num_correspondences as f64 / num_elements.max(1) as f64;

        self.data.quality_metrics.position_uncertainty = 1e-3 / correspondence_ratio;
        self.data.quality_metrics.orientation_uncertainty = 1e-2 / correspondence_ratio;
        self.data.quality_metrics.confidence = correspondence_ratio.min(1.0);
    }

    /// Get calibration confidence
    pub fn get_confidence(&self) -> f64 {
        if let Some(last_snapshot) = self.data.geometry_history.last() {
            last_snapshot.confidence.mean().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Get calibration data
    pub fn data(&self) -> &CalibrationData {
        &self.data
    }
}
