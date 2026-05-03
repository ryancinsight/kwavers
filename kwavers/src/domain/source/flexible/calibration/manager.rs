//! Calibration manager for flexible transducer arrays.

use super::types::{CalibrationData, GeometrySnapshot, KalmanState, QualityMetrics};
use crate::core::error::KwaversResult;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3};

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

impl Default for CalibrationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationManager {
    /// Create a new calibration manager
    #[must_use]
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
        let (_nx, _ny, _nz) = pressure_field.dim();

        let peaks = self.extract_peaks(pressure_field, wavelength)?;
        let correspondences = self.match_reflectors(&peaks, known_reflectors)?;
        let positions = self.estimate_positions(&correspondences, known_reflectors)?;

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
        let n = reflectors.len();
        if n < 4 || measurements.len() < n {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Insufficient measurements for triangulation".to_string(),
            ));
        }

        let mut a_matrix = DMatrix::zeros(n - 1, 3);
        let mut b_vector = DVector::zeros(n - 1);

        let ref_pos = &reflectors[0];
        let ref_dist = measurements[0];

        for i in 1..n {
            let pos = &reflectors[i];
            let dist_diff = measurements[i] - ref_dist;

            a_matrix[(i - 1, 0)] = 2.0 * (pos[0] - ref_pos[0]);
            a_matrix[(i - 1, 1)] = 2.0 * (pos[1] - ref_pos[1]);
            a_matrix[(i - 1, 2)] = 2.0 * (pos[2] - ref_pos[2]);

            b_vector[i - 1] = dist_diff * dist_diff
                - (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
                + (ref_pos[0] * ref_pos[0] + ref_pos[1] * ref_pos[1] + ref_pos[2] * ref_pos[2]);
        }

        let at_a = a_matrix.transpose() * &a_matrix;
        let at_b = a_matrix.transpose() * b_vector;

        let decomp = at_a.lu();
        let solution = decomp
            .solve(&at_b)
            .ok_or(crate::core::error::KwaversError::Numerical(
                crate::core::error::NumericalError::SolverFailed {
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

        if self.kalman_state.is_none() {
            self.initialize_kalman_filter(num_elements, measurement_noise)?;
        }

        let filtered_positions = self.kalman_filter_update(tracking_data, dt)?;

        self.data.geometry_history.push(GeometrySnapshot {
            timestamp,
            positions: filtered_positions.clone(),
            confidence: Array1::ones(num_elements) * (1.0 / (1.0 + measurement_noise)),
        });

        self.last_calibration_time = timestamp;

        Ok(filtered_positions)
    }

    fn initialize_kalman_filter(
        &mut self,
        num_elements: usize,
        measurement_noise: f64,
    ) -> KwaversResult<()> {
        let state_dim = num_elements * 6;
        let meas_dim = num_elements * 3;

        let state = DVector::zeros(state_dim);

        let mut covariance = DMatrix::identity(state_dim, state_dim);
        covariance *= 1.0;

        let mut process_noise = DMatrix::zeros(state_dim, state_dim);
        let accel_variance = 0.01;
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                process_noise[(base + j, base + j)] = accel_variance * 0.25;
                process_noise[(base + j, base + j + 3)] = accel_variance * 0.5;
                process_noise[(base + j + 3, base + j)] = accel_variance * 0.5;
                process_noise[(base + j + 3, base + j + 3)] = accel_variance;
            }
        }

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

    fn kalman_filter_update(
        &mut self,
        measurements: &Array2<f64>,
        dt: f64,
    ) -> KwaversResult<Array2<f64>> {
        let kalman =
            self.kalman_state
                .as_mut()
                .ok_or(crate::core::error::KwaversError::InvalidInput(
                    "Kalman filter not initialized".to_string(),
                ))?;

        let num_elements = measurements.nrows();
        let state_dim = kalman.state.len();
        let meas_dim = num_elements * 3;

        let mut f_matrix = DMatrix::identity(state_dim, state_dim);
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                f_matrix[(base + j, base + j + 3)] = dt;
            }
        }

        let mut h_matrix = DMatrix::zeros(meas_dim, state_dim);
        for i in 0..num_elements {
            for j in 0..3 {
                h_matrix[(i * 3 + j, i * 6 + j)] = 1.0;
            }
        }

        kalman.state = &f_matrix * &kalman.state;
        kalman.covariance = &f_matrix * &kalman.covariance * f_matrix.transpose()
            + &kalman.process_noise * (dt * dt * dt * dt);

        let z = DVector::from_iterator(meas_dim, measurements.iter().copied());
        let y = z - &h_matrix * &kalman.state;
        let s = &h_matrix * &kalman.covariance * h_matrix.transpose() + &kalman.measurement_noise;
        let k = &kalman.covariance
            * h_matrix.transpose()
            * s.try_inverse()
                .ok_or(crate::core::error::KwaversError::Numerical(
                    crate::core::error::NumericalError::SolverFailed {
                        method: "Matrix inversion".to_string(),
                        reason: "Singular innovation covariance".to_string(),
                    },
                ))?;

        kalman.state = &kalman.state + &k * y;
        let i_kh = DMatrix::identity(state_dim, state_dim) - &k * &h_matrix;
        kalman.covariance = i_kh * &kalman.covariance;

        let mut filtered = Array2::zeros((num_elements, 3));
        for i in 0..num_elements {
            for j in 0..3 {
                filtered[[i, j]] = kalman.state[i * 6 + j];
            }
        }

        Ok(filtered)
    }

    fn extract_peaks(
        &self,
        pressure_field: &Array3<f64>,
        wavelength: f64,
    ) -> KwaversResult<Vec<[f64; 3]>> {
        let (nx, ny, nz) = pressure_field.dim();
        let mut peaks = Vec::new();

        let rms: f64 =
            pressure_field.iter().map(|&x| x * x).sum::<f64>().sqrt() / (nx * ny * nz) as f64;
        let threshold = 3.0 * rms;

        let min_separation = (wavelength / 2.0) as usize;

        for i in min_separation..(nx - min_separation) {
            for j in min_separation..(ny - min_separation) {
                for k in min_separation..(nz - min_separation) {
                    let val = pressure_field[[i, j, k]].abs();

                    if val > threshold {
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

    fn match_reflectors(
        &self,
        peaks: &[[f64; 3]],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<Vec<(usize, usize)>> {
        let mut correspondences = Vec::new();

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

    fn estimate_positions(
        &self,
        correspondences: &[(usize, usize)],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<Array2<f64>> {
        let num_elements = correspondences.len();
        let mut positions = Array2::zeros((num_elements, 3));

        for (i, &(_, reflector_idx)) in correspondences.iter().enumerate() {
            if reflector_idx < reflectors.len() {
                positions[[i, 0]] = reflectors[reflector_idx][0];
                positions[[i, 1]] = reflectors[reflector_idx][1];
                positions[[i, 2]] = reflectors[reflector_idx][2];
            }
        }

        Ok(positions)
    }

    fn update_quality_metrics(
        &mut self,
        positions: &Array2<f64>,
        correspondences: &[(usize, usize)],
    ) {
        let num_correspondences = correspondences.len();
        let num_elements = positions.nrows();
        let correspondence_ratio = num_correspondences as f64 / num_elements.max(1) as f64;

        self.data.quality_metrics.position_uncertainty = 1e-3 / correspondence_ratio;
        self.data.quality_metrics.orientation_uncertainty = 1e-2 / correspondence_ratio;
        self.data.quality_metrics.confidence = correspondence_ratio.min(1.0);
    }

    /// Get calibration confidence
    #[must_use]
    pub fn get_confidence(&self) -> f64 {
        if let Some(last_snapshot) = self.data.geometry_history.last() {
            last_snapshot.confidence.mean().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Get calibration data
    #[must_use]
    pub fn data(&self) -> &CalibrationData {
        &self.data
    }
}
