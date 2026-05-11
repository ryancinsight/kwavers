//! Calibration manager for flexible transducer arrays.

use super::types::{CalibrationData, GeometrySnapshot, KalmanState, QualityMetrics};
use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2, Array3};

mod acoustic;
mod kalman;

/// Calibration manager for flexible arrays
#[derive(Debug)]
pub struct CalibrationManager {
    /// Stored calibration data
    pub data: CalibrationData,
    /// Kalman filter state for tracking
    pub(super) kalman_state: Option<KalmanState>,
    /// Last calibration timestamp
    pub(super) last_calibration_time: f64,
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

    /// Perform self-calibration using acoustic reflections.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

    /// Triangulate position from multiple measurements using least-squares.
    /// # Errors
    /// - Returns [`crate::core::error::KwaversError::InvalidInput`] if fewer than 4 reflectors.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn triangulate_position(
        &self,
        measurements: &[f64],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<[f64; 3]> {
        use nalgebra::{DMatrix, DVector};

        let n = reflectors.len();
        if n < 4 || measurements.len() < n {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Insufficient measurements for triangulation".to_owned(),
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

            b_vector[i - 1] = dist_diff.mul_add(
                dist_diff,
                -pos[2].mul_add(
                    pos[2],
                    pos[0].mul_add(pos[0], pos[1] * pos[1]),
                ),
            ) + ref_pos[2].mul_add(
                ref_pos[2],
                ref_pos[0].mul_add(ref_pos[0], ref_pos[1] * ref_pos[1]),
            );
        }

        let at_a = a_matrix.transpose() * &a_matrix;
        let at_b = a_matrix.transpose() * b_vector;

        let decomp = at_a.lu();
        let solution = decomp
            .solve(&at_b)
            .ok_or(crate::core::error::KwaversError::Numerical(
                crate::core::error::NumericalError::SolverFailed {
                    method: "LU decomposition".to_owned(),
                    reason: "Singular matrix in triangulation".to_owned(),
                },
            ))?;

        Ok([solution[0], solution[1], solution[2]])
    }

    /// Process external tracking data with Kalman filtering.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

    /// Get calibration confidence from the last geometry snapshot.
    #[must_use]
    pub fn get_confidence(&self) -> f64 {
        if let Some(last_snapshot) = self.data.geometry_history.last() {
            last_snapshot.confidence.mean().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Borrow calibration data.
    #[must_use]
    pub fn data(&self) -> &CalibrationData {
        &self.data
    }
}
