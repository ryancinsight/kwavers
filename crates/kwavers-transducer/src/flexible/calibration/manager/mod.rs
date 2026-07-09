//! Calibration manager for flexible transducer arrays.

use super::types::{CalibrationData, CalibrationQualityMetrics, GeometrySnapshot, KalmanState};
use kwavers_core::error::KwaversResult;
use leto::{Array2 as LetoArray2, Array3};
use leto::{
    Array1,
    Array2 as NdArray2,
};

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
                quality_metrics: CalibrationQualityMetrics {
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
    ) -> KwaversResult<LetoArray2<f64>> {
        let wavelength = sound_speed / frequency;
        let [_nx, _ny, _nz] = pressure_field.shape();

        let peaks = self.extract_peaks(pressure_field, wavelength)?;
        let correspondences = self.match_reflectors(&peaks, known_reflectors)?;
        let positions = self.estimate_positions(&correspondences, known_reflectors)?;

        self.data.reference_geometry = Some(positions.clone());
        self.update_quality_metrics(&positions, &correspondences);

        Ok(positions)
    }

    /// Triangulate position from multiple measurements using least-squares.
    /// # Errors
    /// - Returns [`kwavers_core::error::KwaversError::InvalidInput`] if fewer than 4 reflectors.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn triangulate_position(
        &self,
        measurements: &[f64],
        reflectors: &[[f64; 3]],
    ) -> KwaversResult<[f64; 3]> {
        let n = reflectors.len();
        if n < 4 || measurements.len() < n {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Insufficient measurements for triangulation".to_owned(),
            ));
        }

        let mut a_matrix = NdArray2::zeros([n - 1, 3]);
        let mut b_vector = Array1::zeros(n - 1);

        let ref_pos = &reflectors[0];
        let ref_dist = measurements[0];

        for i in 1..n {
            let pos = &reflectors[i];
            let dist_diff = measurements[i] - ref_dist;

            a_matrix[[i - 1, 0]] = 2.0 * (pos[0] - ref_pos[0]);
            a_matrix[[i - 1, 1]] = 2.0 * (pos[1] - ref_pos[1]);
            a_matrix[[i - 1, 2]] = 2.0 * (pos[2] - ref_pos[2]);

            b_vector[[i - 1]] = dist_diff.mul_add(
                dist_diff,
                -pos[2].mul_add(pos[2], pos[0].mul_add(pos[0], pos[1] * pos[1])),
            ) + ref_pos[2].mul_add(
                ref_pos[2],
                ref_pos[0].mul_add(ref_pos[0], ref_pos[1] * ref_pos[1]),
            );
        }

        let a_t = a_matrix.t().to_owned();
        let at_a = a_t.dot(&a_matrix);
        let at_b = a_t.dot(&b_vector);

        let solution = solve_linear_system(&at_a, &at_b).ok_or(
            kwavers_core::error::KwaversError::Numerical(
                kwavers_core::error::NumericalError::SolverFailed {
                    method: "Gaussian elimination".to_owned(),
                    reason: "Singular matrix in triangulation".to_owned(),
                },
            ),
        )?;

        Ok(solution)
    }

    /// Process external tracking data with Kalman filtering.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn process_external_tracking(
        &mut self,
        tracking_data: &NdArray2<f64>,
        measurement_noise: f64,
        timestamp: f64,
    ) -> KwaversResult<NdArray2<f64>> {
        let dt = timestamp - self.last_calibration_time;
        let num_elements = tracking_data.shape()[0];

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

fn solve_linear_system(a: &NdArray2<f64>, b: &Array1<f64>) -> Option<[f64; 3]> {
    if a.shape()[0] != 3 || a.shape()[1] != 3 || b.len() != 3 {
        return None;
    }

    let mut aug = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            aug[i][j] = a[[i, j]];
        }
        aug[i][3] = b[i];
    }

    let eps = 1e-15;
    for i in 0..3 {
        let mut pivot = i;
        let mut max_abs = aug[i][i].abs();
        for (r, row) in aug.iter().enumerate().skip(i + 1) {
            let cand = row[i].abs();
            if cand > max_abs {
                max_abs = cand;
                pivot = r;
            }
        }
        if max_abs <= eps {
            return None;
        }
        if pivot != i {
            aug.swap(i, pivot);
        }

        let diag = aug[i][i];
        for v in aug[i].iter_mut().skip(i) {
            *v /= diag;
        }

        for r in 0..3 {
            if r == i {
                continue;
            }
            let factor = aug[r][i];
            if factor.abs() <= eps {
                continue;
            }
            let row_i = aug[i];
            for (c, v) in aug[r].iter_mut().enumerate().skip(i) {
                *v -= factor * row_i[c];
            }
        }
    }

    Some([aug[0][3], aug[1][3], aug[2][3]])
}
