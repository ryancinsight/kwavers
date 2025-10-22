// localization/tdoa.rs - Time Difference of Arrival processing

use super::{Position, SensorArray};
use crate::error::KwaversResult;
use serde::{Deserialize, Serialize};

/// TDOA measurement between sensor pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDOAMeasurement {
    /// Sensor pair (reference sensor, measurement sensor)
    pub sensor_pair: (usize, usize),
    /// Time difference (`t_measurement` - `t_reference`) in seconds
    pub time_difference: f64,
    /// Measurement uncertainty in seconds
    pub uncertainty: f64,
}

impl TDOAMeasurement {
    /// Create new TDOA measurement
    #[must_use]
    pub fn new(sensor1: usize, sensor2: usize, time_diff: f64) -> Self {
        Self {
            sensor_pair: (sensor1, sensor2),
            time_difference: time_diff,
            uncertainty: 1e-9, // Default 1 ns
        }
    }

    /// Convert to distance difference
    #[must_use]
    pub fn to_distance_difference(&self, sound_speed: f64) -> f64 {
        self.time_difference * sound_speed
    }
}

/// TDOA processor for hyperbolic localization
#[derive(Debug)]
pub struct TDOAProcessor {
    measurements: Vec<TDOAMeasurement>,
    sound_speed: f64,
}

impl TDOAProcessor {
    /// Create new TDOA processor
    #[must_use]
    pub fn new(sound_speed: f64) -> Self {
        Self {
            measurements: Vec::new(),
            sound_speed,
        }
    }

    /// Add measurement
    pub fn add_measurement(&mut self, measurement: TDOAMeasurement) {
        self.measurements.push(measurement);
    }

    /// Process measurements to find source location using spherical intersection
    ///
    /// Implements TDOA localization by solving the system of hyperbolic equations
    /// using nonlinear least squares optimization (Gauss-Newton method).
    ///
    /// Algorithm:
    /// 1. Initialize position estimate at array centroid
    /// 2. Iteratively refine position by minimizing range difference residuals
    /// 3. Solve normal equations: (J^T J) Δx = -J^T r
    /// 4. Update position: x_new = x_old + Δx
    /// 5. Converge when ||Δx|| < tolerance
    ///
    /// References:
    /// - Schau & Robinson (1987): "Passive source localization employing intersecting spherical surfaces from time-of-arrival differences"
    /// - Chan & Ho (1994): "A simple and efficient estimator for hyperbolic location"
    /// - Foy (1976): "Position-location solutions by Taylor-series estimation"
    pub fn process(&self, array: &SensorArray) -> KwaversResult<Position> {
        if self.measurements.len() < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 3 TDOA measurements for 3D localization".to_string(),
            ));
        }

        // Initialize position at array centroid
        let mut position = array.centroid();

        // Gauss-Newton iteration parameters
        const MAX_ITERATIONS: usize = 50;
        const TOLERANCE: f64 = 1e-6; // meters

        for iteration in 0..MAX_ITERATIONS {
            // Calculate residuals and Jacobian
            let (residuals, jacobian) = self.compute_residuals_and_jacobian(&position, array);

            // Solve normal equations: (J^T J) Δx = -J^T r
            let jtj = self.compute_jtj(&jacobian);
            let jtr = self.compute_jtr(&jacobian, &residuals);

            // Solve 3x3 system using direct inversion
            let delta = match self.solve_3x3(jtj, jtr) {
                Ok(d) => d,
                Err(_) => {
                    // Matrix singular, return current best estimate
                    log::warn!(
                        "TDOA: Singular matrix at iteration {}, returning current estimate",
                        iteration
                    );
                    break;
                }
            };

            // Update position
            position.x += delta[0];
            position.y += delta[1];
            position.z += delta[2];

            // Check convergence
            let delta_norm = (delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2)).sqrt();
            if delta_norm < TOLERANCE {
                log::debug!("TDOA converged in {} iterations", iteration + 1);
                break;
            }

            if iteration == MAX_ITERATIONS - 1 {
                log::warn!("TDOA: Max iterations reached without full convergence");
            }
        }

        Ok(position)
    }

    /// Compute residuals and Jacobian matrix for current position estimate
    fn compute_residuals_and_jacobian(
        &self,
        position: &Position,
        array: &SensorArray,
    ) -> (Vec<f64>, Vec<[f64; 3]>) {
        let mut residuals = Vec::with_capacity(self.measurements.len());
        let mut jacobian = Vec::with_capacity(self.measurements.len());

        for measurement in &self.measurements {
            let (ref_idx, meas_idx) = measurement.sensor_pair;

            // Get sensor positions
            let ref_pos = array.get_sensor_position(ref_idx);
            let meas_pos = array.get_sensor_position(meas_idx);

            // Calculate distances from source to sensors
            let d_ref = position.distance_to(ref_pos);
            let d_meas = position.distance_to(meas_pos);

            // Calculate expected distance difference
            let expected_dd = d_meas - d_ref;

            // Calculate measured distance difference
            let measured_dd = measurement.to_distance_difference(self.sound_speed);

            // Residual
            let residual = expected_dd - measured_dd;
            residuals.push(residual);

            // Jacobian row: ∂(d_meas - d_ref)/∂x, ∂(d_meas - d_ref)/∂y, ∂(d_meas - d_ref)/∂z
            let dx_meas = if d_meas > 1e-10 {
                (position.x - meas_pos.x) / d_meas
            } else {
                0.0
            };
            let dy_meas = if d_meas > 1e-10 {
                (position.y - meas_pos.y) / d_meas
            } else {
                0.0
            };
            let dz_meas = if d_meas > 1e-10 {
                (position.z - meas_pos.z) / d_meas
            } else {
                0.0
            };

            let dx_ref = if d_ref > 1e-10 {
                (position.x - ref_pos.x) / d_ref
            } else {
                0.0
            };
            let dy_ref = if d_ref > 1e-10 {
                (position.y - ref_pos.y) / d_ref
            } else {
                0.0
            };
            let dz_ref = if d_ref > 1e-10 {
                (position.z - ref_pos.z) / d_ref
            } else {
                0.0
            };

            jacobian.push([dx_meas - dx_ref, dy_meas - dy_ref, dz_meas - dz_ref]);
        }

        (residuals, jacobian)
    }

    /// Compute J^T J (3x3 matrix)
    fn compute_jtj(&self, jacobian: &[[f64; 3]]) -> [[f64; 3]; 3] {
        let mut jtj = [[0.0; 3]; 3];

        for j_row in jacobian {
            for i in 0..3 {
                for k in 0..3 {
                    jtj[i][k] += j_row[i] * j_row[k];
                }
            }
        }

        jtj
    }

    /// Compute -J^T r (3x1 vector)
    fn compute_jtr(&self, jacobian: &[[f64; 3]], residuals: &[f64]) -> [f64; 3] {
        let mut jtr = [0.0; 3];

        for (j_row, &r) in jacobian.iter().zip(residuals.iter()) {
            for i in 0..3 {
                jtr[i] -= j_row[i] * r;
            }
        }

        jtr
    }

    /// Solve 3x3 linear system using Cramer's rule
    fn solve_3x3(&self, a: [[f64; 3]; 3], b: [f64; 3]) -> Result<[f64; 3], ()> {
        // Calculate determinant
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det.abs() < 1e-12 {
            return Err(()); // Singular matrix
        }

        // Cramer's rule
        let det_x = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
            + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);

        let det_y = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
            - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);

        let det_z = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
            - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
            + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        Ok([det_x / det, det_y / det, det_z / det])
    }

    /// Calculate residuals for given position
    #[must_use]
    pub fn calculate_residuals(&self, position: &Position, array: &SensorArray) -> Vec<f64> {
        let mut residuals = Vec::new();

        for measurement in &self.measurements {
            let (ref_id, meas_id) = measurement.sensor_pair;

            if let (Some(ref_sensor), Some(meas_sensor)) =
                (array.get_sensor(ref_id), array.get_sensor(meas_id))
            {
                let ref_distance = ref_sensor.distance_to(position);
                let meas_distance = meas_sensor.distance_to(position);
                let predicted_diff = (meas_distance - ref_distance) / self.sound_speed;

                residuals.push(predicted_diff - measurement.time_difference);
            }
        }

        residuals
    }

    /// Get number of measurements
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }
}
