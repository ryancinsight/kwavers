//! Kalman filter initialization and update for [`CalibrationManager`].

use super::CalibrationManager;
use kwavers_core::error::KwaversResult;
use nalgebra::{DMatrix, DVector};
use ndarray::Array2;

use super::super::types::KalmanState;

impl CalibrationManager {
    /// Initialise Kalman filter state for `num_elements` array elements.
    ///
    /// State vector: [x, y, z, vx, vy, vz] per element (6 per element).
    /// Process noise models constant-acceleration dynamics with `accel_variance = 0.01`.
    pub(super) fn initialize_kalman_filter(
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
        let accel_variance = 0.01_f64;
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                process_noise[(base + j, base + j)] = accel_variance * 0.25;
                process_noise[(base + j, base + j + 3)] = accel_variance * 0.5;
                process_noise[(base + j + 3, base + j)] = accel_variance * 0.5;
                process_noise[(base + j + 3, base + j + 3)] = accel_variance;
            }
        }

        let measurement_noise_matrix =
            DMatrix::identity(meas_dim, meas_dim) * measurement_noise * measurement_noise;

        self.kalman_state = Some(KalmanState {
            state,
            covariance,
            process_noise,
            measurement_noise: measurement_noise_matrix,
        });

        Ok(())
    }

    /// Kalman predict + update step.
    ///
    /// F: constant-velocity state transition; H: position-only measurement.
    /// Returns position estimate for all elements.
    pub(super) fn kalman_filter_update(
        &mut self,
        measurements: &Array2<f64>,
        dt: f64,
    ) -> KwaversResult<Array2<f64>> {
        let kalman =
            self.kalman_state
                .as_mut()
                .ok_or(kwavers_core::error::KwaversError::InvalidInput(
                    "Kalman filter not initialized".to_owned(),
                ))?;

        let num_elements = measurements.nrows();
        let state_dim = kalman.state.len();
        let meas_dim = num_elements * 3;

        // State transition F: x_{k+1} = x_k + dt * v_k
        let mut f_matrix = DMatrix::identity(state_dim, state_dim);
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                f_matrix[(base + j, base + j + 3)] = dt;
            }
        }

        // Measurement matrix H: observe positions only
        let mut h_matrix = DMatrix::zeros(meas_dim, state_dim);
        for i in 0..num_elements {
            for j in 0..3 {
                h_matrix[(i * 3 + j, i * 6 + j)] = 1.0;
            }
        }

        // Predict
        kalman.state = &f_matrix * &kalman.state;
        kalman.covariance = &f_matrix * &kalman.covariance * f_matrix.transpose()
            + &kalman.process_noise * (dt * dt * dt * dt);

        // Update
        let z = DVector::from_iterator(meas_dim, measurements.iter().copied());
        let y = z - &h_matrix * &kalman.state;
        let s = &h_matrix * &kalman.covariance * h_matrix.transpose() + &kalman.measurement_noise;
        let k = &kalman.covariance
            * h_matrix.transpose()
            * s.try_inverse()
                .ok_or(kwavers_core::error::KwaversError::Numerical(
                    kwavers_core::error::NumericalError::SolverFailed {
                        method: "Matrix inversion".to_owned(),
                        reason: "Singular innovation covariance".to_owned(),
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
}
