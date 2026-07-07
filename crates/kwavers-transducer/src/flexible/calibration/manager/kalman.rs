//! Kalman filter initialization and update for [`CalibrationManager`].

use super::CalibrationManager;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::LinearAlgebra;
use leto::Array2 as LetoArray2;
use ndarray::{Array1, Array2};

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

        let state = Array1::zeros(state_dim);

        let mut covariance = identity_matrix(state_dim);
        covariance *= 1.0;

        let mut process_noise = Array2::zeros((state_dim, state_dim));
        let accel_variance = 0.01_f64;
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                process_noise[[base + j, base + j]] = accel_variance * 0.25;
                process_noise[[base + j, base + j + 3]] = accel_variance * 0.5;
                process_noise[[base + j + 3, base + j]] = accel_variance * 0.5;
                process_noise[[base + j + 3, base + j + 3]] = accel_variance;
            }
        }

        let measurement_noise_matrix =
            identity_matrix(meas_dim) * measurement_noise * measurement_noise;

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
        let mut f_matrix = identity_matrix(state_dim);
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                f_matrix[[base + j, base + j + 3]] = dt;
            }
        }

        // Measurement matrix H: observe positions only
        let mut h_matrix = Array2::zeros((meas_dim, state_dim));
        for i in 0..num_elements {
            for j in 0..3 {
                h_matrix[[i * 3 + j, i * 6 + j]] = 1.0;
            }
        }

        // Predict
        kalman.state = f_matrix.dot(&kalman.state);
        let f_t = f_matrix.t().to_owned();
        kalman.covariance =
            f_matrix.dot(&kalman.covariance).dot(&f_t) + &kalman.process_noise * dt.powi(4);

        // Update
        let z = Array1::from_iter(measurements.iter().copied());
        let y = &z - &h_matrix.dot(&kalman.state);
        let h_t = h_matrix.t().to_owned();
        let s = h_matrix.dot(&kalman.covariance).dot(&h_t) + &kalman.measurement_noise;
        let s_inv = invert_matrix(&s)?;
        let k = kalman.covariance.dot(&h_t).dot(&s_inv);

        kalman.state = &kalman.state + &k.dot(&y);
        let i_kh = identity_matrix(state_dim) - k.dot(&h_matrix);
        kalman.covariance = i_kh.dot(&kalman.covariance);

        let mut filtered = Array2::zeros((num_elements, 3));
        for i in 0..num_elements {
            for j in 0..3 {
                filtered[[i, j]] = kalman.state[i * 6 + j];
            }
        }

        Ok(filtered)
    }
}

fn identity_matrix(n: usize) -> Array2<f64> {
    let mut out = Array2::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = 1.0;
    }
    out
}

fn invert_matrix(matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
    let leto_matrix = LetoArray2::from_shape_fn([matrix.nrows(), matrix.ncols()], |[i, j]| {
        matrix[[i, j]]
    });
    let inverse = LinearAlgebra::matrix_inverse(&leto_matrix)?;
    Ok(Array2::from_shape_fn(
        (inverse.shape()[0], inverse.shape()[1]),
        |(i, j)| inverse[[i, j]],
    ))
}
