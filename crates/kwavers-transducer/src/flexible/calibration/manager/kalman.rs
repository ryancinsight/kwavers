//! Kalman filter initialization and update for [`CalibrationManager`].

use super::CalibrationManager;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};
use ndarray::Array2 as NdArray2;

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

        let state = Array1::from_elem([state_dim], 0.0_f64);

        let mut covariance_flat = vec![0.0_f64; state_dim * state_dim];
        for i in 0..state_dim {
            covariance_flat[i * state_dim + i] = 1.0;
        }
        let covariance = Array2::from_shape_vec([state_dim, state_dim], covariance_flat)
            .expect("square covariance");

        let mut process_noise_flat = vec![0.0_f64; state_dim * state_dim];
        let accel_variance = 0.01_f64;
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                process_noise_flat[(base + j) * state_dim + (base + j)] = accel_variance * 0.25;
                process_noise_flat[(base + j) * state_dim + (base + j + 3)] = accel_variance * 0.5;
                process_noise_flat[(base + j + 3) * state_dim + (base + j)] = accel_variance * 0.5;
                process_noise_flat[(base + j + 3) * state_dim + (base + j + 3)] = accel_variance;
            }
        }
        let process_noise = Array2::from_shape_vec([state_dim, state_dim], process_noise_flat)
            .expect("square process noise");

        let meas_noise_val = measurement_noise * measurement_noise;
        let mut meas_noise_flat = vec![0.0_f64; meas_dim * meas_dim];
        for i in 0..meas_dim {
            meas_noise_flat[i * meas_dim + i] = meas_noise_val;
        }
        let measurement_noise_matrix =
            Array2::from_shape_vec([meas_dim, meas_dim], meas_noise_flat)
                .expect("square measurement noise");

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
        measurements: &NdArray2<f64>,
        dt: f64,
    ) -> KwaversResult<NdArray2<f64>> {
        let kalman =
            self.kalman_state
                .as_mut()
                .ok_or(kwavers_core::error::KwaversError::InvalidInput(
                    "Kalman filter not initialized".to_owned(),
                ))?;

        let num_elements = measurements.shape()[0];
        let state_dim = kalman.state.shape()[0];
        let meas_dim = num_elements * 3;

        // Helper closures for dense matrix ops over flat Vec<f64>.
        // Avoids a full matmul trait import; sizes stay in the tens-of-elements.
        let mat_mul = |a: &Array2<f64>, b: &Array2<f64>| -> Array2<f64> {
            let [ra, ca] = a.shape();
            let [_rb, cb] = b.shape();
            let mut out = vec![0.0; ra * cb];
            for i in 0..ra {
                for k in 0..ca {
                    let aik = *a.get([i, k]).unwrap();
                    for j in 0..cb {
                        out[i * cb + j] += aik * *b.get([k, j]).unwrap();
                    }
                }
            }
            Array2::from_shape_vec([ra, cb], out).unwrap()
        };

        let mat_add = |a: &Array2<f64>, b: &Array2<f64>| -> Array2<f64> {
            let [r, c] = a.shape();
            let flat: Vec<f64> = (0..r * c)
                .map(|idx| {
                    *a.get([idx / c, idx % c]).unwrap() + *b.get([idx / c, idx % c]).unwrap()
                })
                .collect();
            Array2::from_shape_vec([r, c], flat).unwrap()
        };

        let mat_transpose = |a: &Array2<f64>| -> Array2<f64> {
            let [r, c] = a.shape();
            let mut flat = vec![0.0; r * c];
            for i in 0..r {
                for j in 0..c {
                    flat[j * r + i] = *a.get([i, j]).unwrap();
                }
            }
            Array2::from_shape_vec([c, r], flat).unwrap()
        };

        let mat_inv = |a: &Array2<f64>| -> Option<Array2<f64>> {
            use leto_ops::application::linalg::lu_decompose;
            lu_decompose(&a.view()).ok()?.inv().ok()
        };

        let mat_matvec = |a: &Array2<f64>, x: &Array1<f64>| -> Array1<f64> {
            let [r, c] = a.shape();
            let mut out = vec![0.0; r];
            for i in 0..r {
                let mut acc = 0.0;
                for k in 0..c {
                    acc += *a.get([i, k]).unwrap() * *x.get([k]).unwrap();
                }
                out[i] = acc;
            }
            Array1::from_vec([r], out).unwrap()
        };

        let vec_sub = |a: &Array1<f64>, b: &Array1<f64>| -> Array1<f64> {
            let n = a.shape()[0];
            let flat: Vec<f64> = (0..n)
                .map(|i| *a.get([i]).unwrap() - *b.get([i]).unwrap())
                .collect();
            Array1::from_vec([n], flat).unwrap()
        };

        let vec_add = |a: &Array1<f64>, b: &Array1<f64>| -> Array1<f64> {
            let n = a.shape()[0];
            let flat: Vec<f64> = (0..n)
                .map(|i| *a.get([i]).unwrap() + *b.get([i]).unwrap())
                .collect();
            Array1::from_vec([n], flat).unwrap()
        };

        // State transition F: x_{k+1} = x_k + dt * v_k
        let mut f_flat = vec![0.0_f64; state_dim * state_dim];
        for i in 0..state_dim {
            f_flat[i * state_dim + i] = 1.0;
        }
        for i in 0..num_elements {
            let base = i * 6;
            for j in 0..3 {
                f_flat[(base + j) * state_dim + (base + j + 3)] = dt;
            }
        }
        let f_matrix = Array2::from_shape_vec([state_dim, state_dim], f_flat)
            .expect("square F matrix");

        // Measurement matrix H: observe positions only
        let mut h_flat = vec![0.0_f64; meas_dim * state_dim];
        for i in 0..num_elements {
            for j in 0..3 {
                h_flat[(i * 3 + j) * state_dim + (i * 6 + j)] = 1.0;
            }
        }
        let h_matrix = Array2::from_shape_vec([meas_dim, state_dim], h_flat)
            .expect("H matrix shape");

        // Predict
        kalman.state = mat_matvec(&f_matrix, &kalman.state);
        let ft = mat_transpose(&f_matrix);
        let fpc = mat_mul(&f_matrix, &kalman.covariance);
        let dt4 = dt * dt * dt * dt;
        let q_scaled: Vec<f64> = (0..state_dim * state_dim)
            .map(|idx| *kalman.process_noise.get([idx / state_dim, idx % state_dim]).unwrap() * dt4)
            .collect();
        let q_s = Array2::from_shape_vec([state_dim, state_dim], q_scaled).unwrap();
        kalman.covariance = mat_add(&mat_mul(&fpc, &ft), &q_s);

        // Update
        let z_flat: Vec<f64> = measurements.iter().copied().collect();
        let z = Array1::from_vec([meas_dim], z_flat).unwrap();

        let hx = mat_matvec(&h_matrix, &kalman.state);
        let y = vec_sub(&z, &hx);

        let ht = mat_transpose(&h_matrix);
        let ph = mat_mul(&kalman.covariance, &ht);
        let hph = mat_mul(&h_matrix, &ph);
        let s = mat_add(&hph, &kalman.measurement_noise);

        let s_inv = mat_inv(&s).ok_or(kwavers_core::error::KwaversError::Numerical(
            kwavers_core::error::NumericalError::SolverFailed {
                method: "Matrix inversion".to_owned(),
                reason: "Singular innovation covariance".to_owned(),
            },
        ))?;

        let k_gain = mat_mul(&ph, &s_inv);
        let ky = mat_matvec(&k_gain, &y);
        kalman.state = vec_add(&kalman.state, &ky);

        let kh = mat_mul(&k_gain, &h_matrix);
        let mut i_kh_flat = vec![0.0_f64; state_dim * state_dim];
        for i in 0..state_dim {
            i_kh_flat[i * state_dim + i] = 1.0;
        }
        let eye = Array2::from_shape_vec([state_dim, state_dim], i_kh_flat).unwrap();
        let i_kh_flat: Vec<f64> = (0..state_dim * state_dim)
            .map(|idx| {
                *eye.get([idx / state_dim, idx % state_dim]).unwrap()
                    - *kh.get([idx / state_dim, idx % state_dim]).unwrap()
            })
            .collect();
        let i_kh = Array2::from_shape_vec([state_dim, state_dim], i_kh_flat).unwrap();
        let old_cov = kalman.covariance.clone();
        kalman.covariance = mat_mul(&i_kh, &old_cov);

        let mut filtered = NdArray2::zeros([num_elements, 3]);
        for i in 0..num_elements {
            for j in 0..3 {
                filtered[[i, j]] = *kalman.state.get([i * 6 + j]).unwrap();
            }
        }

        Ok(filtered)
    }
}
