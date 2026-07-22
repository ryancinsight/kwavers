//! Trilateration solver implementation.

use kwavers_core::error::{KwaversError, KwaversResult};

use super::types::{LocalizationResult, TrilaterationConfig};

/// Trilateration source localizer
#[derive(Debug)]
pub struct Trilateration {
    pub(super) config: TrilaterationConfig,
    pub(super) sensor_positions: Vec<[f64; 3]>,
    pub(super) num_sensors: usize,
}

impl Trilateration {
    /// Create a new trilateration localizer
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(
        sensor_positions: Vec<[f64; 3]>,
        config: TrilaterationConfig,
    ) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();

        if num_sensors < 4 {
            return Err(KwaversError::InvalidInput(
                "Need at least 4 sensors for 3D trilateration".to_owned(),
            ));
        }

        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_owned(),
            ));
        }

        Ok(Self {
            config,
            sensor_positions,
            num_sensors,
        })
    }

    /// Localize source from time-of-arrival measurements
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn localize(&self, arrival_times: &[f64]) -> KwaversResult<LocalizationResult> {
        if arrival_times.len() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} arrival times, got {}",
                self.num_sensors,
                arrival_times.len()
            )));
        }

        let reference_time = arrival_times[0];
        let time_differences: Vec<f64> = arrival_times
            .iter()
            .skip(1)
            .map(|&t| t - reference_time)
            .collect();

        let initial_pos = self.config.initial_guess.unwrap_or_else(|| {
            let sum_x: f64 = self.sensor_positions.iter().map(|p| p[0]).sum();
            let sum_y: f64 = self.sensor_positions.iter().map(|p| p[1]).sum();
            let sum_z: f64 = self.sensor_positions.iter().map(|p| p[2]).sum();
            let n = self.num_sensors as f64;
            [sum_x / n, sum_y / n, sum_z / n]
        });

        // Gauss-Newton iterative solver
        let mut position = initial_pos;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            let (residuals, jacobian) =
                self.compute_residuals_and_jacobian(&position, &time_differences)?;

            let residual_norm = residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

            if residual_norm < self.config.convergence_tolerance {
                converged = true;
                break;
            }

            let update = self.solve_least_squares(&jacobian, &residuals)?;

            position[0] += update[0];
            position[1] += update[1];
            position[2] += update[2];

            let update_norm = update[2]
                .mul_add(
                    update[2],
                    update[0].mul_add(update[0], update[1] * update[1]),
                )
                .sqrt();
            if update_norm < self.config.convergence_tolerance {
                converged = true;
                break;
            }
        }

        let (final_residuals, _) =
            self.compute_residuals_and_jacobian(&position, &time_differences)?;
        let residual = final_residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();
        let uncertainty = residual / (self.num_sensors as f64).sqrt();

        Ok(LocalizationResult {
            position,
            uncertainty,
            residual,
            iterations,
            converged,
        })
    }

    /// Compute residuals and Jacobian for Gauss-Newton iteration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_residuals_and_jacobian(
        &self,
        position: &[f64; 3],
        time_differences: &[f64],
    ) -> KwaversResult<(Vec<f64>, Vec<[f64; 3]>)> {
        let ref_pos = &self.sensor_positions[0];
        let ref_dist = self.distance(position, ref_pos);

        let mut residuals = Vec::with_capacity(time_differences.len());
        let mut jacobian = Vec::with_capacity(time_differences.len());

        for (i, &td) in time_differences.iter().enumerate() {
            let sensor_pos = &self.sensor_positions[i + 1];
            let dist = self.distance(position, sensor_pos);

            let predicted_tdoa = (dist - ref_dist) / self.config.sound_speed;
            let residual = td - predicted_tdoa;
            residuals.push(residual);

            let dx_i = position[0] - sensor_pos[0];
            let dy_i = position[1] - sensor_pos[1];
            let dz_i = position[2] - sensor_pos[2];

            let dx_ref = position[0] - ref_pos[0];
            let dy_ref = position[1] - ref_pos[1];
            let dz_ref = position[2] - ref_pos[2];

            let jac_x = -(dx_i / dist - dx_ref / ref_dist) / self.config.sound_speed;
            let jac_y = -(dy_i / dist - dy_ref / ref_dist) / self.config.sound_speed;
            let jac_z = -(dz_i / dist - dz_ref / ref_dist) / self.config.sound_speed;

            jacobian.push([jac_x, jac_y, jac_z]);
        }

        Ok((residuals, jacobian))
    }

    /// Solve least squares with Levenberg-Marquardt damping: (J^T J + λI)x = J^T b
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn solve_least_squares(
        &self,
        jacobian: &[[f64; 3]],
        residuals: &[f64],
    ) -> KwaversResult<[f64; 3]> {
        let mut jtj = [[0.0; 3]; 3];
        for j_row in jacobian {
            for i in 0..3 {
                for j in 0..3 {
                    jtj[i][j] += j_row[i] * j_row[j];
                }
            }
        }

        // Levenberg-Marquardt damping
        let max_diag = jtj[0][0].abs().max(jtj[1][1].abs()).max(jtj[2][2].abs());
        let damping = (max_diag * 1e-6).max(1e-20);
        jtj[0][0] += damping;
        jtj[1][1] += damping;
        jtj[2][2] += damping;

        let mut neg_jtr = [0.0; 3];
        for (j_row, &r) in jacobian.iter().zip(residuals.iter()) {
            for i in 0..3 {
                neg_jtr[i] -= j_row[i] * r;
            }
        }

        self.solve_3x3(&jtj, &neg_jtr)
    }

    /// Solve 3x3 linear system Ax = b via Gaussian elimination with partial pivoting
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    fn solve_3x3(&self, a: &[[f64; 3]; 3], b: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        let mut aug = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                aug[i][j] = a[i][j];
            }
            aug[i][3] = b[i];
        }

        for k in 0..3 {
            let mut max_row = k;
            for i in (k + 1)..3 {
                if aug[i][k].abs() > aug[max_row][k].abs() {
                    max_row = i;
                }
            }

            if max_row != k {
                aug.swap(k, max_row);
            }

            if aug[k][k].abs() < 1e-30 {
                return Err(KwaversError::InvalidInput(
                    "Singular matrix in trilateration - sensors may be coplanar".to_owned(),
                ));
            }

            for i in (k + 1)..3 {
                let factor = aug[i][k] / aug[k][k];
                let row_k = aug[k];
                for (j, value) in aug[i].iter_mut().enumerate().skip(k) {
                    *value -= factor * row_k[j];
                }
            }
        }

        let mut x = [0.0; 3];
        for i in (0..3).rev() {
            let mut sum = aug[i][3];
            for j in (i + 1)..3 {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }

        Ok(x)
    }

    /// Compute Euclidean distance between two points
    fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
    }
}