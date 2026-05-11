//! `SirtReconstructor` — unified SIRT/ART/OSEM iterative reconstruction.
//!
//! All algorithms solve the linear inverse problem: minimise `‖Ax − b‖²`
//! where A is the system matrix, x is the image, and b is the sensor data.
//!
//! # References
//! - Natterer, F., & Wübbeling, F. (2001). *Mathematical Methods in Image Reconstruction*
//! - Kaczmarz, S. (1937). "Angenäherte Auflösung von Systemen linearer Gleichungen"
//! - Hudson, H. M., & Larkin, R. S. (1994). "Accelerated image reconstruction using ordered subsets"

use crate::core::error::KwaversResult;
use crate::math::inverse_problems::ModelRegularizer3D;
use log::debug;
use ndarray::{Array1, Array2, Array3};

use super::config::{SirtAlgorithm, SirtConfig, SirtResult};

/// Unified iterative reconstructor for SIRT, ART, and OSEM.
#[derive(Debug)]
pub struct SirtReconstructor {
    config: SirtConfig,
}

impl SirtReconstructor {
    /// Create a new reconstructor with the given configuration.
    #[must_use] 
    pub fn new(config: SirtConfig) -> Self {
        Self { config }
    }

    /// Reconstruct an image from sensor data.
    ///
    /// # Arguments
    /// - `system_matrix` — Forward problem matrix A (m × n).
    /// - `sensor_data` — Measured data b (m).
    /// - `grid_size` — 3D grid dimensions `(nx, ny, nz)` where n = nx·ny·nz.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn reconstruct(
        &self,
        system_matrix: &Array2<f64>,
        sensor_data: &Array1<f64>,
        grid_size: (usize, usize, usize),
    ) -> KwaversResult<SirtResult> {
        let start_time = std::time::Instant::now();

        let (m, n) = system_matrix.dim();
        assert_eq!(n, grid_size.0 * grid_size.1 * grid_size.2);
        assert_eq!(m, sensor_data.len());

        let mut x = Array1::zeros(n);
        let mut residual_history = Vec::new();

        let row_norms = self.compute_row_norms(system_matrix);
        let col_norms = self.compute_col_norms(system_matrix);
        let regularizer = ModelRegularizer3D::new(self.config.regularization);

        for iteration in 0..self.config.max_iterations {
            match self.config.algorithm {
                SirtAlgorithm::Sirt => {
                    x = self.sirt_iteration(
                        system_matrix,
                        &x,
                        sensor_data,
                        &row_norms,
                        &col_norms,
                    )?;
                }
                SirtAlgorithm::Art => {
                    self.art_iteration(system_matrix, &mut x, sensor_data, &row_norms)?;
                }
                SirtAlgorithm::Osem { num_subsets } => {
                    self.osem_iteration(
                        system_matrix,
                        &mut x,
                        sensor_data,
                        num_subsets,
                        &row_norms,
                    )?;
                }
            }

            if self.config.regularization.is_active() {
                let mut image_3d = self.reshape_to_3d(&x, grid_size);
                let model_3d = Array3::zeros(grid_size);
                regularizer.apply_to_gradient(&mut image_3d, &model_3d);
                x = self.reshape_to_1d(&image_3d);
            }

            let residual = system_matrix.dot(&x) - sensor_data;
            let residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
            residual_history.push(residual_norm);

            if self.config.verbose {
                debug!(
                    "Iteration {}: residual = {:.6e}",
                    iteration + 1,
                    residual_norm
                );
            }

            if iteration > 0 {
                let rel_change = (residual_history[iteration] - residual_history[iteration - 1])
                    .abs()
                    / (residual_history[iteration - 1] + 1e-12);

                if rel_change < self.config.min_relative_change
                    || residual_norm < self.config.tolerance
                {
                    if self.config.verbose {
                        debug!("Converged after {} iterations", iteration + 1);
                    }
                    return Ok(SirtResult {
                        image: self.reshape_to_3d(&x, grid_size),
                        iterations: iteration + 1,
                        final_residual: residual_norm,
                        residual_history,
                        converged: true,
                        computation_time: start_time.elapsed().as_secs_f64(),
                    });
                }
            }
        }

        let final_residual = residual_history.last().copied().unwrap_or(0.0);
        Ok(SirtResult {
            image: self.reshape_to_3d(&x, grid_size),
            iterations: self.config.max_iterations,
            final_residual,
            residual_history,
            converged: false,
            computation_time: start_time.elapsed().as_secs_f64(),
        })
    }

    // ==================== Algorithm Implementations ====================

    /// SIRT iteration: `x^(k+1) = x^(k) + λ·(D_R · A^T · (b − A·x^(k)))`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn sirt_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        b: &Array1<f64>,
        _row_norms: &Array1<f64>,
        col_norms: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let mut x_new = x.clone();
        let residual = b - &a.dot(x);
        let backproj = a.t().dot(&residual);

        for (j, &col_norm) in col_norms.iter().enumerate() {
            if col_norm > 1e-12 {
                x_new[j] += self.config.relaxation_factor * backproj[j] / col_norm;
            }
        }

        Ok(x_new)
    }

    /// ART iteration: `x^(k+1) = x^(k) + (λ/‖aᵢ‖²)·aᵢᵀ·(bᵢ − aᵢ·x^(k))`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn art_iteration(
        &self,
        a: &Array2<f64>,
        x: &mut Array1<f64>,
        b: &Array1<f64>,
        row_norms: &Array1<f64>,
    ) -> KwaversResult<()> {
        let (m, _n) = a.dim();

        for i in 0..m {
            let row = a.row(i);
            let row_norm_sq = row_norms[i] * row_norms[i];

            if row_norm_sq > 1e-12 {
                let residual = b[i] - row.dot(x);
                let update = self.config.relaxation_factor * residual / row_norm_sq;

                for (j, &a_ij) in row.iter().enumerate() {
                    x[j] += update * a_ij;
                }
            }
        }

        Ok(())
    }

    /// OSEM iteration: ordered subsets EM.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn osem_iteration(
        &self,
        a: &Array2<f64>,
        x: &mut Array1<f64>,
        b: &Array1<f64>,
        num_subsets: usize,
        row_norms: &Array1<f64>,
    ) -> KwaversResult<()> {
        let (m, _n) = a.dim();
        let subset_size = m.div_ceil(num_subsets);

        for subset_idx in 0..num_subsets {
            let start_row = subset_idx * subset_size;
            let end_row = ((subset_idx + 1) * subset_size).min(m);

            for i in start_row..end_row {
                let row = a.row(i);
                let row_norm_sq = row_norms[i] * row_norms[i];

                if row_norm_sq > 1e-12 {
                    let residual = b[i] - row.dot(x);
                    let update = self.config.relaxation_factor * residual / row_norm_sq;

                    for (j, &a_ij) in row.iter().enumerate() {
                        x[j] += update * a_ij;
                    }
                }
            }
        }

        Ok(())
    }

    // ==================== Helper Functions ====================

    fn compute_row_norms(&self, a: &Array2<f64>) -> Array1<f64> {
        let (m, _n) = a.dim();
        let mut norms = Array1::zeros(m);
        for i in 0..m {
            norms[i] = a.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
        }
        norms
    }

    fn compute_col_norms(&self, a: &Array2<f64>) -> Array1<f64> {
        let (_m, n) = a.dim();
        let mut norms = Array1::zeros(n);
        for j in 0..n {
            norms[j] = a.column(j).iter().map(|x| x * x).sum::<f64>().sqrt();
        }
        norms
    }

    fn reshape_to_3d(&self, x: &Array1<f64>, grid_size: (usize, usize, usize)) -> Array3<f64> {
        let mut img = Array3::zeros(grid_size);
        for (idx, &val) in x.iter().enumerate() {
            let (i, j, k) = self.linear_to_3d(idx, grid_size);
            img[[i, j, k]] = val;
        }
        img
    }

    fn reshape_to_1d(&self, img: &Array3<f64>) -> Array1<f64> {
        let (nx, ny, nz) = img.dim();
        let mut x = Array1::zeros(nx * ny * nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = self.to_linear_index(i, j, k, (nx, ny, nz));
                    x[idx] = img[[i, j, k]];
                }
            }
        }
        x
    }

    fn linear_to_3d(&self, idx: usize, grid_size: (usize, usize, usize)) -> (usize, usize, usize) {
        let (_nx, ny, nz) = grid_size;
        let i = idx / (ny * nz);
        let j = (idx % (ny * nz)) / nz;
        let k = idx % nz;
        (i, j, k)
    }

    fn to_linear_index(
        &self,
        i: usize,
        j: usize,
        k: usize,
        grid_size: (usize, usize, usize),
    ) -> usize {
        i * grid_size.1 * grid_size.2 + j * grid_size.2 + k
    }
}
