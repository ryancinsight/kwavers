//! Preconditioned conjugate-gradient driver loop for the diffusion system
//! `A Φ = S` produced by [`super::operator`] / [`super::preconditioner`].

use anyhow::Result;
use leto::Array3;

use super::DiffusionSolver;

impl DiffusionSolver {
    /// Solve steady-state diffusion equation for given source distribution.
    ///
    /// # Arguments
    ///
    /// - `source`: Isotropic source term `S(r)` in W/m³.
    ///
    /// # Returns
    ///
    /// Optical fluence field `Φ(r)` in W/m².
    ///
    /// # Algorithm
    ///
    /// Preconditioned conjugate gradient (PCG) with Jacobi preconditioner:
    /// 1. Discretize PDE into linear system `Ax = b`.
    /// 2. Iterate `x_{k+1} = x_k + α_k p_k` until `‖r_k‖ < tol`.
    /// 3. Apply extrapolated boundary conditions at domain boundaries via
    ///    `DiffusionSolver::apply_operator`.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn solve(&self, source: &Array3<f64>) -> Result<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();

        if source.shape() != [nx, ny, nz] {
            anyhow::bail!(
                "Source dimensions {:?} do not match grid dimensions ({}, {}, {})",
                source.shape(),
                nx,
                ny,
                nz
            );
        }

        let shape = [nx, ny, nz];
        let mut fluence = Array3::zeros(shape);
        let mut residual = source.clone();

        let preconditioner = self.compute_preconditioner();
        let mut preconditioned_residual = Self::mul_elementwise(&residual, &preconditioner);
        let mut search_direction = preconditioned_residual.clone();

        let mut residual_dot_z = Self::dot(&residual, &preconditioned_residual);
        let initial_residual_norm = residual_dot_z.sqrt();

        if self.config.verbose {
            tracing::info!(
                "DiffusionSolver: Initial residual = {:.6e}",
                initial_residual_norm
            );
        }

        for iter in 0..self.config.max_iterations {
            let a_times_p = self.apply_operator(&search_direction);
            let p_dot_ap = Self::dot(&search_direction, &a_times_p);

            if p_dot_ap.abs() < 1e-30 {
                if self.config.verbose {
                    tracing::warn!(
                        "DiffusionSolver: Near-zero denominator at iteration {}",
                        iter
                    );
                }
                break;
            }

            let alpha = residual_dot_z / p_dot_ap;
            Self::add_scaled_in_place(&mut fluence, &search_direction, alpha);
            Self::add_scaled_in_place(&mut residual, &a_times_p, -alpha);

            let residual_norm = Self::dot(&residual, &residual).sqrt();
            let relative_residual = residual_norm / (initial_residual_norm + 1e-30);

            if self.config.verbose && iter % 100 == 0 {
                tracing::debug!(
                    "DiffusionSolver: Iteration {}, relative residual = {:.6e}",
                    iter,
                    relative_residual
                );
            }

            if relative_residual < self.config.tolerance {
                if self.config.verbose {
                    tracing::info!(
                        "DiffusionSolver: Converged in {} iterations (residual = {:.6e})",
                        iter + 1,
                        relative_residual
                    );
                }
                return Ok(fluence);
            }

            preconditioned_residual = Self::mul_elementwise(&residual, &preconditioner);

            let residual_dot_z_new = Self::dot(&residual, &preconditioned_residual);
            let beta = residual_dot_z_new / residual_dot_z;
            residual_dot_z = residual_dot_z_new;

            search_direction = Self::combine(&preconditioned_residual, &search_direction, beta);
        }

        anyhow::bail!(
            "DiffusionSolver: Failed to converge in {} iterations",
            self.config.max_iterations
        )
    }

    fn dot(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
        let [nx, ny, nz] = left.shape();
        let mut sum = 0.0;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    sum += left[index] * right[index];
                }
            }
        }
        sum
    }

    fn mul_elementwise(left: &Array3<f64>, right: &Array3<f64>) -> Array3<f64> {
        let [nx, ny, nz] = left.shape();
        let mut result = Array3::zeros([nx, ny, nz]);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    result[index] = left[index] * right[index];
                }
            }
        }
        result
    }

    fn add_scaled_in_place(target: &mut Array3<f64>, source: &Array3<f64>, scale: f64) {
        let [nx, ny, nz] = target.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    target[index] += scale * source[index];
                }
            }
        }
    }

    fn combine(left: &Array3<f64>, right: &Array3<f64>, right_scale: f64) -> Array3<f64> {
        let [nx, ny, nz] = left.shape();
        let mut result = Array3::zeros([nx, ny, nz]);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    result[index] = left[index] + right_scale * right[index];
                }
            }
        }
        result
    }
}
