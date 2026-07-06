//! Preconditioned conjugate-gradient driver loop for the diffusion system
//! `A Φ = S` produced by [`super::operator`] / [`super::preconditioner`].

use anyhow::Result;
use leto::Array3 as LetoArray3;
use ndarray::Array3;

use super::{DiffusionSolver, DiffusionVolume};

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
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn solve(&self, source: &Array3<f64>) -> Result<Array3<f64>> {
        self.solve_volume(source)
    }

    /// Solve steady-state diffusion equation for a Leto source distribution.
    ///
    /// # Errors
    /// - Returns [`Err`] if the source shape differs from the solver grid or
    ///   if the PCG iteration does not converge within `max_iterations`.
    pub fn solve_leto(&self, source: &LetoArray3<f64>) -> Result<LetoArray3<f64>> {
        self.solve_volume(source)
    }

    fn solve_volume<V>(&self, source: &V) -> Result<V>
    where
        V: DiffusionVolume,
    {
        let (nx, ny, nz) = self.grid.dimensions();

        if source.shape3() != [nx, ny, nz] {
            anyhow::bail!(
                "Source dimensions {:?} do not match grid dimensions ({}, {}, {})",
                source.shape3(),
                nx,
                ny,
                nz
            );
        }

        let shape = [nx, ny, nz];
        let mut fluence = V::zeros(shape);
        let mut residual = source.clone();

        let preconditioner = self.compute_preconditioner_volume::<V>();
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
            let a_times_p = self.apply_operator_volume(&search_direction);
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

    fn dot<V>(left: &V, right: &V) -> f64
    where
        V: DiffusionVolume,
    {
        let [nx, ny, nz] = left.shape3();
        let mut sum = 0.0;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    sum += left.value(index) * right.value(index);
                }
            }
        }
        sum
    }

    fn mul_elementwise<V>(left: &V, right: &V) -> V
    where
        V: DiffusionVolume,
    {
        let [nx, ny, nz] = left.shape3();
        let mut result = V::zeros([nx, ny, nz]);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    result.set_value(index, left.value(index) * right.value(index));
                }
            }
        }
        result
    }

    fn add_scaled_in_place<V>(target: &mut V, source: &V, scale: f64)
    where
        V: DiffusionVolume,
    {
        let [nx, ny, nz] = target.shape3();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    target.set_value(index, target.value(index) + scale * source.value(index));
                }
            }
        }
    }

    fn combine<V>(left: &V, right: &V, right_scale: f64) -> V
    where
        V: DiffusionVolume,
    {
        let [nx, ny, nz] = left.shape3();
        let mut result = V::zeros([nx, ny, nz]);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = [i, j, k];
                    result.set_value(index, left.value(index) + right_scale * right.value(index));
                }
            }
        }
        result
    }
}
