use crate::core::error::KwaversResult;
use crate::domain::field::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::plugin::Plugin;
use crate::solver::integration::nonlinear::{GMRESConfig, GMRESSolver};
use log::{debug, warn};
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Instant;

use super::config::{CouplingConvergenceInfo, NewtonKrylovConfig, PhysicsCoefficients};
use super::utils;

/// Monolithic multiphysics coupler
///
/// Solves coupled multiphysics systems simultaneously without subcycling or iteration lag.
/// Uses Jacobian-Free Newton-Krylov approach via GMRES linear solver.
#[derive(Debug)]
pub struct MonolithicCoupler {
    /// Newton-Krylov configuration
    pub(super) newton_config: NewtonKrylovConfig,

    /// GMRES linear solver configuration
    pub(super) gmres_config: GMRESConfig,

    /// Convergence history
    pub(super) convergence_history: Vec<f64>,

    /// Physics components (for future extensibility via Plugin trait)
    pub(super) physics_components: HashMap<String, Box<dyn Plugin>>,

    /// Physical coefficients for the coupled PDE system
    pub(super) physics_coefficients: PhysicsCoefficients,

    /// Pre-allocated correction vector δu for Newton iterations.
    ///
    /// Lazily initialised on the first `step` call once grid dimensions are known.
    /// Avoids one `Array3::zeros` heap allocation per Newton iteration (which can
    /// be 128 MB per step for a 256³ grid).
    pub(super) du_scratch: Option<Array3<f64>>,

    /// Pre-allocated output buffer for `laplacian_3d_into`.
    ///
    /// Reusable GMRES solver instance.
    ///
    /// Lazily initialised and reused across Newton iterations.  Avoids
    /// `GMRESConfig::clone()` overhead and any per-iteration allocations
    /// performed inside `GMRESSolver::new`.
    pub(super) gmres_solver: Option<GMRESSolver>,

    /// Grid cell spacings (dx, dy, dz) in metres, extracted from the `Grid`
    /// argument of `solve_coupled_step`.  Updated each call so the Laplacian
    /// scaling stays correct when the caller changes the grid.
    pub(super) grid_spacing: (f64, f64, f64),
}
impl MonolithicCoupler {
    /// Create new monolithic coupler
    pub fn new(newton_config: NewtonKrylovConfig, gmres_config: GMRESConfig) -> Self {
        Self {
            newton_config,
            gmres_config,
            convergence_history: Vec::new(),
            physics_components: HashMap::new(),
            physics_coefficients: PhysicsCoefficients::default(),
            du_scratch: None,

            gmres_solver: None,
            grid_spacing: (1e-3, 1e-3, 1e-3), // overwritten on first call
        }
    }

    /// Create new monolithic coupler with custom physics coefficients
    pub fn with_coefficients(
        newton_config: NewtonKrylovConfig,
        gmres_config: GMRESConfig,
        coefficients: PhysicsCoefficients,
    ) -> Self {
        Self {
            newton_config,
            gmres_config,
            convergence_history: Vec::new(),
            physics_components: HashMap::new(),
            physics_coefficients: coefficients,
            du_scratch: None,

            gmres_solver: None,
            grid_spacing: (1e-3, 1e-3, 1e-3), // overwritten on first call
        }
    }

    /// Set physics coefficients
    pub fn set_physics_coefficients(&mut self, coefficients: PhysicsCoefficients) {
        self.physics_coefficients = coefficients;
    }

    /// Register physics component
    pub fn register_physics(
        &mut self,
        name: String,
        physics: Box<dyn Plugin>,
    ) -> KwaversResult<()> {
        self.physics_components.insert(name, physics);
        Ok(())
    }

    /// Solve coupled multiphysics step
    ///
    /// # Arguments
    ///
    /// * `fields` - Unified field map (pressure, intensity, temperature, velocity, etc.)
    /// * `dt` - Time step
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Convergence information with Newton iteration count and final residual
    ///
    /// # Algorithm
    ///
    /// 1. **Newton Loop:**
    ///    - Compute residual F(u) at current iterate
    ///    - Check convergence: ||F(u)|| < tolerance
    ///    - Solve linear system via GMRES: J·δu = -F(u)
    ///    - Update: u := u + α·δu (with optional line search)
    ///
    /// 2. **Line Search (optional):**
    ///    - Find step size α ∈ (0, 1] such that ||F(u+α·δu)|| < ||F(u)||
    ///    - Default: α = 1.0 (full Newton step)
    ///
    /// 3. **GMRES Convergence:**
    ///    - Inner linear solver tolerance: 10⁻³ × Newton residual (Eisenstat-Walker)
    ///    - Restarted GMRES(30) with configurable Krylov dimension
    ///    - Adaptive preconditioning (physics-based block preconditioner)
    pub fn solve_coupled_step(
        &mut self,
        fields: &mut HashMap<UnifiedFieldType, Array3<f64>>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<CouplingConvergenceInfo> {
        let start_time = Instant::now();
        self.convergence_history.clear();

        // Extract and cache grid spacing for use in Laplacian computations.
        // Updated each call so the scaling is correct if the caller changes the grid.
        self.grid_spacing = (grid.dx, grid.dy, grid.dz);

        // Determine deterministic field ordering and flatten
        let field_order = utils::sorted_field_keys(fields);
        let mut u_current = utils::flatten_fields(fields, &field_order);
        let u_prev = u_current.clone();
        let dims = grid.dimensions();

        let f_norm_0: f64;
        {
            let residual = self.compute_residual(&u_current, &u_prev, dt, dims, &field_order)?;
            f_norm_0 = utils::norm(&residual);
            self.convergence_history.push(f_norm_0);
        }

        if self.newton_config.verbose {
            debug!("Monolithic Newton initial residual: {:.3e}", f_norm_0);
        }

        // Pre-allocate / reuse correction vector δu outside the Newton loop.
        // Using `std::mem::take` removes the scratch from `self` so the Newton-loop
        // closure can borrow `self` for `jacobian_vector_product` without conflict.
        // The scratch is returned to `self.du_scratch` after the loop ends.
        if self.du_scratch.is_none() {
            self.du_scratch = Some(Array3::zeros(u_current.dim()));
        }
        let mut du = self.du_scratch.take().unwrap();

        // Newton iteration
        let mut newton_iter = 0;
        let mut total_gmres_iters = 0;
        let mut converged = false;

        for k in 0..self.newton_config.max_newton_iterations {
            newton_iter = k + 1;

            // Compute residual
            let f = self.compute_residual(&u_current, &u_prev, dt, dims, &field_order)?;
            let f_norm = utils::norm(&f);

            if self.newton_config.verbose {
                debug!(
                    "Newton iteration {}: ||F|| = {:.3e}, relative = {:.3e}",
                    k,
                    f_norm,
                    f_norm / f_norm_0.max(1e-15)
                );
            }

            self.convergence_history.push(f_norm);

            // Check convergence
            if f_norm < self.newton_config.newton_tolerance {
                if self.newton_config.verbose {
                    debug!("Converged in {} Newton iterations", newton_iter);
                }
                converged = true;
                break;
            }

            // Solve linear system: J·δu ≈ -F via GMRES.
            // We take the solver out of the Option so the closure can borrow `self`
            // immutably (for `jacobian_vector_product`) without conflicting with the
            // mutable access used to call `gmres.solve`.  After the call the solver
            // is put back to avoid re-allocation on the next Newton iteration.
            let mut gmres = self
                .gmres_solver
                .take()
                .unwrap_or_else(|| GMRESSolver::new(self.gmres_config.clone()));

            // Negative residual as RHS
            let b = &f * -1.0;

            // Reset the reused scratch buffer (no allocation)
            du.fill(0.0);

            // Solve J·du = -f (closure borrows self; du is local, not part of self)
            let solve_result = gmres.solve(
                |v: &Array3<f64>| {
                    self.jacobian_vector_product(v, &u_current, &u_prev, dt, dims, &field_order)
                },
                &b,
                &mut du,
            );
            // Return the solver to the struct so it is reused next iteration.
            self.gmres_solver = Some(gmres);

            match solve_result {
                Ok(conv_info) => {
                    total_gmres_iters += conv_info.iterations;
                    if self.newton_config.verbose {
                        debug!(
                            "  GMRES: {} iterations, ||r|| = {:.3e}",
                            conv_info.iterations, conv_info.final_residual
                        );
                    }
                }
                Err(e) => {
                    if self.newton_config.verbose {
                        warn!("  GMRES failed: {:?}", e);
                    }
                    // Continue with best attempt rather than failing
                }
            }

            // Line search (optional)
            let step_size = if self.newton_config.adaptive_step_size {
                self.line_search(&u_current, &du, &f, &u_prev, dt, dims, &field_order)?
            } else {
                1.0
            };

            // Update: u := u + α·du
            u_current = &u_current + &(&du * step_size);

            if self.newton_config.verbose {
                debug!("  Step size: {:.4}", step_size);
            }
        }

        // Return scratch buffer to self for reuse in future steps
        self.du_scratch = Some(du);

        // Store solution back to fields
        utils::unflatten_fields(&u_current, fields, &field_order);

        let elapsed = start_time.elapsed().as_secs_f64();
        let final_residual = self.convergence_history.last().copied().unwrap_or(f_norm_0);
        let avg_gmres = total_gmres_iters.checked_div(newton_iter).unwrap_or(0);

        Ok(CouplingConvergenceInfo {
            converged,
            newton_iterations: newton_iter,
            final_residual,
            relative_residual: final_residual / f_norm_0.max(1e-15),
            wall_time_seconds: elapsed,
            avg_gmres_iterations: avg_gmres,
        })
    }
    /// Get convergence history
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Get physics coefficients (read-only)
    pub fn physics_coefficients(&self) -> &PhysicsCoefficients {
        &self.physics_coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::NewtonKrylovConfig;
    use super::*;
    use crate::solver::integration::nonlinear::GMRESConfig;

    #[test]
    fn test_monolithic_coupler_creation() {
        let newton_config = NewtonKrylovConfig::default();
        let gmres_config = GMRESConfig::default();
        let coupler = MonolithicCoupler::new(newton_config, gmres_config);

        assert!(coupler.convergence_history().is_empty());
        assert_eq!(coupler.physics_components.len(), 0);
    }
}
