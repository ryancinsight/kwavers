use super::super::config::CouplingConvergenceInfo;
use super::super::residual_metric::norm;
use super::super::state_vector::{flatten_fields, sorted_field_keys, unflatten_fields};
use super::MonolithicCoupler;
use crate::integration::nonlinear::GMRESSolver;
use crate::workspace::inplace_ops::scale_inplace;
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use leto::Array3;
use log::{debug, warn};
use std::collections::HashMap;
use std::time::Instant;

impl MonolithicCoupler {
    /// Solve one coupled multiphysics step.
    ///
    /// # Arguments
    /// - `fields`: unified field map containing pressure, fluence,
    ///   temperature, and optional passive fields.
    /// - `dt`: positive finite time step in seconds.
    /// - `grid`: computational grid whose dimensions match every field.
    ///
    /// # Algorithm
    /// 1. Build the stacked Newton state `u` and previous-state snapshot.
    /// 2. Iterate Newton residual solves until `||F(u)|| < tolerance`.
    /// 3. Solve `J·δu = -F(u)` with GMRES using Jacobian-free products.
    /// 4. Apply either the full step or adaptive residual-checked line search.
    /// 5. Unpack the converged stacked state back into the field map.
    ///
    /// # Errors
    /// - Returns validation errors for invalid `dt`, Newton settings, empty
    ///   fields, or field/grid shape mismatches.
    /// - Propagates residual, line-search, or Jacobian-vector errors.
    pub fn solve_coupled_step(
        &mut self,
        fields: &mut HashMap<UnifiedFieldType, Array3<f64>>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<CouplingConvergenceInfo> {
        self.validate_solve_inputs(fields, dt, grid)?;

        let start_time = Instant::now();
        self.convergence_history.clear();
        self.grid_spacing = (grid.dx, grid.dy, grid.dz);

        let field_order = sorted_field_keys(fields);
        let mut u_current = flatten_fields(fields, &field_order);
        let mut u_prev = self
            .u_prev_scratch
            .take()
            .filter(|scratch| scratch.shape() == u_current.shape())
            .unwrap_or_else(|| Array3::zeros(u_current.shape()));
        u_prev.assign(&u_current);
        let dims = grid.dimensions();

        let f_norm_0 = {
            let residual = self.compute_residual(&u_current, &u_prev, dt, dims, &field_order)?;
            let norm = norm(&residual);
            self.convergence_history.push(norm);
            norm
        };

        if self.newton_config.verbose {
            debug!("Monolithic Newton initial residual: {:.3e}", f_norm_0);
        }

        if self.du_scratch.is_none() {
            self.du_scratch = Some(Array3::zeros(u_current.shape()));
        }
        let mut du = self.du_scratch.take().unwrap();
        let mut rhs_scratch = self.rhs_scratch.take();

        let mut newton_iter = 0;
        let mut total_gmres_iters = 0;
        let mut converged = false;

        for k in 0..self.newton_config.max_newton_iterations {
            newton_iter = k + 1;

            let f = self.compute_residual(&u_current, &u_prev, dt, dims, &field_order)?;
            let f_norm = norm(&f);

            if self.newton_config.verbose {
                debug!(
                    "Newton iteration {}: ||F|| = {:.3e}, relative = {:.3e}",
                    k,
                    f_norm,
                    f_norm / f_norm_0.max(1e-15)
                );
            }

            self.convergence_history.push(f_norm);

            if f_norm < self.newton_config.newton_tolerance {
                if self.newton_config.verbose {
                    debug!("Converged in {} Newton iterations", newton_iter);
                }
                converged = true;
                break;
            }

            let mut gmres = self
                .gmres_solver
                .take()
                .unwrap_or_else(|| GMRESSolver::new(self.gmres_config.clone()));

            let rhs = rhs_scratch.get_or_insert_with(|| Array3::zeros(f.shape()));
            if rhs.shape() != f.shape() {
                *rhs = Array3::zeros(f.shape());
            }
            rhs.assign(&f);
            scale_inplace(rhs, -1.0);

            du.fill(0.0);

            let solve_result = gmres.solve(
                |v: &Array3<f64>| {
                    self.jacobian_vector_product(v, &u_current, &u_prev, dt, dims, &field_order)
                },
                &*rhs,
                &mut du,
            );
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
                Err(error) => {
                    if self.newton_config.verbose {
                        warn!("  GMRES failed: {:?}", error);
                    }
                }
            }

            let step_size = if self.newton_config.adaptive_step_size {
                self.line_search(&u_current, &du, &f, &u_prev, dt, dims, &field_order)?
            } else {
                1.0
            };

            for (u_value, delta) in u_current.iter_mut().zip(du.iter()) {
                {
                    *u_value += step_size * delta;
                };
            }

            if self.newton_config.verbose {
                debug!("  Step size: {:.4}", step_size);
            }
        }

        self.du_scratch = Some(du);
        self.u_prev_scratch = Some(u_prev);
        self.rhs_scratch = rhs_scratch;

        unflatten_fields(&u_current, fields, &field_order);

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
}
