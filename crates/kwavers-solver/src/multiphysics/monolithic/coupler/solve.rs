use super::super::config::CouplingConvergenceInfo;
use super::super::residual::JacobianOperator;
use super::super::residual_metric::norm;
use super::super::state_vector::{flatten_fields, sorted_field_keys, unflatten_fields};
use super::MonolithicCoupler;
use crate::krylov::{GmresConvergenceInfo, KrylovWorkspace};
use crate::workspace::inplace_ops::scale_inplace;
use athena_core::Identity;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use leto::{Array1, Array3};
use log::{debug, warn};
use std::collections::HashMap;
use std::time::Instant;

/// Reinterpret a dense stacked state as the flat Krylov vector of the same
/// elements, and back.
///
/// Both shapes are dense row-major over one allocation, so the reinterpretation
/// moves no data: element `i` of the vector is element `i` of the stacked state
/// in row-major order, which is exactly the correspondence the residual and the
/// Jacobian operator already share.
fn into_flat(state: Array3<f64>) -> KwaversResult<Array1<f64>> {
    let [rows, columns, planes] = state.shape();
    let length = rows * columns * planes;
    state
        .into_shape([length])
        .map_err(|error| reshape_error(&error))
}

/// Inverse of [`into_flat`].
fn into_stacked(vector: Array1<f64>, shape: [usize; 3]) -> KwaversResult<Array3<f64>> {
    vector
        .into_shape(shape)
        .map_err(|error| reshape_error(&error))
}

fn reshape_error(error: &leto::LetoError) -> KwaversError {
    KwaversError::Numerical(NumericalError::MatrixDimension {
        operation: "monolithic Newton state reshape".to_owned(),
        expected: "dense row-major stacked state".to_owned(),
        actual: error.to_string(),
    })
}

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
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
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

            let stacked_shape = f.shape();
            let rhs = rhs_scratch.get_or_insert_with(|| Array3::zeros(stacked_shape));
            if rhs.shape() != stacked_shape {
                *rhs = Array3::zeros(stacked_shape);
            }
            rhs.assign(&f);
            scale_inplace(rhs, -1.0);

            du.fill(0.0);

            // Athena solves over flat vectors; the stacked state and the Krylov
            // vector are the same allocation viewed at two shapes.
            let right_hand_side = into_flat(
                rhs_scratch
                    .take()
                    .expect("invariant: right-hand side scratch was just populated"),
            )?;
            let mut correction = into_flat(du)?;

            let dimension = correction.len();
            let mut workspace = match self.krylov_workspace.take() {
                Some(workspace) if workspace.dimension() == dimension => workspace,
                _ => KrylovWorkspace::new(self.gmres_config.krylov_dim, dimension)?,
            };
            let policy = self.gmres_config.policy()?;

            let (solve_result, physics_failure) = {
                let operator =
                    JacobianOperator::new(self, &u_current, &u_prev, dt, dims, &field_order);
                let result = workspace.solve(
                    &operator,
                    &Identity,
                    &right_hand_side,
                    &mut correction,
                    policy,
                );
                (result, operator.take_failure())
            };
            self.krylov_workspace = Some(workspace);

            du = into_stacked(correction, stacked_shape)?;
            rhs_scratch = Some(into_stacked(right_hand_side, stacked_shape)?);

            // A residual that could not be evaluated is a physics failure, not
            // a convergence outcome: the iterate it would have produced does
            // not exist, so it propagates rather than being logged away.
            if let Some(error) = physics_failure {
                self.du_scratch = Some(du);
                self.u_prev_scratch = Some(u_prev);
                self.rhs_scratch = rhs_scratch;
                return Err(error);
            }

            let conv_info = GmresConvergenceInfo::from_report(&solve_result?);
            total_gmres_iters += conv_info.iterations;
            if !conv_info.converged {
                warn!(
                    "  GMRES did not converge: {} iterations, ||r|| = {:.3e}",
                    conv_info.iterations, conv_info.final_residual
                );
            } else if self.newton_config.verbose {
                debug!(
                    "  GMRES: {} iterations, ||r|| = {:.3e}",
                    conv_info.iterations, conv_info.final_residual
                );
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
