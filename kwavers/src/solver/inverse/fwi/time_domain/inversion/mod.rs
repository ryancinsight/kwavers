//! FWI inversion drivers: single-source `invert`, multi-source variants, shot-gradient dispatch.

mod multi_source;
mod shot_gradient;

use super::{geometry::FwiGeometry, FwiProcessor};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use ndarray::Array3;

impl FwiProcessor {
    /// Perform Full Waveform Inversion (single-source).
    ///
    /// Minimizes `J(c) = (dt/2) Σ_{r,t} (d_syn − d_obs)²` by gradient descent
    /// with max-norm normalization and Armijo line search.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn invert(
        &self,
        observed_data: &ndarray::Array2<f64>,
        initial_model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        geometry.validate(grid, self.parameters.nt)?;
        if self.parameters.nt < 3 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires at least 3 time samples to form a second derivative"
                        .to_owned(),
                },
            ));
        }

        let mut current_model = initial_model.clone();
        self.apply_model_constraints(&mut current_model);
        let mut prev_objective: Option<f64> = None;
        let max_iterations = self.parameters.max_iterations;

        for iteration in 0..max_iterations {
            let (synthetic_data, forward_history) =
                self.forward_model(&current_model, geometry, grid)?;
            let objective = self.compute_l2_objective(observed_data, &synthetic_data)?;

            if let Some(previous) = prev_objective {
                let relative_change = (previous - objective).abs() / previous.max(f64::EPSILON);
                if relative_change < self.parameters.tolerance {
                    log::info!(
                        "FWI converged after {} iterations with objective: {:.6e}",
                        iteration,
                        objective
                    );
                    break;
                }
            }

            let residual = self.compute_adjoint_source(observed_data, &synthetic_data)?;
            let adjoint_source = self.build_adjoint_source(&residual, geometry)?;
            let gradient = self.adjoint_model(
                &adjoint_source,
                &current_model,
                grid,
                &forward_history,
                geometry.source.p_mask.as_ref(),
            )?;
            let smoothed_gradient = self.smooth_gradient(&gradient);
            let regularized_gradient =
                self.apply_regularization(&smoothed_gradient, &current_model)?;

            let grad_max = regularized_gradient
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.max(x.abs()));
            let grad_min = regularized_gradient
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.min(x));
            log::info!(
                "FWI iter {} objective={:.6e} grad_max={:.6e} grad_min={:.6e}",
                iteration,
                objective,
                grad_max,
                grad_min
            );
            let normalized_gradient = if grad_max > f64::EPSILON {
                &regularized_gradient / grad_max
            } else {
                regularized_gradient
            };

            let step_size = self.line_search(
                &current_model,
                &normalized_gradient,
                observed_data,
                geometry,
                grid,
            )?;
            log::info!("FWI iter {} step_size={:.6e}", iteration, step_size);

            if step_size == 0.0 {
                log::info!(
                    "FWI stalled at iter {}: line search returned no descent step (J={:.6e})",
                    iteration,
                    objective
                );
                break;
            }

            current_model = &current_model - &(&normalized_gradient * step_size);
            self.apply_model_constraints(&mut current_model);
            let c_min_after = current_model.iter().copied().fold(f64::INFINITY, f64::min);
            let c_max_after = current_model
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            log::info!(
                "FWI iter {} model_range=[{:.1},{:.1}]",
                iteration,
                c_min_after,
                c_max_after
            );
            prev_objective = Some(objective);

            log::debug!(
                "FWI iteration {}: objective = {:.6e}, step_size = {:.6e}",
                iteration,
                objective,
                step_size
            );
        }

        Ok(current_model)
    }
}
