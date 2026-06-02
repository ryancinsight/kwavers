//! FWI inversion drivers: single-source `invert`, multi-source variants, shot-gradient dispatch.

mod multi_source;
mod shot_gradient;

use super::{geometry::FwiGeometry, FwiProcessor};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_domain::grid::Grid;
use ndarray::Array3;

impl FwiProcessor {
    /// Perform Full Waveform Inversion (single-source).
    ///
    /// Minimizes the configured data misfit `J(d_syn, d_obs)` (default L2
    /// least-squares `J = (dt/2) Σ_{r,t} (d_syn − d_obs)²`; see
    /// [`FwiProcessor::with_misfit`](super::super::FwiProcessor::with_misfit) for
    /// the cycle-skipping-robust envelope / phase / Wasserstein alternatives) by
    /// gradient descent with max-norm normalization and Armijo line search. The
    /// objective, convergence test, line search, and adjoint source all use the
    /// same selected functional.
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
            let (objective, updated_model, step_size) =
                self.descent_update(&current_model, observed_data, geometry, grid)?;

            if let Some(previous) = prev_objective {
                let relative_change = (previous - objective).abs() / previous.max(f64::EPSILON);
                if relative_change < self.parameters.tolerance {
                    log::info!(
                        "FWI converged after {iteration} iterations with objective: {objective:.6e}"
                    );
                    break;
                }
            }

            if step_size == 0.0 {
                log::info!(
                    "FWI stalled at iter {iteration}: line search returned no descent step \
                     (J={objective:.6e})"
                );
                break;
            }

            log::info!("FWI iter {iteration}: J={objective:.6e} step={step_size:.6e}");
            current_model = updated_model;
            prev_objective = Some(objective);
        }

        Ok(current_model)
    }

    /// One gradient-descent step shared by every time-domain FWI driver.
    ///
    /// Runs the forward model at `current_model`, evaluates the configured
    /// misfit objective, back-propagates the misfit-specific adjoint source to
    /// form the smoothed and regularized gradient, normalizes by its max-norm,
    /// and takes an Armijo line-search step. Returns
    /// `(objective_at_current_model, updated_model, step_size)`; when the line
    /// search finds no descent (`step_size == 0`), `updated_model` equals the
    /// input model.
    ///
    /// This is the single source of truth for the per-iteration update used by
    /// [`Self::invert`], [`Self::invert_multiscale`](super::FwiProcessor), and
    /// [`Self::invert_encoded`](super::FwiProcessor).
    /// # Errors
    /// - Propagates any [`KwaversError`] from the forward/adjoint solve, the
    ///   misfit evaluation, regularization, or the line search.
    pub(in crate::inverse::fwi::time_domain) fn descent_update(
        &self,
        current_model: &Array3<f64>,
        observed_data: &ndarray::Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<(f64, Array3<f64>, f64)> {
        let (synthetic_data, forward_history) =
            self.forward_model(current_model, geometry, grid)?;
        let objective = self.compute_misfit_objective(observed_data, &synthetic_data)?;

        let residual = self.compute_adjoint_source(observed_data, &synthetic_data)?;
        let adjoint_source = self.build_adjoint_source(&residual, geometry)?;
        let gradient = self.adjoint_model(
            &adjoint_source,
            current_model,
            grid,
            &forward_history,
            geometry.source.p_mask.as_ref(),
        )?;
        let smoothed_gradient = self.smooth_gradient(&gradient);
        let regularized_gradient = self.apply_regularization(&smoothed_gradient, current_model)?;

        let grad_max = regularized_gradient
            .iter()
            .copied()
            .fold(0.0_f64, |a, x| a.max(x.abs()));
        let normalized_gradient = if grad_max > f64::EPSILON {
            &regularized_gradient / grad_max
        } else {
            regularized_gradient
        };

        let step_size =
            self.line_search(current_model, &normalized_gradient, observed_data, geometry, grid)?;

        if step_size == 0.0 {
            return Ok((objective, current_model.clone(), 0.0));
        }

        let mut updated_model = current_model - &(&normalized_gradient * step_size);
        self.apply_model_constraints(&mut updated_model);
        Ok((objective, updated_model, step_size))
    }
}
