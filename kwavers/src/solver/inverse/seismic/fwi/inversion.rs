//! Inversion drivers: single-source `invert`, multi-source variants, shot-gradient dispatch.

use super::{geometry::FwiGeometry, gradient::mute_gradient_near_sources, FwiProcessor};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use ndarray::{Array2, Array3, Zip};

impl FwiProcessor {
    /// Perform Full Waveform Inversion (single-source).
    ///
    /// Minimizes `J(c) = (dt/2) Σ_{r,t} (d_syn − d_obs)²` by gradient descent
    /// with max-norm normalization and Armijo line search.
    pub fn invert(
        &self,
        observed_data: &Array2<f64>,
        initial_model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        geometry.validate(grid, self.parameters.nt)?;
        if self.parameters.nt < 3 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires at least 3 time samples to form a second derivative"
                        .to_string(),
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

    /// Multi-source FWI inversion.
    ///
    /// Joint objective: `J(c) = Σᵢ Jᵢ(c)` where each `Jᵢ` is the per-shot
    /// acoustic L2 misfit.  The reduced gradient is the sum of per-shot
    /// adjoint-state gradients.
    ///
    /// # References
    /// - Marquet et al. (2013). *Phys. Med. Biol.* 58, 2937.
    /// - Guasch et al. (2020). *npj Digital Medicine* 3, 28.
    pub fn invert_multi_source(
        &self,
        shots: &[(FwiGeometry, Array2<f64>)],
        initial_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        if shots.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_multi_source requires at least one shot".to_string(),
                },
            ));
        }
        for (geometry, _) in shots {
            geometry.validate(grid, self.parameters.nt)?;
        }

        let (nx, ny, nz) = grid.dimensions();
        let mut current_model = initial_model.clone();
        self.apply_model_constraints(&mut current_model);

        for iteration in 0..self.parameters.max_iterations {
            let mut total_objective = 0.0_f64;
            let mut total_gradient = Array3::<f64>::zeros((nx, ny, nz));
            for (geometry, observed_data) in shots.iter() {
                let (obj, grad) =
                    self.compute_shot_gradient(&current_model, geometry, observed_data, grid)?;
                total_objective += obj;
                Zip::from(&mut total_gradient)
                    .and(&grad)
                    .par_for_each(|a, &b| *a += b);
            }

            let smoothed = self.smooth_gradient(&total_gradient);
            let regularized = self.apply_regularization(&smoothed, &current_model)?;

            let grad_max = regularized
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.max(x.abs()));
            log::info!(
                "FWI multi-source iter {} joint_J={:.6e} grad_max={:.6e}",
                iteration,
                total_objective,
                grad_max
            );

            let mut normalized = regularized;
            if grad_max > f64::EPSILON {
                normalized.mapv_inplace(|g| g / grad_max);
            }

            let step_size = self.line_search_multi(&current_model, &normalized, shots, grid)?;
            log::info!(
                "FWI multi-source iter {} step_size={:.6e}",
                iteration,
                step_size
            );

            if step_size == 0.0 {
                log::info!(
                    "FWI multi-source stalled at iter {}: J={:.6e}",
                    iteration,
                    total_objective
                );
                break;
            }

            Zip::from(&mut current_model)
                .and(&normalized)
                .par_for_each(|c, &g| *c -= g * step_size);
            self.apply_model_constraints(&mut current_model);

            let c_max = current_model
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let c_min = current_model.iter().copied().fold(f64::INFINITY, f64::min);
            log::info!(
                "FWI multi-source iter {} model_range=[{:.1},{:.1}]",
                iteration,
                c_min,
                c_max
            );
        }

        Ok(current_model)
    }

    /// Multi-source FWI with a frozen (skull) mask for brain tissue imaging.
    ///
    /// Identical to [`invert_multi_source`] except that:
    /// - Voxels where `frozen_mask` is `true` are never updated.
    /// - Brain voxel velocity is clamped to `[c_min, c_max]` after each iteration.
    ///
    /// Reference: Guasch (2020) §Methods "Brain FWI".
    pub fn invert_multi_source_masked(
        &self,
        shots: &[(FwiGeometry, Array2<f64>)],
        initial_model: &Array3<f64>,
        reference_model: &Array3<f64>,
        frozen_mask: &Array3<bool>,
        c_min: f64,
        c_max: f64,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        if shots.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_multi_source_masked requires at least one shot".to_string(),
                },
            ));
        }
        for (geometry, _) in shots {
            geometry.validate(grid, self.parameters.nt)?;
        }

        let (nx, ny, nz) = grid.dimensions();
        let mut current_model = initial_model.clone();

        Zip::from(&mut current_model)
            .and(frozen_mask)
            .and(reference_model)
            .par_for_each(|c, &frozen, &r| {
                if frozen {
                    *c = r;
                } else {
                    *c = c.clamp(c_min, c_max);
                }
            });

        for iteration in 0..self.parameters.max_iterations {
            let mut total_objective = 0.0_f64;
            let mut total_gradient = Array3::<f64>::zeros((nx, ny, nz));
            for (geometry, observed_data) in shots.iter() {
                let (obj, grad) =
                    self.compute_shot_gradient(&current_model, geometry, observed_data, grid)?;
                total_objective += obj;
                Zip::from(&mut total_gradient)
                    .and(&grad)
                    .par_for_each(|a, &b| *a += b);
            }

            Zip::from(&mut total_gradient)
                .and(frozen_mask)
                .par_for_each(|g, &frozen| {
                    if frozen {
                        *g = 0.0;
                    }
                });

            let smoothed = self.smooth_gradient(&total_gradient);
            let mut regularized = self.apply_regularization(&smoothed, &current_model)?;

            // Re-zero skull voxels after smoothing to prevent smooth-gradient leakage
            // from triggering spurious CFL violations in the line search.
            Zip::from(&mut regularized)
                .and(frozen_mask)
                .par_for_each(|g, &frozen| {
                    if frozen {
                        *g = 0.0;
                    }
                });

            let grad_max = regularized
                .iter()
                .copied()
                .fold(0.0_f64, |a, x| a.max(x.abs()));
            log::info!(
                "FWI masked iter {} joint_J={:.6e} grad_max={:.6e}",
                iteration,
                total_objective,
                grad_max
            );

            let mut normalized = regularized;
            if grad_max > f64::EPSILON {
                normalized.mapv_inplace(|g| g / grad_max);
            }

            let step_size = self.line_search_multi(&current_model, &normalized, shots, grid)?;
            log::info!("FWI masked iter {} step_size={:.6e}", iteration, step_size);

            if step_size == 0.0 {
                log::info!(
                    "FWI masked stalled at iter {}: J={:.6e}",
                    iteration,
                    total_objective
                );
                break;
            }

            Zip::from(&mut current_model)
                .and(&normalized)
                .par_for_each(|c, &g| *c -= g * step_size);

            Zip::from(&mut current_model)
                .and(frozen_mask)
                .and(reference_model)
                .par_for_each(|c, &frozen, &r| {
                    if frozen {
                        *c = r;
                    } else {
                        *c = c.clamp(c_min, c_max);
                    }
                });

            let c_max_model = current_model
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let c_min_model = current_model.iter().copied().fold(f64::INFINITY, f64::min);
            log::info!(
                "FWI masked iter {} model_range=[{:.1},{:.1}]",
                iteration,
                c_min_model,
                c_max_model
            );
        }

        Ok(current_model)
    }

    /// Compute the per-shot objective and physics gradient for one shot gather.
    ///
    /// Returns `(Jᵢ, ∂Jᵢ/∂c)`.  Applies near-source gradient mute when
    /// `FwiParameters::source_mute_radius > 0`.
    pub(super) fn compute_shot_gradient(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        observed_data: &Array2<f64>,
        grid: &Grid,
    ) -> KwaversResult<(f64, Array3<f64>)> {
        let (synthetic_data, forward_history) = self.forward_model(model, geometry, grid)?;
        let objective = self.compute_l2_objective(observed_data, &synthetic_data)?;
        let residual = self.compute_adjoint_source(observed_data, &synthetic_data)?;
        let adjoint_source = self.build_adjoint_source(&residual, geometry)?;
        let mut gradient = self.adjoint_model(
            &adjoint_source,
            model,
            grid,
            &forward_history,
            geometry.source.p_mask.as_ref(),
        )?;

        if self.parameters.source_mute_radius > 0 {
            if let Some(p_mask) = geometry.source.p_mask.as_ref() {
                mute_gradient_near_sources(
                    &mut gradient,
                    p_mask,
                    self.parameters.source_mute_radius,
                );
            }
        }

        Ok((objective, gradient))
    }
}
