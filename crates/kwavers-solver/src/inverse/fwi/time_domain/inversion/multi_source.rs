//! Multi-source FWI inversion variants: standard and skull-masked brain imaging.

use super::super::{geometry::FwiGeometry, FwiProcessor};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_domain::grid::Grid;
use ndarray::{Array2, Array3, Zip};

impl FwiProcessor {
    /// Multi-source FWI inversion.
    ///
    /// Joint objective: `J(c) = Σᵢ Jᵢ(c)` where each `Jᵢ` is the per-shot
    /// acoustic L2 misfit.  The reduced gradient is the sum of per-shot
    /// adjoint-state gradients.
    ///
    /// # References
    /// - Marquet et al. (2013). *Phys. Med. Biol.* 58, 2937.
    /// - Guasch et al. (2020). *npj Digital Medicine* 3, 28.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn invert_multi_source(
        &self,
        shots: &[(FwiGeometry, Array2<f64>)],
        initial_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        if shots.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "invert_multi_source requires at least one shot".to_owned(),
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
            for (geometry, observed_data) in shots {
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
                normalized.par_mapv_inplace(|g| g / grad_max);
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
    /// Identical to [`Self::invert_multi_source`] except that:
    /// - Voxels where `frozen_mask` is `true` are never updated.
    /// - Brain voxel velocity is clamped to `[c_min, c_max]` after each iteration (inclusive range).
    ///
    /// Reference: Guasch (2020) §Methods "Brain FWI".
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[allow(clippy::too_many_arguments)]
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
                    message: "invert_multi_source_masked requires at least one shot".to_owned(),
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
            for (geometry, observed_data) in shots {
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
                normalized.par_mapv_inplace(|g| g / grad_max);
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
}
