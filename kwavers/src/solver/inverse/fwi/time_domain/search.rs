//! Line-search and joint-objective helpers.

use super::{geometry::FwiGeometry, FwiProcessor};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array2, Array3, Zip};
use rayon::prelude::*;

impl FwiProcessor {
    /// Compute the model objective by running a sensor-only forward simulation.
    ///
    /// Calls `forward_model_sensor_only` — no pressure-history `Array4` is
    /// allocated.  Peak memory ~55 MB per call; safe to call from `par_iter`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_objective(
        &self,
        model: &Array3<f64>,
        observed_data: &Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let synthetic_data = self.forward_model_sensor_only(model, geometry, grid)?;
        self.compute_misfit_objective(observed_data, &synthetic_data)
    }

    /// Evaluate the joint objective `J = Σᵢ Jᵢ(model)` across all shots.
    ///
    /// Shots are independent: each forward model reads `model` and `grid`
    /// immutably, so the loop runs fully in parallel via Rayon.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_joint_objective(
        &self,
        model: &Array3<f64>,
        shots: &[(FwiGeometry, Array2<f64>)],
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let results: Vec<KwaversResult<f64>> = shots
            .par_iter()
            .map(|(geometry, observed_data)| {
                self.compute_objective(model, observed_data, geometry, grid)
            })
            .collect();
        let mut total = 0.0_f64;
        for result in results {
            total += result?;
        }
        Ok(total)
    }

    /// Line search for optimal step size (single-source).
    ///
    /// Uses the Armijo sufficient-decrease condition with `c1 = 0`: any trial
    /// step that strictly reduces the objective is accepted.
    ///
    /// Reference: Nocedal & Wright (2006) §3.1, Condition (3.6a) with c₁ → 0.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn line_search(
        &self,
        model: &Array3<f64>,
        gradient: &Array3<f64>,
        observed_data: &Array2<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let mut step_size = self.parameters.step_size;
        let max_iter = 10;

        let current_objective = self.compute_objective(model, observed_data, geometry, grid)?;

        let mut test_model = Array3::<f64>::zeros(model.dim());

        for _ in 0..max_iter {
            let s = step_size;
            Zip::from(&mut test_model)
                .and(model)
                .and(gradient)
                .par_for_each(|t, &m, &g| *t = s.mul_add(-g, m));
            let test_objective =
                self.compute_objective(&test_model, observed_data, geometry, grid)?;

            if test_objective < current_objective {
                return Ok(step_size);
            }

            step_size *= 0.5;
        }

        Ok(0.0)
    }

    /// Line search for multi-source inversion.
    ///
    /// ## Algorithm
    ///
    /// First tries `c − α·g` (standard gradient descent, `g = +∂J/∂c`).  If all
    /// halvings fail, tries `c + α·g` as a fallback.  Returns a **signed** step:
    /// positive `α` → caller applies `c − α·g`; negative `α` → caller applies
    /// `c + |α|·g`.  Returns `0.0` when neither direction satisfies sufficient
    /// decrease.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn line_search_multi(
        &self,
        model: &Array3<f64>,
        gradient: &Array3<f64>,
        shots: &[(FwiGeometry, Array2<f64>)],
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let max_halvings = 5;
        let current_obj = self.compute_joint_objective(model, shots, grid)?;

        let mut test_model = Array3::<f64>::zeros(model.dim());

        let mut step = self.parameters.step_size;
        for _ in 0..max_halvings {
            Zip::from(&mut test_model)
                .and(model)
                .and(gradient)
                .par_for_each(|t, &m, &g| *t = g.mul_add(-step, m));
            let test_obj = self.compute_joint_objective(&test_model, shots, grid)?;
            if test_obj < current_obj {
                return Ok(step);
            }
            step *= 0.5;
        }

        let mut step = self.parameters.step_size;
        for _ in 0..max_halvings {
            Zip::from(&mut test_model)
                .and(model)
                .and(gradient)
                .par_for_each(|t, &m, &g| *t = g.mul_add(step, m));
            let test_obj = self.compute_joint_objective(&test_model, shots, grid)?;
            if test_obj < current_obj {
                log::info!(
                    "FWI line search: gradient sign flipped — using c += step · g \
                     (g = −∂J/∂c convention)"
                );
                return Ok(-step);
            }
            step *= 0.5;
        }

        Ok(0.0)
    }
}
