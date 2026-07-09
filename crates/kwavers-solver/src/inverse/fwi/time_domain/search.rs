//! Line-search and joint-objective helpers.

use super::{geometry::FwiGeometry, FwiProcessor};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, map_collect_with, Adaptive};
use leto::{
    Array2,
    Array3,
};

const TRIAL_MODEL_CHUNK_LEN: usize = 4096;

fn write_trial_model(
    test_model: &mut Array3<f64>,
    model: &Array3<f64>,
    gradient: &Array3<f64>,
    gradient_scale: f64,
) {
    debug_assert_eq!(test_model.dim(), model.dim());
    debug_assert_eq!(test_model.dim(), gradient.dim());
    let (_, ny, nz) = model.dim();
    let values = test_model
        .as_slice_memory_order_mut()
        .expect("invariant: Array3::zeros trial model has contiguous storage");
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        values,
        TRIAL_MODEL_CHUNK_LEN,
        |chunk_index, chunk| {
            let base = chunk_index * TRIAL_MODEL_CHUNK_LEN;
            for (offset, target) in chunk.iter_mut().enumerate() {
                let linear = base + offset;
                let k = linear % nz;
                let j = (linear / nz) % ny;
                let i = linear / (ny * nz);
                *target = gradient[[i, j, k]].mul_add(gradient_scale, model[[i, j, k]]);
            }
        },
    );
}

impl FwiProcessor {
    /// Compute the model objective by running a sensor-only forward simulation.
    ///
    /// Calls `forward_model_sensor_only` — no pressure-history `Array4` is
    /// allocated.  Peak memory ~55 MB per call; safe to call from the Atlas
    /// execution provider.
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
    /// immutably, so the loop runs through the Atlas execution provider.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_joint_objective(
        &self,
        model: &Array3<f64>,
        shots: &[(FwiGeometry, Array2<f64>)],
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let results: Vec<KwaversResult<f64>> =
            map_collect_with::<Adaptive, _, _, _>(shots, |(geometry, observed_data)| {
                self.compute_objective(model, observed_data, geometry, grid)
            });
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
            write_trial_model(&mut test_model, model, gradient, -s);
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
            write_trial_model(&mut test_model, model, gradient, -step);
            let test_obj = self.compute_joint_objective(&test_model, shots, grid)?;
            if test_obj < current_obj {
                return Ok(step);
            }
            step *= 0.5;
        }

        let mut step = self.parameters.step_size;
        for _ in 0..max_halvings {
            write_trial_model(&mut test_model, model, gradient, step);
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
