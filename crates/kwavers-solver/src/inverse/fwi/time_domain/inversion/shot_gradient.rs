//! Per-shot objective and gradient computation for multi-source FWI.

use super::super::{geometry::FwiGeometry, gradient::mute_gradient_near_sources, FwiProcessor};
use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use ndarray::{Array2, Array3};

impl FwiProcessor {
    /// Compute the per-shot objective and physics gradient for one shot gather.
    ///
    /// Returns `(Jᵢ, ∂Jᵢ/∂c)`.  Applies near-source gradient mute when
    /// `FwiParameters::source_mute_radius > 0`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_shot_gradient(
        &self,
        model: &Array3<f64>,
        geometry: &FwiGeometry,
        observed_data: &Array2<f64>,
        grid: &Grid,
    ) -> KwaversResult<(f64, Array3<f64>)> {
        let (synthetic_data, forward_history) = self.forward_model(model, geometry, grid)?;
        let objective = self.compute_misfit_objective(observed_data, &synthetic_data)?;
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
