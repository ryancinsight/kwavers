//! Per-shot objective and gradient computation for multi-source FWI.

use super::super::{geometry::FwiGeometry, gradient::mute_gradient_near_sources, FwiProcessor};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
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
        // Shared forward + misfit + raw-gradient pass. For the exact self-adjoint
        // engine without a sponge this takes the memory-efficient reverse-
        // reconstruction path (no stored O(nt·N) forward history); the FDTD/PSTD
        // `Solver` engine and the damped SA engine use the stored history. Pairing
        // an SA forward with the FDTD `adjoint_model` would mix operators and
        // corrupt the gradient, so the engine dispatch lives in the shared helper.
        let (objective, mut gradient) =
            self.forward_misfit_raw_gradient(model, observed_data, geometry, grid)?;

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
