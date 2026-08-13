//! Staggered-grid velocity divergence computation.

use crate::geometry::SolverGeometry;
use kwavers_core::error::KwaversResult;

use super::super::solver::FdtdSolver;

impl FdtdSolver {
    /// Compute velocity divergence on a staggered grid.
    ///
    /// `div(v) = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z`
    ///
    /// Uses the **Yee divergence**, the negative adjoint of the forward
    /// difference the velocity update applies, rather than a general backward
    /// difference. The two differ only at the low face — zero flux there versus
    /// a one-sided difference — but that row is what makes the leapfrog
    /// symplectic. With the one-sided closure a lossless standing wave grew its
    /// energy by nearly five orders of magnitude over two thousand steps
    /// (KW-SOL-081); with the adjoint closure it is bounded.
    ///
    /// For `CylindricalAS` geometry the cylindrical `ur/r` correction is added.
    /// CPML gradient corrections are applied per-direction when enabled.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(crate) fn compute_divergence_staggered(&mut self) -> KwaversResult<()> {
        self.staggered_operator
            .apply_divergence_x_into(self.fields.ux.view(), &mut self.dvx_scratch)?;
        self.staggered_operator
            .apply_divergence_y_into(self.fields.uy.view(), &mut self.dvy_scratch)?;
        self.staggered_operator
            .apply_divergence_z_into(self.fields.uz.view(), &mut self.divergence_scratch)?;

        if self.config.geometry == SolverGeometry::CylindricalAS {
            let dz = self.grid.dz;
            let [nx, _ny, nz] = self.divergence_scratch.shape();
            let uz_view = self.fields.uz.view();
            for i in 0..nx {
                self.divergence_scratch[[i, 0, 0]] += uz_view[[i, 0, 0]] / (0.5 * dz);
            }
            for k in 1..nz {
                let r_center = k as f64 * dz;
                // Borrow disjoint slices from `fields.uz` and `divergence_scratch` without
                // intermediate Vec allocation; split_at avoids overlapping borrows.
                let uz_k = uz_view
                    .slice(&[(0, nx, 1), (0, 1, 1), (k, k + 1, 1)])
                    .unwrap();
                let uz_km1 = uz_view
                    .slice(&[(0, nx, 1), (0, 1, 1), (k - 1, k, 1)])
                    .unwrap();
                let mut dvz_k = self
                    .divergence_scratch
                    .slice_mut(&[(0, nx, 1), (0, 1, 1), (k, k + 1, 1)])
                    .unwrap();
                leto_ops::zip_mut_with(&mut dvz_k, (&uz_k, &uz_km1), |d, (uk, ukm1)| {
                    *d += (*uk + *ukm1) / (2.0 * r_center);
                })
                .expect("invariant: cylindrical divergence slice shapes match");
            }
        }

        if let Some(ref mut cpml) = self.cpml_boundary {
            cpml.update_and_apply_v_gradient_correction(&mut self.dvx_scratch, 0);
            cpml.update_and_apply_v_gradient_correction(&mut self.dvy_scratch, 1);
            cpml.update_and_apply_v_gradient_correction(&mut self.divergence_scratch, 2);
        }

        super::accumulate_two_fields(
            &mut self.divergence_scratch,
            &self.dvx_scratch,
            &self.dvy_scratch,
        );
        Ok(())
    }
}
