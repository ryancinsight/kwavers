//! Staggered-grid velocity divergence computation.

use crate::geometry::SolverGeometry;
use kwavers_core::error::KwaversResult;
use ndarray::{s, Zip};

use super::super::solver::FdtdSolver;

impl FdtdSolver {
    /// Compute velocity divergence on a staggered grid using backward differences.
    ///
    /// `div(v) = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z`
    ///
    /// For `CylindricalAS` geometry the cylindrical `ur/r` correction is added.
    /// CPML gradient corrections are applied per-direction when enabled.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn compute_divergence_staggered(&mut self) -> KwaversResult<()> {
        self.staggered_operator
            .apply_backward_x_into(self.fields.ux.view(), &mut self.dvx_scratch)?;
        self.staggered_operator
            .apply_backward_y_into(self.fields.uy.view(), &mut self.dvy_scratch)?;
        self.staggered_operator
            .apply_backward_z_into(self.fields.uz.view(), &mut self.divergence_scratch)?;

        if self.config.geometry == SolverGeometry::CylindricalAS {
            let dz = self.grid.dz;
            let (nx, _ny, nz) = self.divergence_scratch.dim();
            for i in 0..nx {
                self.divergence_scratch[[i, 0, 0]] += self.fields.uz[[i, 0, 0]] / (0.5 * dz);
            }
            for k in 1..nz {
                let r_center = k as f64 * dz;
                // Borrow disjoint slices from `fields.uz` and `divergence_scratch` without
                // intermediate Vec allocation; split_at avoids overlapping borrows.
                let uz_k = self.fields.uz.slice(s![.., 0, k]);
                let uz_km1 = self.fields.uz.slice(s![.., 0, k - 1]);
                let mut dvz_k = self.divergence_scratch.slice_mut(s![.., 0, k]);
                Zip::from(&mut dvz_k)
                    .and(&uz_k)
                    .and(&uz_km1)
                    .for_each(|d, &uk, &ukm1| {
                        *d += (uk + ukm1) / (2.0 * r_center);
                    });
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
