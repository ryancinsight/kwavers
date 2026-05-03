//! Staggered-grid velocity divergence computation.

use crate::core::error::KwaversResult;
use crate::solver::geometry::Geometry;
use ndarray::{s, Zip};

use super::super::solver::FdtdSolver;

impl FdtdSolver {
    /// Compute velocity divergence on a staggered grid using backward differences.
    ///
    /// `div(v) = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z`
    ///
    /// For `CylindricalAS` geometry the cylindrical `ur/r` correction is added.
    /// CPML gradient corrections are applied per-direction when enabled.
    pub(crate) fn compute_divergence_staggered(&mut self) -> KwaversResult<()> {
        self.staggered_operator
            .apply_backward_x_into(self.fields.ux.view(), &mut self.dvx_scratch)?;
        self.staggered_operator
            .apply_backward_y_into(self.fields.uy.view(), &mut self.dvy_scratch)?;
        self.staggered_operator
            .apply_backward_z_into(self.fields.uz.view(), &mut self.divergence_scratch)?;

        if self.config.geometry == Geometry::CylindricalAS {
            let dz = self.grid.dz;
            let (nx, _ny, nz) = self.divergence_scratch.dim();
            for i in 0..nx {
                self.divergence_scratch[[i, 0, 0]] += self.fields.uz[[i, 0, 0]] / (0.5 * dz);
            }
            for k in 1..nz {
                let r_center = k as f64 * dz;
                let uz_k_vals: Vec<f64> =
                    self.fields.uz.slice(s![.., 0, k]).iter().copied().collect();
                let uz_km1_vals: Vec<f64> = self
                    .fields
                    .uz
                    .slice(s![.., 0, k - 1])
                    .iter()
                    .copied()
                    .collect();
                let mut dvz_k = self.divergence_scratch.slice_mut(s![.., 0, k]);
                Zip::from(&mut dvz_k)
                    .and(ndarray::ArrayView1::from(&uz_k_vals))
                    .and(ndarray::ArrayView1::from(&uz_km1_vals))
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

        Zip::from(&mut self.divergence_scratch)
            .and(&self.dvx_scratch)
            .and(&self.dvy_scratch)
            .par_for_each(|d, &dx, &dy| *d += dx + dy);
        Ok(())
    }
}
