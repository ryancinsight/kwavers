use super::HybridSolver;
use crate::core::error::KwaversResult;
use crate::solver::forward::hybrid::domain_decomposition::{DomainRegion, DomainType};
use ndarray::s;

impl HybridSolver {
    /// Perform a single time step
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        self.pstd_solver.fields.p.assign(&self.fields.p);
        self.pstd_solver.fields.ux.assign(&self.fields.ux);
        self.pstd_solver.fields.uy.assign(&self.fields.uy);
        self.pstd_solver.fields.uz.assign(&self.fields.uz);

        self.fdtd_solver.fields.p.assign(&self.fields.p);
        self.fdtd_solver.fields.ux.assign(&self.fields.ux);
        self.fdtd_solver.fields.uy.assign(&self.fields.uy);
        self.fdtd_solver.fields.uz.assign(&self.fields.uz);

        self.pstd_solver.step_forward()?;
        self.fdtd_solver.step_forward()?;

        let regions = self.regions.clone();
        for region in &regions {
            match region.domain_type {
                DomainType::PSTD => {
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    self.fields
                        .p
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.p.slice(slice));
                    self.fields
                        .ux
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.ux.slice(slice));
                    self.fields
                        .uy
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uy.slice(slice));
                    self.fields
                        .uz
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uz.slice(slice));
                }
                DomainType::FDTD => {
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    self.fields
                        .p
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.p.slice(slice));
                    self.fields
                        .ux
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.ux.slice(slice));
                    self.fields
                        .uy
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uy.slice(slice));
                    self.fields
                        .uz
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uz.slice(slice));
                }
                DomainType::Hybrid => {
                    self.apply_hybrid_region_blended_internal(region)?;
                }
            }
        }

        self.time_step += 1;
        Ok(())
    }
    /// Apply hybrid region blended internal.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_hybrid_region_blended_internal(
        &mut self,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        let nx = region.end.0 - region.start.0;
        let ny = region.end.1 - region.start.1;
        let nz = region.end.2 - region.start.2;
        const BLEND_WIDTH: usize = 5;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dist_from_boundary = ((i.min(nx - i - 1))
                        .min(j.min(ny - j - 1))
                        .min(k.min(nz - k - 1)))
                        as f64;
                    let weight = if dist_from_boundary < BLEND_WIDTH as f64 {
                        0.5 * (1.0
                            + (std::f64::consts::PI * dist_from_boundary / BLEND_WIDTH as f64)
                                .cos())
                    } else {
                        1.0
                    };
                    let gi = region.start.0 + i;
                    let gj = region.start.1 + j;
                    let gk = region.start.2 + k;

                    self.fields.p[[gi, gj, gk]] = weight * self.pstd_solver.fields.p[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.p[[gi, gj, gk]];
                    self.fields.ux[[gi, gj, gk]] = weight
                        * self.pstd_solver.fields.ux[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.ux[[gi, gj, gk]];
                    self.fields.uy[[gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uy[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uy[[gi, gj, gk]];
                    self.fields.uz[[gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uz[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uz[[gi, gj, gk]];
                }
            }
        }
        Ok(())
    }
}
