use super::SimdStencilProcessor;
use crate::core::error::KwaversResult;
use ndarray::Array3;

impl SimdStencilProcessor {
    /// Fused pressure-and-velocity update (single pass, cache-tiled).
    ///
    /// Combines both field updates in one loop pass for improved arithmetic intensity.
    /// Uses pre-allocated scratch buffers; updates `velocity` in-place via swap.
    ///
    /// ## Returns
    ///
    /// Updated pressure field; `velocity` is updated in-place.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn fused_update(
        &mut self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        velocity: &mut Array3<f64>,
        velocity_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let dx2 = self.config.dx * self.config.dx;
        let half_dx_inv = 1.0 / (2.0 * self.config.dx);
        let tile = self.config.tile_size.max(1);
        let (ni, nj, nk) = (self.nx, self.ny, self.nz);

        self.pres_scratch.assign(pressure_prev);
        self.vel_scratch.assign(velocity);

        for kb in (1..nk - 1).step_by(tile) {
            for jb in (1..nj - 1).step_by(tile) {
                for ib in (1..ni - 1).step_by(tile) {
                    let k_end = (kb + tile).min(nk - 1);
                    let j_end = (jb + tile).min(nj - 1);
                    let i_end = (ib + tile).min(ni - 1);
                    for k in kb..k_end {
                        for j in jb..j_end {
                            for i in ib..i_end {
                                let laplacian = (2.0f64.mul_add(-pressure[[i, j, k]], pressure[[i + 1, j, k]])
                                    + pressure[[i - 1, j, k]])
                                    / dx2
                                    + (2.0f64.mul_add(-pressure[[i, j, k]], pressure[[i, j + 1, k]])
                                        + pressure[[i, j - 1, k]])
                                        / dx2
                                    + (2.0f64.mul_add(-pressure[[i, j, k]], pressure[[i, j, k + 1]])
                                        + pressure[[i, j, k - 1]])
                                        / dx2;

                                self.pres_scratch[[i, j, k]] = self.pressure_coeff.mul_add(velocity_div[[i, j, k]], self.pressure_coeff.mul_add(laplacian, 2.0f64.mul_add(pressure[[i, j, k]], -pressure_prev[[i, j, k]])));

                                let dp_dx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]])
                                    * half_dx_inv;
                                self.vel_scratch[[i, j, k]] =
                                    self.velocity_coeff.mul_add(dp_dx, velocity[[i, j, k]]);
                            }
                        }
                    }
                }
            }
        }

        std::mem::swap(velocity, &mut self.vel_scratch);
        self.apply_boundary_conditions_velocity(velocity)?;

        let mut pressure_new = Array3::zeros((ni, nj, nk));
        pressure_new.assign(&self.pres_scratch);
        self.apply_boundary_conditions_pressure(&mut pressure_new)?;
        Ok(pressure_new)
    }
}
