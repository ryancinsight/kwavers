use super::FdtdSimdStencilProcessor;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl FdtdSimdStencilProcessor {
    /// Update pressure field using cache-tiled stencil.
    ///
    /// ## Algorithm: Cache-Tiled 3D Stencil (Kamil et al. 2010, §2.2)
    ///
    /// ```text
    /// p[i,j,k]^(n+1) = 2p^n - p^(n-1) + c²Δt²/Δx² ∇²p^n + c²Δt² (∇·u)
    /// ```
    ///
    /// Loop nest blocked with `tile_size` so that each `tile_size³` sub-volume
    /// fits in L1/L2 cache before eviction. Writes into pre-allocated `pres_scratch`
    /// then copies out; boundary values use zero-gradient Neumann conditions.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn update_pressure(
        &mut self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        velocity_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if pressure.shape() != pressure_prev.shape() || pressure.shape() != velocity_div.shape() {
            return kwavers_core::error::KwaversResult::Err(
                kwavers_core::error::KwaversError::InvalidInput(
                    "Field dimensions must match".to_owned(),
                ),
            );
        }

        let dx2 = self.config.dx * self.config.dx;
        let tile = self.config.tile_size.max(1);
        let (ni, nj, nk) = (self.nx, self.ny, self.nz);

        self.pres_scratch.assign(pressure_prev);

        for kb in (1..nk - 1).step_by(tile) {
            for jb in (1..nj - 1).step_by(tile) {
                for ib in (1..ni - 1).step_by(tile) {
                    let k_end = (kb + tile).min(nk - 1);
                    let j_end = (jb + tile).min(nj - 1);
                    let i_end = (ib + tile).min(ni - 1);
                    for k in kb..k_end {
                        for j in jb..j_end {
                            for i in ib..i_end {
                                let laplacian = (2.0f64
                                    .mul_add(-pressure[[i, j, k]], pressure[[i + 1, j, k]])
                                    + pressure[[i - 1, j, k]])
                                    / dx2
                                    + (2.0f64
                                        .mul_add(-pressure[[i, j, k]], pressure[[i, j + 1, k]])
                                        + pressure[[i, j - 1, k]])
                                        / dx2
                                    + (2.0f64
                                        .mul_add(-pressure[[i, j, k]], pressure[[i, j, k + 1]])
                                        + pressure[[i, j, k - 1]])
                                        / dx2;

                                self.pres_scratch[[i, j, k]] = self.pressure_coeff.mul_add(
                                    velocity_div[[i, j, k]],
                                    self.pressure_coeff.mul_add(
                                        laplacian,
                                        2.0f64.mul_add(
                                            pressure[[i, j, k]],
                                            -pressure_prev[[i, j, k]],
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }

        // `pres_scratch` already holds the boundary rows (copied from
        // `pressure_prev` above) and the freshly-computed interior, so cloning
        // it directly is bit-identical to allocating a zeroed array and
        // assigning, while skipping the redundant zero-fill pass.
        let mut result = self.pres_scratch.clone();
        self.apply_boundary_conditions_pressure(&mut result)?;
        Ok(result)
    }
}
