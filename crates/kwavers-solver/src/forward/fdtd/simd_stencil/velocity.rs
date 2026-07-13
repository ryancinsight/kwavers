use super::FdtdSimdStencilProcessor;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl FdtdSimdStencilProcessor {
    /// Update velocity field using in-place scratch buffer (no per-step allocation).
    ///
    /// ```text
    /// u[i,j,k]^(n+1) = u^n − (Δt/ρ) ∂p/∂x
    /// ```
    ///
    /// Writes into `self.vel_scratch`, then swaps heap pointers with `velocity`
    /// via `std::mem::swap` — zero copies, zero allocation.
    /// Loop is cache-tiled identically to `update_pressure` (Kamil et al. 2010).
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn update_velocity(
        &mut self,
        velocity: &mut Array3<f64>,
        pressure: &Array3<f64>,
    ) -> KwaversResult<()> {
        if velocity.shape() != pressure.shape() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Field dimensions must match".to_owned(),
            ));
        }

        let half_dx_inv = 1.0 / (2.0 * self.config.dx);
        let tile = self.config.tile_size.max(1);
        let (ni, nj, nk) = (self.nx, self.ny, self.nz);

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
        Ok(())
    }
}
