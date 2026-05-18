use super::GenericSimdStencilProcessor;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

impl GenericSimdStencilProcessor {
    /// Update pressure field using cache-tiled stencil.
    ///
    /// # Algorithm: Cache-Tiled 3D Stencil (Kamil et al. 2010, §2.2)
    ///
    /// The loop nest is blocked with `tile_size` in each dimension so that a
    /// `tile_size³` sub-volume fits in L1/L2 cache before eviction:
    /// ```text
    /// for each (kb, jb, ib) tile origin with step tile_size:
    ///   for k in kb..min(kb+tile, nz-1):
    ///     for j in jb..min(jb+tile, ny-1):
    ///       for i in ib..min(ib+tile, nx-1):
    ///         stencil kernel
    /// ```
    ///
    /// Writes into pre-allocated `pres_scratch`; boundary values copied from
    /// the previous time-step field at boundaries.
    ///
    /// # References
    ///
    /// - Kamil, S. et al. (2010). "Auto-tuning stencil codes for cache-oblivious algorithms".
    ///   *SC '10 Companion*. §2.2.
    /// - Williams, S. et al. (2009). "Roofline: An insightful visual performance model".
    ///   *Commun. ACM* 52(4), 65–76.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn update_pressure(
        &mut self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        velocity_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if pressure.shape() != pressure_prev.shape() || pressure.shape() != velocity_div.shape() {
            return Err(KwaversError::InvalidInput(
                "Field dimensions must match".to_string(),
            ));
        }

        let dx2 = self.config.dx * self.config.dx;
        let tile = self.config.tile_size.max(1);
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        // Reset boundary to copy from previous step (boundary conditions applied after)
        self.pres_scratch.assign(pressure_prev);

        // Cache-tiled interior update
        for kb in (1..nz - 1).step_by(tile) {
            for jb in (1..ny - 1).step_by(tile) {
                for ib in (1..nx - 1).step_by(tile) {
                    let k_end = (kb + tile).min(nz - 1);
                    let j_end = (jb + tile).min(ny - 1);
                    let i_end = (ib + tile).min(nx - 1);
                    for k in kb..k_end {
                        for j in jb..j_end {
                            for i in ib..i_end {
                                let laplacian = (pressure[[i + 1, j, k]]
                                    - 2.0 * pressure[[i, j, k]]
                                    + pressure[[i - 1, j, k]])
                                    / dx2
                                    + (pressure[[i, j + 1, k]] - 2.0 * pressure[[i, j, k]]
                                        + pressure[[i, j - 1, k]])
                                        / dx2
                                    + (pressure[[i, j, k + 1]] - 2.0 * pressure[[i, j, k]]
                                        + pressure[[i, j, k - 1]])
                                        / dx2;

                                self.pres_scratch[[i, j, k]] = 2.0 * pressure[[i, j, k]]
                                    - pressure_prev[[i, j, k]]
                                    + self.pressure_coeff * laplacian
                                    + self.pressure_coeff * velocity_div[[i, j, k]];
                            }
                        }
                    }
                }
            }
        }

        let mut result = Array3::zeros((nx, ny, nz));
        result.assign(&self.pres_scratch);
        self.apply_boundary_conditions_pressure(&mut result)?;
        Ok(result)
    }
    /// Fused pressure-and-velocity update (single pass, cache-tiled).
    ///
    /// Combines both field updates in one loop pass for improved arithmetic intensity.
    /// Uses pre-allocated scratch buffers; updates `velocity` in-place via swap.
    ///
    /// # Returns
    ///
    /// Updated pressure field (velocity is updated in-place).
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
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        self.pres_scratch.assign(pressure_prev);
        self.vel_scratch.assign(velocity);

        for kb in (1..nz - 1).step_by(tile) {
            for jb in (1..ny - 1).step_by(tile) {
                for ib in (1..nx - 1).step_by(tile) {
                    let k_end = (kb + tile).min(nz - 1);
                    let j_end = (jb + tile).min(ny - 1);
                    let i_end = (ib + tile).min(nx - 1);
                    for k in kb..k_end {
                        for j in jb..j_end {
                            for i in ib..i_end {
                                let laplacian = (pressure[[i + 1, j, k]]
                                    - 2.0 * pressure[[i, j, k]]
                                    + pressure[[i - 1, j, k]])
                                    / dx2
                                    + (pressure[[i, j + 1, k]] - 2.0 * pressure[[i, j, k]]
                                        + pressure[[i, j - 1, k]])
                                        / dx2
                                    + (pressure[[i, j, k + 1]] - 2.0 * pressure[[i, j, k]]
                                        + pressure[[i, j, k - 1]])
                                        / dx2;

                                self.pres_scratch[[i, j, k]] = 2.0 * pressure[[i, j, k]]
                                    - pressure_prev[[i, j, k]]
                                    + self.pressure_coeff * laplacian
                                    + self.pressure_coeff * velocity_div[[i, j, k]];

                                let dp_dx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]])
                                    * half_dx_inv;
                                self.vel_scratch[[i, j, k]] =
                                    velocity[[i, j, k]] + self.velocity_coeff * dp_dx;
                            }
                        }
                    }
                }
            }
        }

        std::mem::swap(velocity, &mut self.vel_scratch);
        self.apply_boundary_conditions_velocity(velocity)?;

        let mut pressure_new = Array3::zeros((nx, ny, nz));
        pressure_new.assign(&self.pres_scratch);
        self.apply_boundary_conditions_pressure(&mut pressure_new)?;
        Ok(pressure_new)
    }
}

#[cfg(test)]
mod tests {
    use super::super::GenericSimdStencilConfig;
    use super::*;

    #[test]
    fn test_pressure_update() {
        let config = GenericSimdStencilConfig::default();
        let mut processor = GenericSimdStencilProcessor::new(16, 16, 16, config).unwrap();

        let pressure = Array3::ones((16, 16, 16));
        let pressure_prev = Array3::ones((16, 16, 16));
        let velocity_div = Array3::zeros((16, 16, 16));

        let result = processor.update_pressure(&pressure, &pressure_prev, &velocity_div);

        let updated = result.unwrap();
        assert_eq!(updated.shape(), pressure.shape());
    }
    #[test]
    fn test_fused_update() {
        let config = GenericSimdStencilConfig::default();
        let mut processor = GenericSimdStencilProcessor::new(16, 16, 16, config).unwrap();

        let pressure = Array3::ones((16, 16, 16));
        let pressure_prev = Array3::ones((16, 16, 16));
        let mut velocity = Array3::zeros((16, 16, 16));
        let velocity_dim = velocity.dim();
        let velocity_div = Array3::zeros((16, 16, 16));

        let result =
            processor.fused_update(&pressure, &pressure_prev, &mut velocity, &velocity_div);

        let p_new = result.unwrap();
        assert_eq!(p_new.shape(), pressure.shape());
        assert_eq!(velocity.dim(), velocity_dim);
    }

    /// Verify tiled and non-tiled (tile=256) results are bitwise identical on a 17³ grid.
    ///
    /// Non-power-of-two grid size exercises tile boundary handling.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_tiling_matches_naive() {
        let n = 17usize;
        let mut config_tiled = GenericSimdStencilConfig::default();
        config_tiled.tile_size = 8;
        let mut processor_tiled = GenericSimdStencilProcessor::new(n, n, n, config_tiled).unwrap();

        let mut config_naive = GenericSimdStencilConfig::default();
        config_naive.tile_size = 256; // effectively no tiling
        let mut processor_naive = GenericSimdStencilProcessor::new(n, n, n, config_naive).unwrap();

        let pressure = Array3::from_elem((n, n, n), 1000.0_f64);
        let pressure_prev = Array3::from_elem((n, n, n), 990.0_f64);
        let velocity_div = Array3::from_elem((n, n, n), 0.1_f64);

        let p_tiled = processor_tiled
            .update_pressure(&pressure, &pressure_prev, &velocity_div)
            .unwrap();
        let p_naive = processor_naive
            .update_pressure(&pressure, &pressure_prev, &velocity_div)
            .unwrap();

        let max_diff = p_tiled
            .iter()
            .zip(p_naive.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < f64::EPSILON * 100.0,
            "Tiled and naive pressure stencils must be identical; max diff = {max_diff:.2e}"
        );
    }
}
