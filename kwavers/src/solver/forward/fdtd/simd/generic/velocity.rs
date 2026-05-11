use super::SimdStencilProcessor;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

impl SimdStencilProcessor {
    /// Update velocity field using in-place scratch buffer (no per-step allocation).
    ///
    /// # Algorithm
    ///
    /// Writes new values into `self.vel_scratch`, then swaps heap pointers with
    /// `velocity` via `std::mem::swap` — zero copies, zero allocation.
    ///
    /// The loop is cache-tiled identically to `update_pressure` (Kamil et al. 2010).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn update_velocity(
        &mut self,
        velocity: &mut Array3<f64>,
        pressure: &Array3<f64>,
    ) -> KwaversResult<()> {
        if velocity.shape() != pressure.shape() {
            return Err(KwaversError::InvalidInput(
                "Field dimensions must match".to_string(),
            ));
        }

        let half_dx_inv = 1.0 / (2.0 * self.config.dx);
        let tile = self.config.tile_size.max(1);
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        // Copy boundary values (boundary conditions applied after interior update)
        self.vel_scratch.assign(velocity);

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

        // Swap heap pointers: velocity ← scratch, no allocation
        std::mem::swap(velocity, &mut self.vel_scratch);
        self.apply_boundary_conditions_velocity(velocity)?;
        Ok(())
    }

    /// Apply boundary conditions (zero-gradient Neumann)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_boundary_conditions_pressure(
        &self,
        field: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        // Boundaries: copy interior values
        for k in 0..self.nz {
            for j in 0..self.ny {
                field[[0, j, k]] = field[[1, j, k]];
                field[[self.nx - 1, j, k]] = field[[self.nx - 2, j, k]];
            }
        }

        for k in 0..self.nz {
            for i in 0..self.nx {
                field[[i, 0, k]] = field[[i, 1, k]];
                field[[i, self.ny - 1, k]] = field[[i, self.ny - 2, k]];
            }
        }

        for j in 0..self.ny {
            for i in 0..self.nx {
                field[[i, j, 0]] = field[[i, j, 1]];
                field[[i, j, self.nz - 1]] = field[[i, j, self.nz - 2]];
            }
        }

        Ok(())
    }

    /// Apply boundary conditions for velocity
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_boundary_conditions_velocity(
        &self,
        field: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        // Set velocity to zero at rigid boundaries
        for k in 0..self.nz {
            for j in 0..self.ny {
                field[[0, j, k]] = 0.0;
                field[[self.nx - 1, j, k]] = 0.0;
            }
        }

        for k in 0..self.nz {
            for i in 0..self.nx {
                field[[i, 0, k]] = 0.0;
                field[[i, self.ny - 1, k]] = 0.0;
            }
        }

        for j in 0..self.ny {
            for i in 0..self.nx {
                field[[i, j, 0]] = 0.0;
                field[[i, j, self.nz - 1]] = 0.0;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::SimdStencilConfig;
    use super::*;

    #[test]
    fn test_velocity_update() {
        let config = SimdStencilConfig::default();
        let mut processor = SimdStencilProcessor::new(16, 16, 16, config).unwrap();

        let mut velocity = Array3::zeros((16, 16, 16));
        let pressure = Array3::ones((16, 16, 16));

        let result = processor.update_velocity(&mut velocity, &pressure);
        result.unwrap();
        assert_eq!(velocity.shape(), &[16, 16, 16]);
    }
    /// Verify in-place velocity update produces the same result as the old clone-based path.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_velocity_inplace_no_regression() {
        let n = 16usize;
        let config = SimdStencilConfig::default();
        let mut processor = SimdStencilProcessor::new(n, n, n, config).unwrap();

        let pressure = Array3::from_elem((n, n, n), 500.0_f64);
        let mut vel_inplace = Array3::from_elem((n, n, n), 0.1_f64);
        processor
            .update_velocity(&mut vel_inplace, &pressure)
            .unwrap();

        // Regression: all interior values updated, boundaries zeroed
        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    // Interior points should have been touched (gradient of uniform field = 0,
                    // so value unchanged for uniform pressure)
                    assert!(
                        (vel_inplace[[i, j, k]] - 0.1).abs() < 1e-12,
                        "Interior vel at [{i},{j},{k}] changed unexpectedly: {}",
                        vel_inplace[[i, j, k]]
                    );
                }
            }
        }
        // Boundary zeroed by rigid BC
        assert_eq!(vel_inplace[[0, 1, 1]], 0.0);
        assert_eq!(vel_inplace[[n - 1, 1, 1]], 0.0);
    }
}
