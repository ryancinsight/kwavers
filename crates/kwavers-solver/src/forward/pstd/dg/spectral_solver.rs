//! Spectral solver implementation
//!
//! This module implements high-order spectral methods using FFT
//! for solving PDEs in smooth regions.

use crate::forward::lanes::for_each_z_lane;
use crate::pstd::utils::{compute_anti_aliasing_filter, compute_wavenumbers};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_math::fft::{Fft3d, Fft3dInOutExt, Shape3D};
use leto::Array3;
use std::sync::Arc;

/// Spectral solver using FFT-based methods
pub struct RegionPSTDSolver {
    order: usize,
    grid: Arc<Grid>,
    /// `−|k|²·filter` over the half spectrum `(nx, ny, nz/2+1)`; the Laplacian
    /// symbol and the anti-aliasing filter are both even in `k`.
    laplacian_symbol: Array3<f64>,
    wave_speed: f64,
    prev_field: Array3<f64>,
    has_prev_field: bool,
    fft: Fft3d,
    /// Half spectrum of the field: the transform writes it, the symbol scales
    /// it in place and the inverse consumes it.
    field_hat: Array3<Complex64>,
    laplacian: Array3<f64>,
}

impl std::fmt::Debug for RegionPSTDSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionPSTDSolver")
            .field("order", &self.order)
            .field("grid_dim", &(self.grid.nx, self.grid.ny, self.grid.nz))
            .field("grid", &self.grid)
            .field("laplacian_symbol", &self.laplacian_symbol.shape())
            .field("wave_speed", &self.wave_speed)
            .field("prev_field", &self.prev_field.shape())
            .field("has_prev_field", &self.has_prev_field)
            .field("fft", &"<fft-plan>")
            .field("field_hat", &self.field_hat.shape())
            .field("laplacian", &self.laplacian.shape())
            .finish()
    }
}

impl RegionPSTDSolver {
    /// Create a new spectral solver with default wave speed
    pub fn new(order: usize, grid: Arc<Grid>) -> Self {
        Self::with_wave_speed(order, grid, SOUND_SPEED_WATER_SIM) // Default sound speed
    }

    /// Create a new spectral solver with specified wave speed
    ///
    /// # Panics
    ///
    /// Panics when a grid dimension is zero: the FFT plan shape validates at
    /// construction, and a zero extent has no transform.
    pub fn with_wave_speed(order: usize, grid: Arc<Grid>, wave_speed: f64) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let nz_half = nz / 2 + 1;
        let (kx, ky, kz) = compute_wavenumbers(&grid);
        let filter = compute_anti_aliasing_filter(&grid, 2.0 / 3.0, order.max(1) as u32);
        // The first `nz/2+1` z bins are the non-negative wavenumbers the half
        // spectrum holds, in the same order as the full tables.
        let laplacian_symbol = Array3::from_shape_fn([nx, ny, nz_half], |[i, j, k]| {
            let (kx, ky, kz) = (kx[[i, j, k]], ky[[i, j, k]], kz[[i, j, k]]);
            -kz.mul_add(kz, kx.mul_add(kx, ky * ky)) * filter[[i, j, k]]
        });

        Self {
            order,
            grid,
            laplacian_symbol,
            wave_speed,
            prev_field: Array3::zeros((nx, ny, nz)),
            has_prev_field: false,
            fft: Fft3d::new(
                Shape3D::new(nx, ny, nz).expect("invariant: grid dimensions are non-zero"),
            ),
            field_hat: Array3::from_elem([nx, ny, nz_half], Complex64::new(0.0, 0.0)),
            laplacian: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Advance one spectral leapfrog step, writing into caller-provided `output`.
    ///
    /// ## Algorithm
    /// Verlet/leapfrog second-order time integration for the wave equation:
    /// ```text
    ///   u^{n+1} = 2u^n − u^{n−1} + (c·Δt)² · filter · ∇²u^n    (n ≥ 1)
    ///   u^{1}   = u^0 + ½(c·Δt)² · filter · ∇²u^0               (first step)
    /// ```
    /// Only cells where `mask[i,j,k]` is true are updated; others copy `field` unchanged.
    ///
    /// ## Performance
    /// Zero allocations per call when `output` is a pre-allocated caller buffer:
    /// the field transforms into the persistent half spectrum and inverts into
    /// the persistent Laplacian. `has_prev_field` selects the first-step Taylor
    /// update without storing history in an `Option<Array3<_>>`.
    ///
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] if the wave speed is not
    ///   positive, or if `field`, `mask` or `output` does not have the grid
    ///   shape the solver was built for.
    ///
    /// # Panics
    /// - Unreachable: the symbol table and the half spectrum are allocated
    ///   contiguous at construction.
    pub fn spectral_wave_step_into(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,
        mask: &Array3<bool>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        if c <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "wave_speed".to_owned(),
                value: c,
                reason: "wave speed must be positive".to_owned(),
            }));
        }
        for actual in [field.shape(), mask.shape(), output.shape()] {
            if actual != self.laplacian.shape() {
                return Err(KwaversError::Validation(
                    ValidationError::DimensionMismatch {
                        expected: format!("{:?}", self.laplacian.shape()),
                        actual: format!("{actual:?}"),
                    },
                ));
            }
        }

        self.wave_speed = c;

        self.fft.forward_r2c_into(field, &mut self.field_hat);
        let [_, ny, nz_half] = self.field_hat.shape();
        let symbol = self
            .laplacian_symbol
            .as_slice()
            .expect("invariant: the symbol table is built contiguous");
        let spectrum = self
            .field_hat
            .as_slice_mut()
            .expect("invariant: the half spectrum is allocated contiguous");
        for_each_z_lane(
            spectrum,
            [ny, nz_half],
            size_of::<Complex64>() + size_of::<f64>(),
            |start, _, _, lane| {
                for (value, &scale) in lane.iter_mut().zip(&symbol[start..start + nz_half]) {
                    *value *= scale;
                }
            },
        );
        self.fft
            .inverse_c2r_into(&mut self.field_hat, &mut self.laplacian);

        let coeff = (c * dt) * (c * dt);

        if self.has_prev_field {
            for ((((out, &use_spectral), &u), &lap), &u_prev) in output
                .iter_mut()
                .zip(mask.iter())
                .zip(field.iter())
                .zip(self.laplacian.iter())
                .zip(self.prev_field.iter())
            {
                *out = if use_spectral {
                    coeff.mul_add(lap, 2.0f64.mul_add(u, -u_prev))
                } else {
                    u
                };
            }
        } else {
            for (((out, &use_spectral), &u), &lap) in output
                .iter_mut()
                .zip(mask.iter())
                .zip(field.iter())
                .zip(self.laplacian.iter())
            {
                *out = if use_spectral {
                    (0.5 * coeff).mul_add(lap, u)
                } else {
                    u
                };
            }
        }

        self.prev_field.assign(field);
        self.has_prev_field = true;

        Ok(())
    }

    /// Convenience wrapper — allocates and returns the next field.
    /// Prefer [`Self::spectral_wave_step_into`] in time-step loops.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn spectral_wave_step(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        let mut next = Array3::zeros(field.shape());
        self.spectral_wave_step_into(field, dt, c, mask, &mut next)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::RegionPSTDSolver;
    use crate::pstd::utils::compute_anti_aliasing_filter;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::TWO_PI;
    use kwavers_grid::Grid;
    use leto::Array3;
    use std::sync::Arc;

    #[test]
    fn spectral_step_reuses_preallocated_previous_field_storage() {
        let grid = Arc::new(Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap());
        let mut solver = RegionPSTDSolver::new(4, grid);
        let prev_ptr = solver.prev_field.as_ptr();
        let field = Array3::from_shape_fn((4, 4, 4), |[i, j, k]| {
            (i as f64 + 0.25 * j as f64 - 0.5 * k as f64).sin()
        });
        let mask = Array3::from_elem([4, 4, 4], true);
        let mut output = Array3::zeros((4, 4, 4));

        solver
            .spectral_wave_step_into(&field, 1.0e-5, SOUND_SPEED_WATER_SIM, &mask, &mut output)
            .unwrap();

        assert_eq!(solver.prev_field.as_ptr(), prev_ptr);
        assert!(solver.has_prev_field);
        assert_eq!(solver.prev_field, field);

        let second_input = output.clone();
        solver
            .spectral_wave_step_into(
                &second_input,
                1.0e-5,
                SOUND_SPEED_WATER_SIM,
                &mask,
                &mut output,
            )
            .unwrap();

        assert_eq!(solver.prev_field.as_ptr(), prev_ptr);
    }

    /// **Theorem (filtered spectral Laplacian of a Fourier mode).** For
    /// `u = cos(k·x)` with DFT-representable `k` and bin `b`, the step's
    /// Laplacian is `−|k|²·F(b)·u`, `F` the anti-aliasing filter, and the first
    /// leapfrog step is `u + ½(cΔt)²·∇²u`. Even and odd `nz` cover both
    /// half-spectrum depths; the mode's bins `±b` lie on both sides of `kz = 0`.
    #[test]
    fn spectral_step_applies_the_filtered_laplacian_of_a_fourier_mode() {
        const ORDER: usize = 4;
        let (dx, dy, dz) = (1.0e-3, 2.0e-3, 1.5e-3);
        let bin = [1_usize, 2, 3];
        for [nx, ny, nz] in [[16, 12, 8], [15, 12, 9]] {
            let grid = Arc::new(Grid::new(nx, ny, nz, dx, dy, dz).unwrap());
            let filter = compute_anti_aliasing_filter(&grid, 2.0 / 3.0, ORDER as u32)[bin];
            let k = [
                TWO_PI * bin[0] as f64 / (nx as f64 * dx),
                TWO_PI * bin[1] as f64 / (ny as f64 * dy),
                TWO_PI * bin[2] as f64 / (nz as f64 * dz),
            ];
            let k_sq = k.iter().map(|k| k * k).sum::<f64>();
            let field = Array3::from_shape_fn([nx, ny, nz], |[i, j, l]| {
                (k[0] * i as f64 * dx + k[1] * j as f64 * dy + k[2] * l as f64 * dz).cos()
            });
            let mask = Array3::from_elem([nx, ny, nz], true);
            let mut output = Array3::zeros([nx, ny, nz]);
            let (dt, c) = (1.0e-7, SOUND_SPEED_WATER_SIM);
            let mut solver = RegionPSTDSolver::new(ORDER, grid);

            solver
                .spectral_wave_step_into(&field, dt, c, &mask, &mut output)
                .unwrap();

            // A forward and an inverse output each sum N rounded terms, so the
            // Laplacian carries at most 2·N·ε of the mode amplitude |k|²·F.
            let n = (nx * ny * nz) as f64;
            let bound = 2.0 * n * f64::EPSILON;
            let coeff = 0.5 * ((c * dt) * (c * dt));
            for ((&lap, &u), &next) in solver.laplacian.iter().zip(field.iter()).zip(output.iter())
            {
                let expected = -k_sq * filter * u;
                let error = (lap - expected).abs() / (k_sq * filter);
                assert!(
                    error <= bound,
                    "shape {:?}: Laplacian {lap} vs {expected}, relative error {error:e} > {bound:e}",
                    [nx, ny, nz]
                );
                assert_eq!(next, coeff.mul_add(lap, u));
            }
        }
    }
}
