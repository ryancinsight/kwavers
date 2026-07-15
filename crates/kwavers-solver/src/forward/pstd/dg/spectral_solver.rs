//! Spectral solver implementation
//!
//! This module implements high-order spectral methods using FFT
//! for solving PDEs in smooth regions.

use crate::pstd::utils::{compute_anti_aliasing_filter, compute_wavenumbers};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_math::fft::{Fft3d, Fft3dInOutExt, Shape3D};
use leto::Array3 as LetoArray3;
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use std::sync::Arc;

/// Spectral solver using FFT-based methods
pub struct RegionPSTDSolver {
    order: usize,
    grid: Arc<Grid>,
    k2: LetoArray3<f64>,
    filter: LetoArray3<f64>,
    wave_speed: f64,
    prev_field: Array3<f64>,
    has_prev_field: bool,
    fft: Fft3d,
    field_hat: LetoArray3<Complex64>,
    lap_hat: LetoArray3<Complex64>,
    scratch_hat: LetoArray3<Complex64>,
    laplacian: Array3<f64>,
}

impl std::fmt::Debug for RegionPSTDSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionPSTDSolver")
            .field("order", &self.order)
            .field("grid_dim", &(self.grid.nx, self.grid.ny, self.grid.nz))
            .field("wave_speed", &self.wave_speed)
            .field("has_prev_field", &self.has_prev_field)
            .finish()
    }
}

#[inline]
fn dense_indices(index: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let plane = ny * nz;
    let i = index / plane;
    let rem = index % plane;
    let j = rem / nz;
    let k = rem % nz;
    (i, j, k)
}

fn fill_laplacian_symbol(
    k2: &mut LetoArray3<f64>,
    kx: &LetoArray3<f64>,
    ky: &LetoArray3<f64>,
    kz: &LetoArray3<f64>,
) {
    assert_eq!(
        k2.shape(),
        kx.shape(),
        "invariant: DG spectral k2 shape matches kx"
    );
    assert_eq!(
        k2.shape(),
        ky.shape(),
        "invariant: DG spectral k2 shape matches ky"
    );
    assert_eq!(
        k2.shape(),
        kz.shape(),
        "invariant: DG spectral k2 shape matches kz"
    );

    if let (Some(k2_values), Some(kx_values), Some(ky_values), Some(kz_values)) = (
        k2.as_slice_mut(),
        kx.as_slice(),
        ky.as_slice(),
        kz.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(k2_values, |index, value| {
            let kx = kx_values[index];
            let ky = ky_values[index];
            let kz = kz_values[index];
            *value = kz.mul_add(kz, kx.mul_add(kx, ky * ky));
        });
        return;
    }

    let [nx, ny, nz] = k2.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let kx = kx[[i, j, k]];
                let ky = ky[[i, j, k]];
                let kz = kz[[i, j, k]];
                k2[[i, j, k]] = kz.mul_add(kz, kx.mul_add(kx, ky * ky));
            }
        }
    }
}

fn apply_laplacian_symbol(
    lap_hat: &mut LetoArray3<Complex64>,
    field_hat: &LetoArray3<Complex64>,
    k2: &LetoArray3<f64>,
    filter: &LetoArray3<f64>,
) {
    let [nx, ny, nz] = lap_hat.shape();
    assert_eq!(
        lap_hat.shape(),
        field_hat.shape(),
        "invariant: DG spectral Laplacian spectrum shape matches field spectrum"
    );
    assert_eq!(
        lap_hat.shape(),
        k2.shape(),
        "invariant: DG spectral Laplacian spectrum shape matches k2"
    );
    assert_eq!(
        lap_hat.shape(),
        filter.shape(),
        "invariant: DG spectral Laplacian spectrum shape matches filter"
    );

    if let (Some(lap_values), Some(field_values), Some(k2_values), Some(filter_values)) = (
        lap_hat.as_slice_mut(),
        field_hat.as_slice(),
        k2.as_slice(),
        filter.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(lap_values, |index, output| {
            *output =
                field_values[index] * Complex64::new(-k2_values[index] * filter_values[index], 0.0);
        });
        return;
    }

    for index in 0..nx * ny * nz {
        let (i, j, k) = dense_indices(index, ny, nz);
        lap_hat[[i, j, k]] =
            field_hat[[i, j, k]] * Complex64::new(-k2[[i, j, k]] * filter[[i, j, k]], 0.0);
    }
}

impl RegionPSTDSolver {
    /// Create a new spectral solver with default wave speed
    pub fn new(order: usize, grid: Arc<Grid>) -> Self {
        Self::with_wave_speed(order, grid, SOUND_SPEED_WATER_SIM) // Default sound speed
    }

    /// Create a new spectral solver with specified wave speed
    pub fn with_wave_speed(order: usize, grid: Arc<Grid>, wave_speed: f64) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (kx, ky, kz) = compute_wavenumbers(&grid);

        let mut k2 = LetoArray3::zeros([nx, ny, nz]);
        fill_laplacian_symbol(&mut k2, &kx, &ky, &kz);

        let filter = compute_anti_aliasing_filter(&grid, 2.0 / 3.0, order.max(1) as u32);

        let complex_zeros = Complex64::new(0.0, 0.0);
        let field_hat = LetoArray3::from_elem([nx, ny, nz], complex_zeros);
        let lap_hat = LetoArray3::from_elem([nx, ny, nz], complex_zeros);
        let scratch_hat = LetoArray3::from_elem([nx, ny, nz], complex_zeros);
        let laplacian = Array3::zeros((nx, ny, nz));
        let prev_field = Array3::zeros((nx, ny, nz));

        Self {
            order,
            grid,
            k2,
            filter,
            wave_speed,
            prev_field,
            has_prev_field: false,
            fft: Fft3d::new(Shape3D { nx, ny, nz }),
            field_hat,
            lap_hat,
            scratch_hat,
            laplacian,
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
    /// Zero allocations per call when `output` is a pre-allocated caller buffer.
    /// `prev_field` is allocated at construction and updated via `.assign()`;
    /// `has_prev_field` selects the first-step Taylor update without storing
    /// history in an `Option<Array3<_>>`.
    ///
    /// ## Precondition
    /// `output` must have the same shape as `field`.
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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
        if field.shape() != mask.shape() {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{:?}", field.shape()),
                    actual: format!("{:?}", mask.shape()),
                },
            ));
        }
        if field.shape() != output.shape() {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{:?}", field.shape()),
                    actual: format!("{:?}", output.shape()),
                },
            ));
        }

        self.wave_speed = c;

        let field_leto = LetoArray3::from_shape_vec(
            [field.shape()[0], field.shape()[1], field.shape()[2]],
            field.iter().copied().collect(),
        )
        .expect("DG spectral field shape must match its Leto FFT shape");
        self.fft.forward_into(&field_leto, &mut self.field_hat);
        apply_laplacian_symbol(&mut self.lap_hat, &self.field_hat, &self.k2, &self.filter);
        let mut laplacian =
            LetoArray3::<f64>::zeros([field.shape()[0], field.shape()[1], field.shape()[2]]);
        self.fft
            .inverse_into(&self.lap_hat, &mut laplacian, &mut self.scratch_hat);
        for i in 0..field.shape()[0] {
            for j in 0..field.shape()[1] {
                for k in 0..field.shape()[2] {
                    self.laplacian[[i, j, k]] = laplacian[[i, j, k]];
                }
            }
        }

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
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
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
}
