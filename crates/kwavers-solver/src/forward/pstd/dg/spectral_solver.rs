//! Spectral solver implementation
//!
//! This module implements high-order spectral methods using FFT
//! for solving PDEs in smooth regions.

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_domain::grid::Grid;
use kwavers_math::fft::{Fft3d, Fft3dInOutExt, Shape3D};
use crate::pstd::utils::{compute_anti_aliasing_filter, compute_wavenumbers};
use ndarray::{Array3, Zip};
use num_complex::Complex64;
use std::sync::Arc;

/// Spectral solver using FFT-based methods
pub struct RegionPSTDSolver {
    order: usize,
    grid: Arc<Grid>,
    k2: Array3<f64>,
    filter: Array3<f64>,
    wave_speed: f64,
    prev_field: Array3<f64>,
    has_prev_field: bool,
    fft: Fft3d,
    field_hat: Array3<Complex64>,
    lap_hat: Array3<Complex64>,
    scratch_hat: Array3<Complex64>,
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

impl RegionPSTDSolver {
    /// Create a new spectral solver with default wave speed
    pub fn new(order: usize, grid: Arc<Grid>) -> Self {
        Self::with_wave_speed(order, grid, SOUND_SPEED_WATER_SIM) // Default sound speed
    }

    /// Create a new spectral solver with specified wave speed
    pub fn with_wave_speed(order: usize, grid: Arc<Grid>, wave_speed: f64) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (kx, ky, kz) = compute_wavenumbers(&grid);

        let mut k2 = Array3::zeros((nx, ny, nz));
        Zip::from(&mut k2)
            .and(&kx)
            .and(&ky)
            .and(&kz)
            .for_each(|k2, &kx, &ky, &kz| {
                *k2 = kz.mul_add(kz, kx.mul_add(kx, ky * ky));
            });

        let filter = compute_anti_aliasing_filter(&grid, 2.0 / 3.0, order.max(1) as u32);

        let complex_zeros = Complex64::new(0.0, 0.0);
        let field_hat = Array3::from_elem((nx, ny, nz), complex_zeros);
        let lap_hat = Array3::from_elem((nx, ny, nz), complex_zeros);
        let scratch_hat = Array3::from_elem((nx, ny, nz), complex_zeros);
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
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
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
        if field.dim() != mask.dim() {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{:?}", field.dim()),
                    actual: format!("{:?}", mask.dim()),
                },
            ));
        }
        if field.dim() != output.dim() {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{:?}", field.dim()),
                    actual: format!("{:?}", output.dim()),
                },
            ));
        }

        self.wave_speed = c;

        self.fft.forward_into(field, &mut self.field_hat);
        Zip::from(&mut self.lap_hat)
            .and(&self.field_hat)
            .and(&self.k2)
            .and(&self.filter)
            .for_each(|out, &u_hat, &k2, &f| {
                *out = u_hat * Complex64::new(-k2 * f, 0.0);
            });
        self.fft
            .inverse_into(&self.lap_hat, &mut self.laplacian, &mut self.scratch_hat);

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
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn spectral_wave_step(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        let mut next = Array3::zeros(field.dim());
        self.spectral_wave_step_into(field, dt, c, mask, &mut next)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::RegionPSTDSolver;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_domain::grid::Grid;
    use ndarray::Array3;
    use std::sync::Arc;

    #[test]
    fn spectral_step_reuses_preallocated_previous_field_storage() {
        let grid = Arc::new(Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap());
        let mut solver = RegionPSTDSolver::new(4, grid);
        let prev_ptr = solver.prev_field.as_ptr();
        let field = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
            (i as f64 + 0.25 * j as f64 - 0.5 * k as f64).sin()
        });
        let mask = Array3::from_elem((4, 4, 4), true);
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
