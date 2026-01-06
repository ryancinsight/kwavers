//! Spectral solver implementation
//!
//! This module implements high-order spectral methods using FFT
//! for solving PDEs in smooth regions.

use crate::error::{KwaversError, ValidationError};
use crate::fft::ProcessorFft3d;
use crate::grid::Grid;
use crate::solver::spectral::utils::{compute_anti_aliasing_filter, compute_wavenumbers};
use crate::KwaversResult;
use ndarray::{Array3, Zip};
use num_complex::Complex64;
use std::sync::Arc;

/// Spectral solver using FFT-based methods
pub struct RegionSpectralSolver {
    order: usize,
    grid: Arc<Grid>,
    k2: Array3<f64>,
    filter: Array3<f64>,
    wave_speed: f64,
    prev_field: Option<Array3<f64>>,
    fft: ProcessorFft3d,
    field_hat: Array3<Complex64>,
    lap_hat: Array3<Complex64>,
    scratch_hat: Array3<Complex64>,
    laplacian: Array3<f64>,
}

impl std::fmt::Debug for RegionSpectralSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionSpectralSolver")
            .field("order", &self.order)
            .field("grid_dim", &(self.grid.nx, self.grid.ny, self.grid.nz))
            .field("wave_speed", &self.wave_speed)
            .field("has_prev_field", &self.prev_field.is_some())
            .finish()
    }
}

impl RegionSpectralSolver {
    /// Create a new spectral solver with default wave speed
    pub fn new(order: usize, grid: Arc<Grid>) -> Self {
        Self::with_wave_speed(order, grid, 1500.0) // Default sound speed
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
                *k2 = kx * kx + ky * ky + kz * kz;
            });

        let filter = compute_anti_aliasing_filter(&grid, 2.0 / 3.0, order.max(1) as u32);

        let complex_zeros = Complex64::new(0.0, 0.0);
        let field_hat = Array3::from_elem((nx, ny, nz), complex_zeros);
        let lap_hat = Array3::from_elem((nx, ny, nz), complex_zeros);
        let scratch_hat = Array3::from_elem((nx, ny, nz), complex_zeros);
        let laplacian = Array3::zeros((nx, ny, nz));

        Self {
            order,
            grid,
            k2,
            filter,
            wave_speed,
            prev_field: None,
            fft: ProcessorFft3d::new(nx, ny, nz),
            field_hat,
            lap_hat,
            scratch_hat,
            laplacian,
        }
    }

    /// Apply spectral method for wave equation
    pub fn spectral_wave_step(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        if c <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "wave_speed".to_string(),
                value: c,
                reason: "wave speed must be positive".to_string(),
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

        let cdt = c * dt;
        let coeff = cdt * cdt;

        let mut next = field.clone();
        if let Some(prev) = self.prev_field.as_ref() {
            for ((((out, &use_spectral), &u), &lap), &u_prev) in next
                .iter_mut()
                .zip(mask.iter())
                .zip(field.iter())
                .zip(self.laplacian.iter())
                .zip(prev.iter())
            {
                if use_spectral {
                    *out = 2.0 * u - u_prev + coeff * lap;
                } else {
                    *out = u;
                }
            }
        } else {
            for (((out, &use_spectral), &u), &lap) in next
                .iter_mut()
                .zip(mask.iter())
                .zip(field.iter())
                .zip(self.laplacian.iter())
            {
                if use_spectral {
                    *out = u + 0.5 * coeff * lap;
                } else {
                    *out = u;
                }
            }
        }

        self.prev_field = Some(field.clone());
        Ok(next)
    }
}
