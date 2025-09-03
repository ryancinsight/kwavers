//! Spectral solver implementation
//!
//! This module implements high-order spectral methods using FFT
//! for solving PDEs in smooth regions.

use super::traits::{NumericalSolver, SpectralOperations};
use crate::error::{KwaversError, ValidationError};
use crate::grid::Grid;
use crate::utils::ifft_3d;
use crate::KwaversResult;
use ndarray::{s, Array3, Zip};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::sync::Arc;

/// Spectral solver using FFT-based methods
#[derive(Debug)]
pub struct SpectralSolver {
    order: usize,
    grid: Arc<Grid>,
    /// Wavenumber arrays for spectral derivatives
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,
    /// Filter for de-aliasing (2/3 rule)
    filter: Array3<f64>,
    /// Wave speed for the solver
    wave_speed: f64,
}

impl SpectralSolver {
    /// Create a new spectral solver with default wave speed
    pub fn new(order: usize, grid: Arc<Grid>) -> Self {
        Self::with_wave_speed(order, grid, crate::physics::constants::SOUND_SPEED_WATER)
    }

    /// Create a new spectral solver with specified wave speed
    pub fn with_wave_speed(order: usize, grid: Arc<Grid>, wave_speed: f64) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Initialize wavenumber arrays
        let mut kx = Array3::zeros((nx, ny, nz));
        let mut ky = Array3::zeros((nx, ny, nz));
        let mut kz = Array3::zeros((nx, ny, nz));

        // Compute wavenumbers
        let kx_1d = Self::compute_wavenumbers(nx, grid.dx);
        let ky_1d = Self::compute_wavenumbers(ny, grid.dy);
        let kz_1d = Self::compute_wavenumbers(nz, grid.dz);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    kx[[i, j, k]] = kx_1d[i];
                    ky[[i, j, k]] = ky_1d[j];
                    kz[[i, j, k]] = kz_1d[k];
                }
            }
        }

        // Create de-aliasing filter (2/3 rule)
        let mut filter = Array3::from_elem((nx, ny, nz), 1.0);
        let kx_max = 2.0 * PI / grid.dx * (nx as f64 / 3.0);
        let ky_max = 2.0 * PI / grid.dy * (ny as f64 / 3.0);
        let kz_max = 2.0 * PI / grid.dz * (nz as f64 / 3.0);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if kx[[i, j, k]].abs() > kx_max
                        || ky[[i, j, k]].abs() > ky_max
                        || kz[[i, j, k]].abs() > kz_max
                    {
                        filter[[i, j, k]] = 0.0;
                    }
                }
            }
        }

        Self {
            order,
            grid,
            kx,
            ky,
            kz,
            filter,
            wave_speed,
        }
    }

    /// Perform 3D FFT on a complex field
    fn perform_fft_3d(&self, field: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut field_hat = field.clone();
        let mut planner = FftPlanner::<f64>::new();

        // FFT along x
        for j in 0..ny {
            for k in 0..nz {
                let fft = planner.plan_fft_forward(nx);
                let mut slice: Vec<Complex<f64>> = field_hat.slice(s![.., j, k]).to_vec();
                fft.process(&mut slice);
                for (i, val) in slice.into_iter().enumerate() {
                    field_hat[[i, j, k]] = val;
                }
            }
        }

        // FFT along y
        for i in 0..nx {
            for k in 0..nz {
                let fft = planner.plan_fft_forward(ny);
                let mut slice: Vec<Complex<f64>> = field_hat.slice(s![i, .., k]).to_vec();
                fft.process(&mut slice);
                for (j, val) in slice.into_iter().enumerate() {
                    field_hat[[i, j, k]] = val;
                }
            }
        }

        // FFT along z
        for i in 0..nx {
            for j in 0..ny {
                let fft = planner.plan_fft_forward(nz);
                let mut slice: Vec<Complex<f64>> = field_hat.slice(s![i, j, ..]).to_vec();
                fft.process(&mut slice);
                for (k, val) in slice.into_iter().enumerate() {
                    field_hat[[i, j, k]] = val;
                }
            }
        }

        field_hat
    }

    /// Compute wavenumbers for FFT
    fn compute_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
        let mut k = vec![0.0; n];
        let l = n as f64 * dx; // Total domain size

        for i in 0..n {
            if i <= n / 2 {
                k[i] = 2.0 * PI * i as f64 / l;
            } else {
                k[i] = 2.0 * PI * (i as f64 - n as f64) / l;
            }
        }

        k
    }

    /// Apply spectral method for wave equation
    fn spectral_wave_step(
        &self,
        field: &Array3<f64>,
        dt: f64,
        c: f64, // wave speed
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        // Convert to spectral space
        let field_complex = field.mapv(|x| Complex::new(x, 0.0));
        let mut field_hat = self.perform_fft_3d(&field_complex);

        // Apply de-aliasing filter
        Zip::from(&mut field_hat)
            .and(&self.filter)
            .for_each(|f, &filter| *f *= filter);

        // Compute Laplacian in spectral space
        let k2 = &self.kx * &self.kx + &self.ky * &self.ky + &self.kz * &self.kz;

        // Time evolution in spectral space (assuming second-order wave equation)
        // For stability, use exact solution in spectral space
        let omega = k2.mapv(|k2_val| (c * c * k2_val).sqrt());

        Zip::from(&mut field_hat).and(&omega).for_each(|f, &w| {
            let phase = Complex::new((w * dt).cos(), -(w * dt).sin());
            *f *= phase;
        });

        // Transform back to physical space
        let result_complex = ifft_3d(&field_hat, &self.grid);
        let mut result = result_complex;

        // Apply mask to use spectral solution only in smooth regions
        Zip::from(&mut result)
            .and(field)
            .and(mask)
            .for_each(|r, &f, &m| {
                if m {
                    // In discontinuous regions, don't use spectral result
                    *r = f; // Keep original value, DG solver will handle this
                }
            });

        Ok(result)
    }
}

impl NumericalSolver for SpectralSolver {
    fn solve(
        &self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        // Use configured wave speed from the solver
        self.spectral_wave_step(field, dt, self.wave_speed, mask)
    }

    fn max_stable_dt(&self, grid: &Grid) -> f64 {
        // CFL condition for spectral methods
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let k_max = PI / dx_min;

        // Spectral methods have stricter stability requirements
        0.5 * dx_min / (self.wave_speed * self.order as f64)
    }

    fn update_order(&mut self, order: usize) {
        self.order = order;
        // Note: In a full implementation, we might need to update filter based on order
    }
}

impl SpectralOperations for SpectralSolver {
    fn spectral_derivative(
        &self,
        field: &Array3<f64>,
        direction: usize,
    ) -> KwaversResult<Array3<f64>> {
        // Convert to spectral space
        let field_complex = field.mapv(|x| Complex::new(x, 0.0));
        let mut field_hat = self.perform_fft_3d(&field_complex);

        // Apply derivative in spectral space
        match direction {
            0 => {
                // x-derivative
                Zip::from(&mut field_hat)
                    .and(&self.kx)
                    .for_each(|f, &k| *f *= Complex::new(0.0, k));
            }
            1 => {
                // y-derivative
                Zip::from(&mut field_hat)
                    .and(&self.ky)
                    .for_each(|f, &k| *f *= Complex::new(0.0, k));
            }
            2 => {
                // z-derivative
                Zip::from(&mut field_hat)
                    .and(&self.kz)
                    .for_each(|f, &k| *f *= Complex::new(0.0, k));
            }
            _ => {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "direction".to_string(),
                    value: direction.to_string(),
                    constraint: "Must be 0, 1, or 2".to_string(),
                }))
            }
        }

        // Apply filter
        Zip::from(&mut field_hat)
            .and(&self.filter)
            .for_each(|f, &filter| *f *= filter);

        // Transform back
        let result_complex = ifft_3d(&field_hat, &self.grid);
        Ok(result_complex)
    }

    fn apply_filter(&self, field: &mut Array3<f64>) {
        // Apply spectral filter for de-aliasing
        let field_complex = field.mapv(|x| Complex::new(x, 0.0));
        let mut field_hat = self.perform_fft_3d(&field_complex);
        Zip::from(&mut field_hat)
            .and(&self.filter)
            .for_each(|f, &filter| *f *= filter);

        let filtered_complex = ifft_3d(&field_hat, &self.grid);
        field.assign(&filtered_complex);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_solver_creation() {
        let grid = Arc::new(Grid::new(32, 32, 32, 1.0, 1.0, 1.0).unwrap());
        let solver = SpectralSolver::new(8, grid);
        assert_eq!(solver.order, 8);
    }

    #[test]
    fn test_wavenumber_computation() {
        let k = SpectralSolver::compute_wavenumbers(8, 1.0);
        assert_eq!(k.len(), 8);
        assert!((k[0]).abs() < 1e-10); // k=0 at origin
        assert!((k[1] - 2.0 * PI / 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_derivative() {
        // Test that the spectral solver can compute derivatives
        // We'll verify basic functionality without checking exact values
        let n = 16;
        let grid = Arc::new(Grid::new(n, n, n, 1.0, 1.0, 1.0).unwrap());
        let solver = SpectralSolver::new(8, grid.clone());

        // Test 1: Derivative of a constant field should be zero
        let const_field = Array3::from_elem((n, n, n), 1.0);
        let deriv_const = solver.spectral_derivative(&const_field, 0).unwrap();
        let max_deriv = deriv_const.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(
            max_deriv < 1e-10,
            "Derivative of constant should be near zero, got max: {}",
            max_deriv
        );

        // Test 2: Verify that derivative computation doesn't panic for various fields
        let fields = vec![
            Array3::zeros((n, n, n)),          // Zero field
            Array3::from_elem((n, n, n), 1.0), // Constant field
            Array3::from_elem((n, n, n), 5.0), // Another constant
        ];

        for field in fields {
            // Should not panic
            let _ = solver.spectral_derivative(&field, 0).unwrap();
            let _ = solver.spectral_derivative(&field, 1).unwrap();
            let _ = solver.spectral_derivative(&field, 2).unwrap();
        }

        // Test 3: Verify that the derivative has the same shape as input
        let test_field = Array3::from_elem((n, n, n), 1.0);
        let deriv = solver.spectral_derivative(&test_field, 0).unwrap();
        assert_eq!(
            deriv.dim(),
            test_field.dim(),
            "Derivative should have same dimensions as input"
        );
    }
}
