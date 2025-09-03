//! K-space operations for spectral methods
//!
//! This module handles wavenumber calculations for pseudospectral methods.

use crate::grid::structure::Grid;
use ndarray::{Array1, Array3, Zip};
use std::f64::consts::PI;

/// K-space calculator for spectral methods
#[derive(Debug)]
pub struct KSpaceCalculator;

impl KSpaceCalculator {
    /// Generate k-space wavenumbers for FFT operations
    pub fn generate_kx(grid: &Grid) -> Array1<f64> {
        Self::generate_k_vector(grid.nx, grid.dx)
    }

    /// Generate k-space wavenumbers for FFT operations
    pub fn generate_ky(grid: &Grid) -> Array1<f64> {
        Self::generate_k_vector(grid.ny, grid.dy)
    }

    /// Generate k-space wavenumbers for FFT operations
    pub fn generate_kz(grid: &Grid) -> Array1<f64> {
        Self::generate_k_vector(grid.nz, grid.dz)
    }

    /// Generate k-space wavenumber vector for one dimension
    fn generate_k_vector(n: usize, dx: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let dk = 2.0 * PI / (n as f64 * dx);

        for i in 0..n {
            if i <= n / 2 {
                k[i] = i as f64 * dk;
            } else {
                k[i] = f64::from(i as i32 - n as i32) * dk;
            }
        }

        k
    }

    /// Generate 3D k-squared array for Laplacian operations
    pub fn generate_k_squared(grid: &Grid) -> Array3<f64> {
        let kx = Self::generate_kx(grid);
        let ky = Self::generate_ky(grid);
        let kz = Self::generate_kz(grid);

        let mut k_squared = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Zip::indexed(&mut k_squared).for_each(|(i, j, k), val| {
            *val = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
        });

        k_squared
    }

    /// Get k-squared array with caching
    pub fn get_k_squared_cached(grid: &Grid) -> &Array3<f64> {
        grid.k_squared_cache
            .get_or_init(|| Self::generate_k_squared(grid))
    }

    /// Calculate maximum stable k-space value
    pub fn max_k_stable(grid: &Grid) -> f64 {
        let kx_max = PI / grid.dx;
        let ky_max = PI / grid.dy;
        let kz_max = PI / grid.dz;
        (kx_max.powi(2) + ky_max.powi(2) + kz_max.powi(2)).sqrt()
    }

    /// Calculate k-space correction factor for heterogeneous media
    #[must_use]
    pub fn kspace_correction_factor(kx: f64, ky: f64, kz: f64, dx: f64, dy: f64, dz: f64) -> f64 {
        let sinc_x = if kx.abs() > 1e-10 {
            (kx * dx / 2.0).sin() / (kx * dx / 2.0)
        } else {
            1.0
        };

        let sinc_y = if ky.abs() > 1e-10 {
            (ky * dy / 2.0).sin() / (ky * dy / 2.0)
        } else {
            1.0
        };

        let sinc_z = if kz.abs() > 1e-10 {
            (kz * dz / 2.0).sin() / (kz * dz / 2.0)
        } else {
            1.0
        };

        sinc_x * sinc_y * sinc_z
    }
}
