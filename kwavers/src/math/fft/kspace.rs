//! K-space operations for spectral methods
//!
//! This module handles wavenumber calculations for pseudospectral methods.

use ndarray::{Array1, Array3, Zip};
use std::f64::consts::PI;

/// K-space calculator for spectral methods
#[derive(Debug)]
pub struct KSpaceCalculator;

impl KSpaceCalculator {
    /// Generate k-space wavenumbers for one dimension
    #[must_use] 
    pub fn generate_k_vector(n: usize, dx: f64) -> Array1<f64> {
        Array1::from_vec(apollo::fftfreq(n, dx)).mapv(|cycles_per_unit| 2.0 * PI * cycles_per_unit)
    }

    /// Generate 3D k-squared array for Laplacian operations
    /// # Panics
    /// - Panics if `kx contiguous`.
    /// - Panics if `ky contiguous`.
    /// - Panics if `kz contiguous`.
    ///
    #[must_use] 
    pub fn generate_k_squared(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Array3<f64> {
        let kx = Self::generate_k_vector(nx, dx);
        let ky = Self::generate_k_vector(ny, dy);
        let kz = Self::generate_k_vector(nz, dz);

        let mut k_squared = Array3::zeros((nx, ny, nz));

        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");
        Zip::indexed(&mut k_squared).par_for_each(|(i, j, k), val| {
            *val = kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
        });

        k_squared
    }

    /// Calculate maximum stable k-space value
    #[must_use] 
    pub fn max_k_stable(dx: f64, dy: f64, dz: f64) -> f64 {
        let kx_max = PI / dx;
        let ky_max = PI / dy;
        let kz_max = PI / dz;
        kz_max.mul_add(kz_max, ky_max.mul_add(ky_max, kx_max.powi(2))).sqrt()
    }

    /// Calculate k-space correction factor for heterogeneous media
    #[must_use]
    pub fn kspace_correction_factor(kx: f64, ky: f64, kz: f64, dx: f64, dy: f64, dz: f64) -> f64 {
        let sinc_x = if kx.abs() > 1e-10 {
            let x = kx * dx / 2.0;
            x.sin() / x
        } else {
            1.0
        };

        let sinc_y = if ky.abs() > 1e-10 {
            let x = ky * dy / 2.0;
            x.sin() / x
        } else {
            1.0
        };

        let sinc_z = if kz.abs() > 1e-10 {
            let x = kz * dz / 2.0;
            x.sin() / x
        } else {
            1.0
        };

        sinc_x * sinc_y * sinc_z
    }
}
