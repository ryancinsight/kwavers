//! K-space operations for spectral methods
//!
//! This module handles wavenumber calculations for pseudospectral methods.

use kwavers_core::constants::numerical::TWO_PI;
use leto::{Array1, Array3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use std::f64::consts::PI;

const KSPACE_CHUNK_LEN: usize = 4096;

/// K-space calculator for spectral methods
#[derive(Debug)]
pub struct KSpaceCalculator;

impl KSpaceCalculator {
    /// Generate k-space wavenumbers for one dimension
    #[must_use]
    pub fn generate_k_vector(n: usize, dx: f64) -> Array1<f64> {
        let _freqs = apollo::fftfreq(n, dx);
        let _n = _freqs.len();
        Array1::from_vec([_n], _freqs)
            .expect("k-space length")
            .mapv(|cycles_per_unit| TWO_PI * cycles_per_unit)
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

        let mut k_squared = Array3::zeros([nx, ny, nz]);

        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");
        let values = k_squared
            .as_slice_memory_order_mut()
            .expect("newly allocated k-squared field is contiguous");
        let plane_len = ny * nz;
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            values,
            KSPACE_CHUNK_LEN,
            |chunk_index, chunk| {
                let base = chunk_index * KSPACE_CHUNK_LEN;
                for (offset, val) in chunk.iter_mut().enumerate() {
                    let linear = base + offset;
                    let i = linear / plane_len;
                    let rem = linear % plane_len;
                    let j = rem / nz;
                    let k = rem % nz;
                    *val = kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
                }
            },
        );

        k_squared
    }

    /// Calculate maximum stable k-space value
    #[must_use]
    pub fn max_k_stable(dx: f64, dy: f64, dz: f64) -> f64 {
        let kx_max = PI / dx;
        let ky_max = PI / dy;
        let kz_max = PI / dz;
        kz_max
            .mul_add(kz_max, ky_max.mul_add(ky_max, kx_max.powi(2)))
            .sqrt()
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
