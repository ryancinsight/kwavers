//! K-space operations for spectral methods
//!
//! This module handles wavenumber calculations for pseudospectral methods.

use kwavers_core::constants::numerical::TWO_PI;
use leto::{Array1, Array3};
use moirai_parallel::{for_each_unit_task_mut_with, Adaptive};
use std::f64::consts::PI;

/// K-space calculator for spectral methods
#[derive(Debug)]
pub struct KSpaceCalculator;

impl KSpaceCalculator {
    /// Generate k-space wavenumbers for one dimension
    ///
    /// # Panics
    ///
    /// Panics if the frequency vector returned by Apollo cannot be represented
    /// with the requested one-dimensional shape. Apollo returns one value per
    /// requested sample, so this indicates a violated upstream shape contract.
    #[must_use]
    pub fn generate_k_vector(n: usize, dx: f64) -> Array1<f64> {
        let _freqs = apollo::fftfreq(n, dx);
        let _n = _freqs.len();
        Array1::from_vec([_n], _freqs)
            .expect("k-space length")
            .mapv(|cycles_per_unit| TWO_PI * cycles_per_unit)
    }
}

/// Build a 1-D vector of signed wavenumbers using the **positive-Nyquist**
/// convention.
///
/// For each index `i` in `0..n`:
/// - `i ≤ n/2`: wavenumber = `+i · dk`  (zero and positive frequencies)
/// - `i > n/2`: wavenumber = `(i − n) · dk`  (negative frequencies)
///
/// where `dk = 2π / (n · d)`.
///
/// This convention maps the Nyquist bin `n/2` to the **positive** Nyquist
/// frequency `+π/d`, matching the physics convention used throughout the
/// kwavers spectral solvers.  It differs from `Apollo::fftfreq` (and NumPy
/// `fftfreq`) which map `n/2` to the negative Nyquist for even-length arrays.
///
/// # Note
///
/// `KSpaceCalculator::generate_k_vector` uses the Apollo / NumPy (negative-
/// at-Nyquist) convention and is **not** the same as this function.
#[inline]
#[must_use]
pub fn wavenumber_vec_positive_nyquist(n: usize, d: f64) -> Vec<f64> {
    use eunomia::F64;

    // Counts convert through the conversion module that owns them:
    // `from_count` is exact below 2^53 and rounds to nearest, ties to even,
    // above it -- the same mapping the `as` casts performed. The negative
    // branch negates the exact magnitude instead of routing through a signed
    // index type, so no `usize -> isize` conversion exists at all; negation
    // commutes with the scaling multiplication exactly.
    let dk = TWO_PI / (F64::from_count(n).0 * d);
    (0..n)
        .map(|i| {
            if i <= n / 2 {
                F64::from_count(i).0 * dk
            } else {
                -(F64::from_count(n - i).0 * dk)
            }
        })
        .collect()
}

impl KSpaceCalculator {
    /// Generate 3D k-squared array for Laplacian operations
    ///
    /// # Panics
    ///
    /// Panics if one of the generated vectors or the newly allocated output
    /// cannot be viewed as contiguous storage. These arrays are created as
    /// dense owned buffers, so this indicates a violated storage invariant.
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
            .as_slice_mut()
            .expect("newly allocated k-squared field is contiguous");
        let plane_len = ny * nz;
        // One unit writes one wavenumber; the three k vectors it reads are
        // shared and small, so the width comes from the output byte.
        for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
            values,
            1,
            core::mem::size_of::<f64>(),
            || (),
            |(), first, run| {
                for (offset, val) in run.iter_mut().enumerate() {
                    let linear = first + offset;
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

#[cfg(test)]
mod tests {
    use super::KSpaceCalculator;

    #[test]
    fn every_wavenumber_is_the_sum_of_its_axis_squares() {
        // 32768 cells is several unit tasks wide at one output byte each, and
        // past the policy's parallel floor.
        const N: usize = 32;
        let (dx, dy, dz) = (1.0e-3, 2.0e-3, 5.0e-4);

        let k_squared = KSpaceCalculator::generate_k_squared(N, N, N, dx, dy, dz);
        let kx = KSpaceCalculator::generate_k_vector(N, dx);
        let ky = KSpaceCalculator::generate_k_vector(N, dy);
        let kz = KSpaceCalculator::generate_k_vector(N, dz);

        assert_eq!(k_squared.shape(), [N, N, N]);
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    let expected = kz[k].mul_add(kz[k], kx[i].mul_add(kx[i], ky[j] * ky[j]));
                    assert_eq!(
                        k_squared[[i, j, k]].to_bits(),
                        expected.to_bits(),
                        "k_squared[{i}, {j}, {k}]"
                    );
                }
            }
        }
    }
}
