//! Stolt f-k (frequency–wavenumber) migration for zero-angle plane-wave RF data.
//!
//! Migration focuses the hyperbolic diffraction moveout of each scatterer back to
//! its true position. The Stolt method does this in the 2-D Fourier domain in
//! `O(N log N)`: it remaps the data spectrum `S(k_x, ω)` onto the image spectrum
//! `R(k_x, k_z)` through the exploding-reflector dispersion relation
//!
//! ```text
//! ω = v · sign(k_z) · √(k_x² + k_z²),   v = c/2,
//! ```
//!
//! with the obliquity Jacobian `∂ω/∂k_z = v k_z / √(k_x²+k_z²)`, then inverse-FFTs
//! to the image `r(x, z)` sampled on the depth grid `z = v·t`. This is the
//! production-quality alternative to delay-and-sum for plane-wave imaging.
//!
//! # References
//! - Stolt, R. H. (1978). "Migration by Fourier transform." *Geophysics*, 43(1), 23–48.
//! - Garcia, D., et al. (2013). "Stolt's f-k migration for plane wave ultrasound
//!   imaging." *IEEE TUFFC*, 60(9), 1853–1867.

use kwavers_math::fft::{fft_2d_complex, ifft_2d_complex};
use leto::Array2 as LetoArray2;
use ndarray::Array2;
use kwavers_math::fft::Complex64;

const TAU: f64 = std::f64::consts::TAU;

/// Angular frequency / wavenumber at FFT bin `bin` for an axis of length `n`
/// sampled at spacing `d` (handles the negative-frequency wrap-around).
#[inline]
fn ang_bin(bin: usize, n: usize, d: f64) -> f64 {
    let m = if bin <= n / 2 {
        bin as f64
    } else {
        bin as f64 - n as f64
    };
    TAU * m / (n as f64 * d)
}

/// Stolt f-k migration of zero-angle plane-wave RF data.
///
/// - `data[ix, it]` = `s(x, t)` — receive RF after plane-wave transmit.
/// - `dx` lateral element pitch \[m]; `dt` time sampling \[s]; `sound_speed` `c` \[m/s].
///
/// Returns the migrated image `image[ix, iz]` = `r(x, z)` on the depth grid
/// `z = (c/2)·t` (same shape as the input). The exploding-reflector velocity
/// `v = c/2` accounts for the round trip.
#[must_use]
pub fn fk_stolt_migration(data: &Array2<f64>, dx: f64, dt: f64, sound_speed: f64) -> Array2<f64> {
    let (nx, nt) = data.dim();
    if nx == 0 || nt == 0 {
        return Array2::zeros((nx, nt));
    }
    let v = 0.5 * sound_speed;
    let dz = v * dt;

    // real RF → complex, then 2-D FFT over (x, t) → S(k_x, ω)
    let mut s0 = Array2::<Complex64>::zeros((nx, nt));
    for ((i, j), &val) in data.indexed_iter() {
        s0[[i, j]] = Complex64::new(val, 0.0);
    }
    let s = fft_2d_complex(&s0.into());

    let kx: Vec<f64> = (0..nx).map(|i| ang_bin(i, nx, dx)).collect();
    let omega_bin_scale = (nt as f64) * dt / TAU; // ω → continuous bin index

    // remap onto the image spectrum R(k_x, k_z)
    let mut r: LetoArray2<Complex64> = Array2::<Complex64>::zeros((nx, nt)).into();
    for i in 0..nx {
        let kx2 = kx[i] * kx[i];
        for l in 0..nt {
            let kz = ang_bin(l, nt, dz);
            let kmag = (kx2 + kz * kz).sqrt();
            if kmag == 0.0 {
                r[[i, l]] = s[[i, 0]]; // DC passes through
                continue;
            }
            // target ω on the data axis (same sign as k_z), then its fractional bin
            let omega = v * kmag * kz.signum();
            let m = (omega * omega_bin_scale).rem_euclid(nt as f64);
            if !m.is_finite() {
                continue;
            }
            let m0 = (m.floor() as usize) % nt;
            let m1 = (m0 + 1) % nt;
            let frac = m - m.floor();
            let sval = s[[i, m0]] * (1.0 - frac) + s[[i, m1]] * frac;
            let jacobian = v * kz.abs() / kmag; // obliquity factor
            r[[i, l]] = sval * jacobian;
        }
    }

    let img = ifft_2d_complex(&r);
    let img = img.mapv(|c| c.re);
    let [nx, nt] = img.shape();
    Array2::from_shape_vec(
        (nx, nt),
        img.as_slice()
            .expect("IFFT image must be densely stored")
            .to_vec(),
    )
    .expect("IFFT image shape must match its flattened length")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argmax2(a: &Array2<f64>) -> (usize, usize) {
        let mut best = (0usize, 0usize);
        let mut bv = f64::NEG_INFINITY;
        for ((i, j), &v) in a.indexed_iter() {
            if v.abs() > bv {
                bv = v.abs();
                best = (i, j);
            }
        }
        best
    }

    /// A flat (horizontal) reflector at depth z0 produces a flat event at
    /// t0 = 2 z0 / c. Migration must place its energy back at depth z0 = v·t0.
    #[test]
    fn flat_reflector_maps_to_correct_depth() {
        let (nx, nt) = (32, 256);
        let c = 1540.0;
        let dx = 3.0e-4;
        let dt = 1.0 / 20.0e6; // 20 MHz sampling
        let v = 0.5 * c;
        let dz = v * dt;

        // reflector depth and the flat arrival time
        let z0: f64 = 60.0 * dz; // 60 samples deep
        let t0: f64 = z0 / v; // = 2 z0 / c
        let j0 = (t0 / dt).round() as usize;

        // flat event: a short Gaussian pulse at t0, identical for every x
        let mut data = Array2::<f64>::zeros((nx, nt));
        let sigma = 2.0;
        for i in 0..nx {
            for j in 0..nt {
                let dj = j as f64 - j0 as f64;
                data[[i, j]] = (-(dj * dj) / (2.0 * sigma * sigma)).exp();
            }
        }

        let img = fk_stolt_migration(&data, dx, dt, c);
        // energy per depth row (kx=0 dominated); find the peak depth
        let mut depth_energy = vec![0.0; nt];
        for j in 0..nt {
            for i in 0..nx {
                depth_energy[j] += img[[i, j]] * img[[i, j]];
            }
        }
        let peak_depth = depth_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // expected migrated depth index ≈ z0/dz = 60
        let expected = (z0 / dz).round() as usize;
        assert!(
            (peak_depth as i64 - expected as i64).abs() <= 3,
            "flat reflector migrated to depth bin {peak_depth}, expected ≈ {expected}"
        );
    }

    /// A point scatterer produces a hyperbolic moveout; migration must collapse
    /// it to a focused spot at (x0, z0), and the focus must be sharper than the
    /// unmigrated data.
    #[test]
    fn point_scatterer_focuses_and_sharpens() {
        let (nx, nt) = (48, 384);
        let c = 1540.0;
        let dx = 3.0e-4;
        let dt = 1.0 / 20.0e6;
        let v = 0.5 * c;
        let dz = v * dt;

        let ix0 = nx / 2;
        let x0 = ix0 as f64 * dx;
        let z0 = 120.0 * dz;

        // hyperbolic moveout: t(x) = (1/v)·√((x−x0)² + z0²), Gaussian pulse on it
        let mut data = Array2::<f64>::zeros((nx, nt));
        let sigma = 1.5;
        for i in 0..nx {
            let x = i as f64 * dx;
            let t = ((x - x0).powi(2) + z0 * z0).sqrt() / v;
            let jc = t / dt;
            for j in 0..nt {
                let dj = j as f64 - jc;
                data[[i, j]] += (-(dj * dj) / (2.0 * sigma * sigma)).exp();
            }
        }

        let img = fk_stolt_migration(&data, dx, dt, c);
        let (pi, pj) = argmax2(&img);
        let expected_z = (z0 / dz).round() as usize;
        assert!(
            (pi as i64 - ix0 as i64).abs() <= 2,
            "lateral focus at x-bin {pi}, expected ≈ {ix0}"
        );
        assert!(
            (pj as i64 - expected_z as i64).abs() <= 5,
            "axial focus at z-bin {pj}, expected ≈ {expected_z}"
        );

        // focusing sharpens: migrated peak energy fraction > raw data's
        let peak = img[[pi, pj]].abs();
        let total: f64 = img.iter().map(|v| v.abs()).sum();
        let raw_peak = data.iter().cloned().fold(0.0_f64, f64::max);
        let raw_total: f64 = data.iter().map(|v| v.abs()).sum();
        assert!(
            peak / total > raw_peak / raw_total,
            "migration should concentrate energy (got {} vs raw {})",
            peak / total,
            raw_peak / raw_total
        );
    }
}
