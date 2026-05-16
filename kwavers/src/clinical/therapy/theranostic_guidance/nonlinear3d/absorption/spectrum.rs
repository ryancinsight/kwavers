//! Real-to-complex spectral kernels for fractional-Laplacian absorption.
//!
//! `build_k_power_spectrum` constructs the half-spectrum `|k|^p` weight array
//! consumed by `spectral_filter`, which evaluates `IFFT( weights · FFT(field) )`
//! on the periodic 3-D grid via Apollo's cached real-FFT plans.

use apollo::{fft_3d_r2c_into, ifft_3d_r2c_into, Complex64};
use ndarray::Array3;

/// Build the `|k|^power` spectral-filter array of shape `(n, n, n/2 + 1)`
/// matching the R2C FFT output layout. `k_x` and `k_y` use the full-range
/// `apollo::fftfreq`; `k_z` uses the positive half-range `apollo::rfftfreq`.
/// Apollo returns frequencies in cycles per metre, so we multiply by `2π`
/// to convert to angular wavenumbers in rad/m.
pub(super) fn build_k_power_spectrum(n: usize, spacing_m: f64, power: f64) -> Array3<f64> {
    let nz_c = n / 2 + 1;
    let kx = apollo::fftfreq(n, spacing_m);
    let ky = apollo::fftfreq(n, spacing_m);
    let kz = apollo::rfftfreq(n, spacing_m);
    let two_pi = std::f64::consts::TAU;
    Array3::from_shape_fn((n, n, nz_c), |(ix, iy, iz)| {
        let kx_v = two_pi * kx[ix];
        let ky_v = two_pi * ky[iy];
        let kz_v = two_pi * kz[iz];
        let k_mag = (kx_v * kx_v + ky_v * ky_v + kz_v * kz_v).sqrt();
        if k_mag < 1.0e-12 {
            0.0 // DC bin: by convention `|k|^power → 0`
        } else {
            k_mag.powf(power)
        }
    })
}

/// Compute `IFFT( weights · FFT(field) )` on the periodic 3-D grid.
/// Allocates the FFT and IFFT scratch buffers per call; the FFT plan
/// itself is cached globally by Apollo's `FFT_CACHE_3D`.
pub(super) fn spectral_filter(n: usize, field: &[f64], weights: &Array3<f64>) -> Vec<f64> {
    let nz_c = n / 2 + 1;
    let mut spatial = Array3::<f64>::zeros((n, n, n));
    spatial
        .as_slice_mut()
        .expect("Array3<f64>::zeros is contiguous")
        .copy_from_slice(field);
    let mut spectrum = Array3::<Complex64>::zeros((n, n, nz_c));
    fft_3d_r2c_into(&spatial, &mut spectrum);
    spectrum.iter_mut().zip(weights.iter()).for_each(|(z, &w)| {
        *z *= w;
    });
    let mut scratch = Array3::<Complex64>::zeros((n, n, nz_c));
    ifft_3d_r2c_into(&spectrum, &mut spatial, &mut scratch);
    spatial
        .as_slice()
        .expect("Array3<f64>::zeros is contiguous")
        .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spectral-filter correctness: at `power = 0` the operator collapses
    /// to identity (modulo the DC zero), so applying it to a uniform
    /// non-zero field returns the same field on AC bins. We verify the
    /// `|k|^0 = 1` invariant on the highest-frequency bin.
    #[test]
    fn k_power_spectrum_is_unity_for_power_zero_at_nyquist() {
        let n = 16;
        let spacing_m = 1.0e-4;
        let k = build_k_power_spectrum(n, spacing_m, 0.0);
        let nyquist_corner = k[[n / 2, n / 2, n / 2]];
        assert!(
            (nyquist_corner - 1.0).abs() < 1.0e-12,
            "|k|^0 at Nyquist corner must equal 1.0; got {nyquist_corner}",
        );
        // DC bin must be zero by convention.
        assert!(k[[0, 0, 0]].abs() < 1.0e-30);
    }
}
