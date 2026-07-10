//! Spectral kernels for fractional-Laplacian absorption.
//!
//! `build_k_power_spectrum` constructs the full-spectrum `|k|^p` weight array
//! consumed by `spectral_filter`, which evaluates `IFFT( weights · FFT(field) )`
//! on the periodic 3-D grid via the kwavers FFT facade (apollo-backed, plan-cached).
//!
//! Apollo dropped its public real-to-complex (half-spectrum) transforms, so this
//! uses the full-spectrum complex round-trip. The result is identical: the weights
//! `|k|^p` are real and radially symmetric, hence Hermitian-symmetry-preserving, so
//! `IFFT(weights · FFT(real_field))` is real-valued and equals the half-spectrum
//! computation. The cost is a full `(n,n,n)` complex grid instead of `(n,n,n/2+1)`.

use kwavers_math::fft::{fft_3d_array_into, fftfreq, ifft_3d_array_into, Complex64};
use leto::Array3;

/// Build the `|k|^power` spectral-filter array of full-spectrum shape `(n, n, n)`
/// matching the complex FFT output layout. All three axes use `fftfreq` (cycles
/// per metre), scaled by `2π` to angular wavenumbers in rad/m.
pub(super) fn build_k_power_spectrum(n: usize, spacing_m: f64, power: f64) -> Array3<f64> {
    let kx = fftfreq(n, spacing_m);
    let ky = fftfreq(n, spacing_m);
    let kz = fftfreq(n, spacing_m);
    let two_pi = std::f64::consts::TAU;
    Array3::from_shape_fn((n, n, n), |[ix, iy, iz]| {
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
/// Allocates the complex spectrum buffer per call; the FFT plan itself is cached
/// per shape by the apollo-backed `PlanCacheProvider` behind the kwavers facade.
pub(super) fn spectral_filter(n: usize, field: &[f64], weights: &Array3<f64>) -> Vec<f64> {
    let mut spatial = Array3::<f64>::zeros((n, n, n));
    spatial
        .as_slice_mut()
        .expect("Array3<f64>::zeros is contiguous")
        .copy_from_slice(field);
    let spatial = spatial.into();
    let mut spectrum = Array3::<Complex64>::zeros((n, n, n)).into();
    fft_3d_array_into(&spatial, &mut spectrum);
    spectrum.iter_mut().zip(weights.iter()).for_each(|(z, &w)| {
        *z *= w;
    });
    let mut spatial = Array3::<f64>::zeros((n, n, n)).into();
    ifft_3d_array_into(&mut spectrum, &mut spatial);
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
