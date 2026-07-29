//! Spectral kernels for fractional-Laplacian absorption.
//!
//! `build_k_power_spectrum` constructs the full-spectrum `|k|^p` weight array
//! consumed by `spectral_filter_into`, which evaluates `IFFT( weights · FFT(field) )`
//! on the periodic 3-D grid via caller-owned scratch buffers — zero allocation
//! when the buffers are reused across calls.
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

/// Compute `IFFT(weights · FFT(field))` on the periodic 3-D grid, writing the
/// result into the pre-allocated `out` buffer. `spatial_buf` holds a contiguous
/// copy of `field`; `spectrum_buf` holds the intermediate complex spectrum.
/// All three buffers must be shape `(n, n, n)`. The FFT plan is cached per
/// shape by the apollo-backed `PlanCacheProvider`.
///
/// # Panics
///
/// Panics if any buffer shape or the field length does not match `(n, n, n)`.
pub(super) fn spectral_filter_into(
    n: usize,
    field: &[f64],
    weights: &Array3<f64>,
    out: &mut Array3<f64>,
    spatial_buf: &mut Array3<f64>,
    spectrum_buf: &mut Array3<Complex64>,
) {
    assert_eq!(
        spatial_buf.shape(),
        [n, n, n],
        "spatial_buf shape must match (n,n,n)"
    );
    assert_eq!(
        spectrum_buf.shape(),
        [n, n, n],
        "spectrum_buf shape must match (n,n,n)"
    );
    assert_eq!(out.shape(), [n, n, n], "out shape must match (n,n,n)");
    assert_eq!(field.len(), n * n * n, "field length must be n³");

    spatial_buf
        .as_slice_mut()
        .expect("spatial_buf must be contiguous")
        .copy_from_slice(field);
    fft_3d_array_into(spatial_buf, spectrum_buf);
    spectrum_buf
        .iter_mut()
        .zip(weights.iter())
        .for_each(|(z, &w)| *z *= w);
    ifft_3d_array_into(spectrum_buf, out);
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

    /// Verify that `spectral_filter_into` round-trips correctly: for a
    /// unit-weight array `|k|^0` the output should match the input
    /// (modulo the DC-zero convention).
    #[test]
    fn spectral_filter_into_round_trips_with_unit_weights() {
        let n = 8;
        let cells = n * n * n;
        let field: Vec<f64> = (0..cells).map(|i| ((i * 17) as f64 * 0.1).sin()).collect();
        let weights = build_k_power_spectrum(n, 1.0e-4, 0.0);
        let mut out = Array3::zeros((n, n, n));
        let mut spatial_buf = Array3::zeros((n, n, n));
        let mut spectrum_buf = Array3::zeros((n, n, n));

        spectral_filter_into(
            n,
            &field,
            &weights,
            &mut out,
            &mut spatial_buf,
            &mut spectrum_buf,
        );

        let out_slice = out.as_slice().unwrap();
        let mut max_diff = 0.0;
        for (o, f) in out_slice.iter().zip(field.iter()) {
            let diff = (o - f).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        // DC bin (index 0) is forced to zero by the convention; all other
        // bins should be preserved exactly by the identity filter.
        assert!(
            max_diff < 1.0e-12,
            "max spectral-filter round-trip error: {max_diff}"
        );
    }
}
