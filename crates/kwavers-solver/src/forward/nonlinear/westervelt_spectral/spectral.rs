//! Full-spectrum Laplacian entry points for caller-supplied wavenumber tables.

use kwavers_math::fft::{fft_3d_array_into, ifft_3d_array, ifft_3d_array_into, Complex64};
use leto::Array3;

/// Return the periodic Laplacian `IFFT(−|k|² FFT(field))` in a new volume.
///
/// # Panics
///
/// Panics for empty FFT dimensions or differing field and multiplier shapes.
#[must_use]
pub fn compute_laplacian_spectral(field: &Array3<f64>, k_squared: &Array3<f64>) -> Array3<f64> {
    let mut spectrum = Array3::zeros(field.shape());
    let mut output = Array3::zeros(field.shape());
    compute_laplacian_spectral_into(field, k_squared, &mut spectrum, &mut output);
    output
}

/// Write the periodic Laplacian into `out` using the supplied full-spectrum
/// multiplier in FFT bin order. The complex scratch may be overwritten.
///
/// Contiguous scratch and output reuse their storage on a warm FFT plan.
/// Strided destinations use contiguous staging for the FFT provider.
///
/// # Panics
///
/// Panics for empty FFT dimensions or if the four shapes differ.
pub fn compute_laplacian_spectral_into(
    field: &Array3<f64>,
    k_squared: &Array3<f64>,
    fft_scratch: &mut Array3<Complex64>,
    out: &mut Array3<f64>,
) {
    let shape = field.shape();
    assert_eq!(
        shape,
        k_squared.shape(),
        "Laplacian multiplier shape mismatch"
    );
    assert_eq!(shape, fft_scratch.shape(), "FFT scratch shape mismatch");
    assert_eq!(shape, out.shape(), "Laplacian output shape mismatch");

    let mut staged_spectrum;
    let spectrum = if fft_scratch.as_slice().is_some() {
        fft_scratch
    } else {
        staged_spectrum = Array3::zeros(shape);
        &mut staged_spectrum
    };
    fft_3d_array_into(field, spectrum);
    for (value, &k_sq) in spectrum.iter_mut().zip(k_squared.iter()) {
        *value *= -k_sq;
    }
    if out.as_slice().is_some() {
        ifft_3d_array_into(spectrum, out);
    } else {
        out.assign(&ifft_3d_array(spectrum));
    }
}

#[cfg(test)]
mod tests;
