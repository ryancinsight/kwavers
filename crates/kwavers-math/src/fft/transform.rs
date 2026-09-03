//! Out-of-place and in-place transforms of Leto arrays through Apollo.

use apollo::{PlanCacheProvider, Shape1D};
use leto::{Array1, Array2, Array3};

use super::plan::Fft3d;
use super::Complex64;

/// Forward 1-D FFT of a real Leto array through Apollo's Leto-owned engine.
#[must_use]
pub fn fft_1d_array(field: &Array1<f64>) -> Array1<Complex64> {
    apollo::fft_1d_array(field)
}

/// Inverse 1-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_1d_array(field_hat: &Array1<Complex64>) -> Array1<f64> {
    apollo::ifft_1d_array(field_hat)
}

/// Forward 1-D complex FFT, allocating output.
#[must_use]
pub fn fft_1d_complex(field: &Array1<Complex64>) -> Array1<Complex64> {
    let mut out = field.clone();
    fft_1d_complex_inplace(&mut out);
    out
}

/// Inverse 1-D complex FFT, allocating output.
#[must_use]
pub fn ifft_1d_complex(field_hat: &Array1<Complex64>) -> Array1<Complex64> {
    let mut out = field_hat.clone();
    ifft_1d_complex_inplace(&mut out);
    out
}

/// Forward 1-D complex FFT in place.
pub fn fft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    apollo::fft_1d_complex_inplace(data);
}

/// Inverse 1-D complex FFT in place.
pub fn ifft_1d_complex_inplace(data: &mut Array1<Complex64>) {
    apollo::ifft_1d_complex_inplace(data);
}

/// Forward 1-D complex FFT over a dense slice.
///
/// This preserves the Kwavers `eunomia::Complex64` boundary while Apollo
/// owns Leto/eunomia-native execution internally.
pub fn fft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    let Ok(shape) = Shape1D::new(data.len()) else {
        // A zero-length transform has nothing to transform.
        return;
    };
    <f64 as PlanCacheProvider>::get_1d_plan(shape).forward_complex_slice_inplace(data);
}

/// Inverse 1-D complex FFT over a dense slice.
///
/// This is the slice counterpart of [`ifft_1d_complex_inplace`].
pub fn ifft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    let Ok(shape) = Shape1D::new(data.len()) else {
        // A zero-length transform has nothing to transform.
        return;
    };
    <f64 as PlanCacheProvider>::get_1d_plan(shape).inverse_complex_slice_inplace(data);
}

/// Forward 2-D FFT of a real Leto array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    apollo::fft_2d_array(field)
}

/// Inverse 2-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    apollo::ifft_2d_array(field_hat)
}

/// Forward 2-D complex FFT, allocating output.
#[must_use]
pub fn fft_2d_complex(field: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = field.clone();
    fft_2d_complex_inplace(&mut out);
    out
}

/// Inverse 2-D complex FFT, allocating output.
#[must_use]
pub fn ifft_2d_complex(field_hat: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = field_hat.clone();
    ifft_2d_complex_inplace(&mut out);
    out
}

/// Forward 2-D complex FFT in place.
pub fn fft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    apollo::fft_2d_complex_inplace(data);
}

/// Inverse 2-D complex FFT in place.
pub fn ifft_2d_complex_inplace(data: &mut Array2<Complex64>) {
    apollo::ifft_2d_complex_inplace(data);
}

/// Forward 3-D FFT of a real Leto array.
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex64> {
    apollo::fft_3d_array(field)
}

/// Forward 3-D FFT of a real Leto array into caller-owned storage.
/// Routes to Apollo's zero-alloc `fft_3d_array_into`, avoiding intermediate
/// allocation and element-wise conversion.
///
/// # Panics
///
/// Panics if `out` does not have the same three-dimensional shape as `field`.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    assert_eq!(
        field.shape(),
        out.shape(),
        "fft_3d_array_into: input and output shapes must match"
    );
    apollo::fft_3d_array_into(field, out);
}

/// Inverse 3-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    apollo::ifft_3d_array(field_hat)
}

/// Inverse 3-D FFT into caller-owned real storage.
///
/// Routes to Apollo's zero-alloc `ifft_3d_array_into_spectrum_scratch`, which
/// consumes `field_hat` as its own scratch, avoiding both an intermediate
/// allocation and an element-wise copy. Apollo 0.27 renamed this form: the
/// plain `ifft_3d_array_into` name now belongs to the three-argument variant
/// taking explicit scratch, matching its 1-D and 2-D siblings, which the
/// two-argument form had been inverted against.
///
/// # Panics
///
/// Panics if `out` does not have the same three-dimensional shape as
/// `field_hat`.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_3d_array_into: input and output shapes must match"
    );
    apollo::ifft_3d_array_into_spectrum_scratch(field_hat, out);
}

/// Forward 3-D complex FFT, allocating output.
#[must_use]
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let mut out = field.clone();
    fft_3d_complex_inplace(&mut out);
    out
}

/// Forward 3-D complex FFT into caller-owned storage.
///
/// # Panics
///
/// Panics if `out` does not have the same three-dimensional shape as `field`.
pub fn fft_3d_complex_into(field: &Array3<Complex64>, out: &mut Array3<Complex64>) {
    assert_eq!(
        field.shape(),
        out.shape(),
        "fft_3d_complex_into: input and output shapes must match"
    );
    out.assign(field);
    fft_3d_complex_inplace(out);
}

/// Inverse 3-D complex FFT, allocating output.
#[must_use]
pub fn ifft_3d_complex(field_hat: &Array3<Complex64>) -> Array3<Complex64> {
    let mut out = field_hat.clone();
    ifft_3d_complex_inplace(&mut out);
    out
}

/// Forward 3-D complex FFT in place.
pub fn fft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    apollo::fft_3d_complex_inplace(data);
}

/// Inverse 3-D complex FFT in place.
pub fn ifft_3d_complex_inplace(data: &mut Array3<Complex64>) {
    apollo::ifft_3d_complex_inplace(data);
}

/// Forward 3-D complex FFT along one axis in place.
///
/// Kwavers and Apollo share Leto storage and `eunomia::Complex64`, so this
/// delegates directly without allocating or converting the caller's field.
pub fn fft_3d_axis_complex_inplace(plan: &Fft3d, data: &mut Array3<Complex64>, axis: usize) {
    plan.forward_axis_complex_inplace(data, axis);
}

/// Inverse 3-D complex FFT along one axis in place.
///
/// This preserves the direct zero-copy contract of
/// [`fft_3d_axis_complex_inplace`].
pub fn ifft_3d_axis_complex_inplace(plan: &Fft3d, data: &mut Array3<Complex64>, axis: usize) {
    plan.inverse_axis_complex_inplace(data, axis);
}
