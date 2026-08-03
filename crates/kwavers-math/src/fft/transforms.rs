use super::cache::Fft3d;
use eunomia::Complex64;
use leto::{Array1, Array2, Array3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

const FFT_ASSIGN_CHUNK_LEN: usize = 4096;

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
    <f64 as apollo::PlanCacheProvider>::get_1d_plan(apollo::Shape1D { n: data.len() })
        .forward_complex_slice_inplace(data);
}

/// Inverse 1-D complex FFT over a dense slice.
///
/// This is the slice counterpart of [`ifft_1d_complex_inplace`].
pub fn ifft_1d_complex_slice_inplace(data: &mut [Complex64]) {
    <f64 as apollo::PlanCacheProvider>::get_1d_plan(apollo::Shape1D { n: data.len() })
        .inverse_complex_slice_inplace(data);
}

/// Forward 2-D FFT of a real Leto array.
#[must_use]
pub fn fft_2d_array(field: &Array2<f64>) -> Array2<Complex64> {
    let [nx, ny] = field.shape();
    let mut out = Array2::from_elem([nx, ny], Complex64::default());
    assign_real_to_complex_2d(field, &mut out);
    fft_2d_complex_inplace(&mut out);
    out
}

/// Inverse 2-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_2d_array(field_hat: &Array2<Complex64>) -> Array2<f64> {
    let [nx, ny] = field_hat.shape();
    let mut spectrum = field_hat.clone();
    ifft_2d_complex_inplace(&mut spectrum);
    let mut out = Array2::zeros([nx, ny]);
    assign_complex_real_2d(&spectrum, &mut out);
    out
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
    let [nx, ny, nz] = field.shape();
    let mut out = Array3::from_elem([nx, ny, nz], Complex64::default());
    fft_3d_array_into(field, &mut out);
    out
}

/// Forward 3-D FFT of a real Leto array into caller-owned storage.
/// Routes to Apollo's zero-alloc `fft_3d_array_into`, avoiding intermediate
/// allocation and element-wise conversion.
pub fn fft_3d_array_into(field: &Array3<f64>, out: &mut Array3<Complex64>) {
    assert_eq!(
        field.shape(),
        out.shape(),
        "fft_3d_array_into: input and output shapes must match"
    );
    assign_real_to_complex_3d(field, out);
    fft_3d_complex_inplace(out);
}

/// Inverse 3-D FFT of a complex Leto array, returning the real component.
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex64>) -> Array3<f64> {
    let [nx, ny, nz] = field_hat.shape();
    let mut spectrum = field_hat.clone();
    let mut out = Array3::zeros([nx, ny, nz]);
    ifft_3d_array_into(&mut spectrum, &mut out);
    out
}

/// Inverse 3-D FFT into caller-owned real storage.
/// Routes to Apollo's zero-alloc `ifft_3d_array_into`, avoiding intermediate
/// allocation and element-wise copy.
pub fn ifft_3d_array_into(field_hat: &mut Array3<Complex64>, out: &mut Array3<f64>) {
    assert_eq!(
        field_hat.shape(),
        out.shape(),
        "ifft_3d_array_into: input and output shapes must match"
    );
    ifft_3d_complex_inplace(field_hat);
    assign_complex_real_3d(field_hat, out);
}

/// Forward 3-D complex FFT, allocating output.
#[must_use]
pub fn fft_3d_complex(field: &Array3<Complex64>) -> Array3<Complex64> {
    let mut out = field.clone();
    fft_3d_complex_inplace(&mut out);
    out
}

/// Forward 3-D complex FFT into caller-owned storage.
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

fn assign_real_to_complex_2d(real: &Array2<f64>, complex: &mut Array2<Complex64>) {
    assert_eq!(
        real.shape(),
        complex.shape(),
        "real and complex 2-D FFT arrays must have equal shapes"
    );

    if let (Some(real_values), Some(complex_values)) = (real.as_slice(), complex.as_slice_mut()) {
        assign_real_slice_to_complex(real_values, complex_values);
        return;
    }

    for ([i, j], &real_value) in real.indexed_iter() {
        complex[[i, j]] = Complex64::new(real_value, 0.0);
    }
}

fn assign_real_to_complex_3d(real: &Array3<f64>, complex: &mut Array3<Complex64>) {
    assert_eq!(
        real.shape(),
        complex.shape(),
        "real and complex 3-D FFT arrays must have equal shapes"
    );

    if let (Some(real_values), Some(complex_values)) = (real.as_slice(), complex.as_slice_mut()) {
        assign_real_slice_to_complex(real_values, complex_values);
        return;
    }

    for ([i, j, k], &real_value) in real.indexed_iter() {
        complex[[i, j, k]] = Complex64::new(real_value, 0.0);
    }
}

fn assign_complex_real_2d(complex: &Array2<Complex64>, real: &mut Array2<f64>) {
    assert_eq!(
        complex.shape(),
        real.shape(),
        "complex and real 2-D FFT arrays must have equal shapes"
    );

    if let (Some(complex_values), Some(real_values)) = (complex.as_slice(), real.as_slice_mut()) {
        assign_complex_slice_real(complex_values, real_values);
        return;
    }

    for ([i, j], complex_value) in complex.indexed_iter() {
        real[[i, j]] = complex_value.re;
    }
}

fn assign_complex_real_3d(complex: &Array3<Complex64>, real: &mut Array3<f64>) {
    assert_eq!(
        complex.shape(),
        real.shape(),
        "complex and real 3-D FFT arrays must have equal shapes"
    );

    if let (Some(complex_values), Some(real_values)) = (complex.as_slice(), real.as_slice_mut()) {
        assign_complex_slice_real(complex_values, real_values);
        return;
    }

    for ([i, j, k], complex_value) in complex.indexed_iter() {
        real[[i, j, k]] = complex_value.re;
    }
}

fn assign_real_slice_to_complex(real_values: &[f64], complex_values: &mut [Complex64]) {
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        complex_values,
        FFT_ASSIGN_CHUNK_LEN,
        |chunk_index, chunk| {
            let base = chunk_index * FFT_ASSIGN_CHUNK_LEN;
            for (offset, complex_value) in chunk.iter_mut().enumerate() {
                *complex_value = Complex64::new(real_values[base + offset], 0.0);
            }
        },
    );
}

fn assign_complex_slice_real(complex_values: &[Complex64], real_values: &mut [f64]) {
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        real_values,
        FFT_ASSIGN_CHUNK_LEN,
        |chunk_index, chunk| {
            let base = chunk_index * FFT_ASSIGN_CHUNK_LEN;
            for (offset, real_value) in chunk.iter_mut().enumerate() {
                *real_value = complex_values[base + offset].re;
            }
        },
    );
}
