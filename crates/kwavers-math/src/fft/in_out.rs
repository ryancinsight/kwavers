//! Caller-owned-storage transforms: the `Fft{2,3}dInOutExt` extension traits
//! and the real/complex assignment kernels they run on.

use leto::{Array2, Array3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

use super::plan::{Fft2d, Fft3d};
use super::Complex64;
use apollo::RealFftData;

const FFT_ASSIGN_CHUNK_LEN: usize = 4096;

/// Full-spectrum (nx, ny, nz) complex-to-complex 3-D transforms with caller-owned
/// real and complex storage.
///
/// Local Apollo now accepts Leto arrays and `eunomia::Complex64`. This extension
/// trait preserves Kwavers' Leto/`eunomia` spectral contract at one
/// boundary while Apollo remains the single FFT engine.
pub trait Fft3dInOutExt {
    /// Forward 3-D FFT of a real field into a caller-owned full-spectrum
    /// complex buffer. Equivalent to assigning `field + 0i` into `out` and
    /// running an in-place complex forward FFT.
    fn forward_into(&self, field: &Array3<f64>, out: &mut Array3<Complex64>);

    /// Inverse 3-D FFT of a full-spectrum complex field into a caller-owned
    /// real buffer using a caller-owned complex scratch. Equivalent to
    /// copying `field_hat` into `scratch`, running an in-place complex
    /// inverse FFT on `scratch`, and assigning the real component into `out`.
    fn inverse_into(
        &self,
        field_hat: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    );

    /// Forward real-to-complex 3-D FFT writing the **half-spectrum** `(nx, ny,
    /// nz/2+1)` of a real field through Apollo's half-spectrum pair: the z
    /// lanes by the real split, then x and y on the half volume. `half_out`
    /// must have shape `(nx, ny, nz/2+1)`; a strided `half_out` is filled
    /// through a contiguous staging copy.
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>);

    /// Inverse complex-to-real 3-D FFT from a **half-spectrum** `(nx, ny,
    /// nz/2+1)` into a real field: the real part of the full inverse of the
    /// half spectrum's Hermitian completion. The transform runs in `half_in`
    /// and overwrites it; a caller that still needs the spectrum copies it
    /// first.
    fn inverse_c2r_into(&self, half_in: &mut Array3<Complex64>, out: &mut Array3<f64>);

    /// Forward full-spectrum 3-D FFT of a real field, allocating the output.
    fn forward(&self, real: &Array3<f64>) -> Array3<Complex64>;

    /// Inverse full-spectrum 3-D FFT to a real field, allocating the output.
    fn inverse(&self, spectrum: &Array3<Complex64>) -> Array3<f64>;
}

/// 2-D counterpart to [`Fft3dInOutExt`] with identical semantics.
pub trait Fft2dInOutExt {
    /// Forward 2-D FFT of a real field into a caller-owned full-spectrum
    /// complex buffer.
    fn forward_into(&self, field: &Array2<f64>, out: &mut Array2<Complex64>);

    /// Inverse 2-D FFT of a full-spectrum complex field into a caller-owned
    /// real buffer using a caller-owned complex scratch.
    fn inverse_into(
        &self,
        field_hat: &Array2<Complex64>,
        out: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    );
}

impl Fft2dInOutExt for Fft2d {
    #[inline]
    fn forward_into(&self, field: &Array2<f64>, out: &mut Array2<Complex64>) {
        debug_assert_eq!(
            field.shape(),
            out.shape(),
            "Fft2dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        assign_real_to_complex_2d(field, out);
        self.forward_complex_inplace(out);
    }

    #[inline]
    fn inverse_into(
        &self,
        field_hat: &Array2<Complex64>,
        out: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    ) {
        debug_assert_eq!(
            field_hat.shape(),
            scratch.shape(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        debug_assert_eq!(
            field_hat.shape(),
            out.shape(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        scratch.assign(field_hat);
        self.inverse_complex_inplace(scratch);
        assign_complex_real_2d(scratch, out);
    }
}

impl Fft3dInOutExt for Fft3d {
    #[inline]
    fn forward_into(&self, field: &Array3<f64>, out: &mut Array3<Complex64>) {
        debug_assert_eq!(
            field.shape(),
            out.shape(),
            "Fft3dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        assign_real_to_complex_3d(field, out);
        self.forward_complex_inplace(out);
    }

    #[inline]
    fn inverse_into(
        &self,
        field_hat: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        debug_assert_eq!(
            field_hat.shape(),
            scratch.shape(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        debug_assert_eq!(
            field_hat.shape(),
            out.shape(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        scratch.assign(field_hat);
        self.inverse_complex_inplace(scratch);
        assign_complex_real_3d(scratch, out);
    }

    #[inline]
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>) {
        if half_out.as_slice().is_some() {
            <f64 as RealFftData>::forward_3d_half_into(self, real, half_out);
        } else {
            let mut staged = Array3::from_elem(half_out.shape(), Complex64::default());
            <f64 as RealFftData>::forward_3d_half_into(self, real, &mut staged);
            half_out.assign(&staged);
        }
    }

    #[inline]
    fn inverse_c2r_into(&self, half_in: &mut Array3<Complex64>, out: &mut Array3<f64>) {
        // Apollo transforms a C-contiguous spectrum in place; a strided one is
        // staged once, and so is a strided output.
        let mut staged_spectrum;
        let spectrum = if half_in.as_slice().is_some() {
            half_in
        } else {
            staged_spectrum = Array3::from_elem(half_in.shape(), Complex64::default());
            staged_spectrum.assign(&*half_in);
            &mut staged_spectrum
        };
        if out.as_slice().is_some() {
            <f64 as RealFftData>::inverse_3d_half_into(self, spectrum, out);
        } else {
            let mut staged = Array3::from_elem(out.shape(), 0.0_f64);
            <f64 as RealFftData>::inverse_3d_half_into(self, spectrum, &mut staged);
            out.assign(&staged);
        }
    }

    #[inline]
    fn forward(&self, real: &Array3<f64>) -> Array3<Complex64> {
        let mut out = real.mapv(|v| Complex64::new(v, 0.0));
        self.forward_complex_inplace(&mut out);
        out
    }

    #[inline]
    fn inverse(&self, spectrum: &Array3<Complex64>) -> Array3<f64> {
        let mut tmp = spectrum.clone();
        self.inverse_complex_inplace(&mut tmp);
        tmp.mapv(|c| c.re)
    }
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
