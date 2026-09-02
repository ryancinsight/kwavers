//! Caller-owned-storage transforms: the `Fft{2,3}dInOutExt` extension traits
//! and the real/complex assignment kernels they run on.

use std::cell::RefCell;

use leto::{Array2, Array3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

use super::plan::{Fft2d, Fft3d};
use super::Complex64;

const FFT_ASSIGN_CHUNK_LEN: usize = 4096;

thread_local! {
    /// Per-thread full-spectrum `(nx, ny, nz)` complex scratch used by the
    /// half-spectrum r2c/c2r emulation in [`Fft3dInOutExt`]. Apollo dropped its
    /// public half-spectrum transforms; this scratch lets the ACL run apollo's
    /// full-spectrum complex plan and truncate/expand to the `nz_c = nz/2 + 1`
    /// layout the PSTD core still uses. Resized on grid-shape change, then reused
    /// across timesteps (zero steady-state allocation for a fixed grid).
    static R2C_FULL_SCRATCH: RefCell<Array3<Complex64>> =
        RefCell::new(Array3::from_elem([0, 0, 0], Complex64::default()));
}

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
    /// nz/2+1)` of a real field. The facade computes the full complex
    /// transform through Apollo's Leto path and stores the non-redundant
    /// z-spectrum. `half_out` must have shape `(nx, ny, nz/2+1)`.
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>);

    /// Inverse complex-to-real 3-D FFT from a **half-spectrum** `(nx, ny,
    /// nz/2+1)` into a real field. The facade reconstructs the full Hermitian
    /// spectrum and calls Apollo's full complex inverse path. The `scratch`
    /// argument is retained for call-site compatibility and is unused.
    /// `half_in` must have shape `(nx, ny, nz/2+1)`.
    fn inverse_c2r_into(
        &self,
        half_in: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    );

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
        let [nx, ny, nz] = real.shape();
        let nz_c = nz / 2 + 1;
        debug_assert_eq!(
            half_out.shape(),
            [nx, ny, nz_c],
            "forward_r2c_into: half_out must be (nx, ny, nz/2+1)"
        );
        R2C_FULL_SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.shape() != [nx, ny, nz] {
                *borrow = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
            }
            let full: &mut Array3<Complex64> = &mut borrow;
            assign_real_to_complex_3d(real, full);
            self.forward_complex_inplace(full);
            if let Some(half_values) = half_out.as_slice_mut() {
                let full_values = full
                    .as_slice()
                    .expect("invariant: thread-local FFT scratch is contiguous");
                for (full_row, half_row) in full_values
                    .chunks_exact(nz)
                    .zip(half_values.chunks_exact_mut(nz_c))
                {
                    half_row.copy_from_slice(&full_row[..nz_c]);
                }
            } else {
                half_out.assign(
                    &full
                        .slice(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)])
                        .expect("invariant: validated FFT half-spectrum slice"),
                );
            }
        });
    }

    #[inline]
    fn inverse_c2r_into(
        &self,
        half_in: &Array3<Complex64>,
        out: &mut Array3<f64>,
        _scratch: &mut Array3<Complex64>,
    ) {
        let [nx, ny, nz] = out.shape();
        let nz_c = nz / 2 + 1;
        debug_assert_eq!(
            half_in.shape(),
            [nx, ny, nz_c],
            "inverse_c2r_into: half_in must be (nx, ny, nz/2+1)"
        );
        R2C_FULL_SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.shape() != [nx, ny, nz] {
                *borrow = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
            }
            let full: &mut Array3<Complex64> = &mut borrow;
            if let Some(half_values) = half_in.as_slice() {
                let full_values = full
                    .as_slice_mut()
                    .expect("invariant: thread-local FFT scratch is contiguous");
                for i in 0..nx {
                    let ii = if i == 0 { 0 } else { nx - i };
                    for j in 0..ny {
                        let jj = if j == 0 { 0 } else { ny - j };
                        let full_row_start = (i * ny + j) * nz;
                        let half_row_start = (i * ny + j) * nz_c;
                        full_values[full_row_start..full_row_start + nz_c]
                            .copy_from_slice(&half_values[half_row_start..half_row_start + nz_c]);

                        let mirror_row_start = (ii * ny + jj) * nz_c;
                        for k in nz_c..nz {
                            full_values[full_row_start + k] =
                                half_values[mirror_row_start + nz - k].conj();
                        }
                    }
                }
            } else {
                full.slice_mut(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)])
                    .expect("invariant: validated FFT half-spectrum slice")
                    .assign(half_in);
                for k in nz_c..nz {
                    let kk = nz - k;
                    for i in 0..nx {
                        let ii = if i == 0 { 0 } else { nx - i };
                        for j in 0..ny {
                            let jj = if j == 0 { 0 } else { ny - j };
                            full[[i, j, k]] = half_in[[ii, jj, kk]].conj();
                        }
                    }
                }
            }
            self.inverse_complex_inplace(full);
            assign_complex_real_3d(full, out);
        });
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
