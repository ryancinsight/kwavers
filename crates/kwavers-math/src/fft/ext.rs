use super::cache::{Fft2d, Fft3d};
use super::transforms::{
    fft_2d_array, fft_3d_array, fft_3d_array_into, ifft_2d_array, ifft_3d_array, ifft_3d_array_into,
};
use eunomia::Complex64;
use leto::{Array2, Array3};
use std::cell::RefCell;

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
        let _ = self;
        debug_assert_eq!(
            field.shape(),
            out.shape(),
            "Fft2dInOutExt::forward_into: shape mismatch between real input and complex output"
        );
        out.assign(&fft_2d_array(field));
    }

    #[inline]
    fn inverse_into(
        &self,
        field_hat: &Array2<Complex64>,
        out: &mut Array2<f64>,
        _scratch: &mut Array2<Complex64>,
    ) {
        let _ = self;
        debug_assert_eq!(
            field_hat.shape(),
            out.shape(),
            "Fft2dInOutExt::inverse_into: shape mismatch between complex input and real output"
        );
        out.assign(&ifft_2d_array(field_hat));
    }
}

impl Fft3dInOutExt for Fft3d {
    #[inline]
    fn forward_into(&self, field: &Array3<f64>, out: &mut Array3<Complex64>) {
        let _ = self;
        fft_3d_array_into(field, out);
    }

    #[inline]
    fn inverse_into(
        &self,
        field_hat: &Array3<Complex64>,
        out: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        let _ = self;
        debug_assert_eq!(
            field_hat.shape(),
            scratch.shape(),
            "Fft3dInOutExt::inverse_into: shape mismatch between complex input and complex scratch"
        );
        scratch.assign(field_hat);
        ifft_3d_array_into(scratch, out);
    }

    #[inline]
    fn forward_r2c_into(&self, real: &Array3<f64>, half_out: &mut Array3<Complex64>) {
        let _ = self;
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
            fft_3d_array_into(real, full);
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
        let _ = self;
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
            ifft_3d_array_into(full, out);
        });
    }

    #[inline]
    fn forward(&self, real: &Array3<f64>) -> Array3<Complex64> {
        let _ = self;
        fft_3d_array(real)
    }

    #[inline]
    fn inverse(&self, spectrum: &Array3<Complex64>) -> Array3<f64> {
        let _ = self;
        ifft_3d_array(spectrum)
    }
}

#[cfg(test)]
mod r2c_optimized_tests {
    use super::Fft3dInOutExt;
    use crate::fft::{fft_3d_complex_inplace, get_fft_for_grid};
    use eunomia::Complex64;
    use leto::{Array3, Layout, VecStorage};

    fn check_shape(nx: usize, ny: usize, nz: usize) {
        let nz_c = nz / 2 + 1;
        let fft = get_fft_for_grid(nx, ny, nz);
        let real = Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
            let x = ((i * 131 + j * 17 + k * 7) % 101) as f64 / 101.0 - 0.5;
            (x * std::f64::consts::TAU).sin() + 0.3 * x + 0.1
        });

        let mut half_new = Array3::zeros([nx, ny, nz_c]);
        fft.forward_r2c_into(&real, &mut half_new);
        let mut full = real.mapv(|v| Complex64::new(v, 0.0));
        fft_3d_complex_inplace(&mut full);
        let ref_half = full.slice(&[(0, nx, 1), (0, ny, 1), (0, nz_c, 1)]).unwrap();
        let fwd_err = half_new
            .iter()
            .zip(ref_half.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            fwd_err < 1e-9,
            "forward_r2c({nx},{ny},{nz}) vs full-c2c reference: {fwd_err:.2e}"
        );

        let mut out = Array3::zeros([nx, ny, nz]);
        let mut scratch = Array3::zeros([nx, ny, nz_c]);
        fft.inverse_c2r_into(&half_new, &mut out, &mut scratch);
        let rt_err = out
            .iter()
            .zip(real.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            rt_err < 1e-9,
            "r2c→c2r round-trip ({nx},{ny},{nz}): {rt_err:.2e}"
        );
    }

    #[test]
    fn optimized_r2c_c2r_matches_reference_and_roundtrips() {
        check_shape(8, 6, 10);
        check_shape(7, 5, 9);
        check_shape(16, 16, 16);
        check_shape(12, 1, 8);
    }

    #[test]
    fn strided_half_spectrum_matches_contiguous_paths() {
        const NX: usize = 3;
        const NY: usize = 3;
        const NZ: usize = 4;
        const NZ_C: usize = NZ / 2 + 1;

        let fft = get_fft_for_grid(NX, NY, NZ);
        let real = Array3::from_shape_fn([NX, NY, NZ], |[i, j, k]| {
            (i * 37 + j * 11 + k * 5) as f64 / 17.0 - 2.0
        });
        let mut contiguous_half = Array3::zeros([NX, NY, NZ_C]);
        fft.forward_r2c_into(&real, &mut contiguous_half);

        let row_stride = isize::try_from(NZ_C).expect("test dimensions fit isize");
        let plane_stride = isize::try_from(NY * NZ_C).expect("test dimensions fit isize");
        let strided_layout = Layout::new([NX, NY, NZ_C], [row_stride, plane_stride, 1], 0);
        let mut strided_half = Array3::new(
            strided_layout,
            VecStorage::fill(NX * NY * NZ_C, Complex64::default()),
        )
        .expect("invariant: transposed test layout fits its owned storage");
        assert!(strided_half.as_slice().is_none());

        fft.forward_r2c_into(&real, &mut strided_half);
        for (actual, expected) in strided_half.iter().zip(contiguous_half.iter()) {
            assert_eq!(actual, expected);
        }

        let mut contiguous_real = Array3::zeros([NX, NY, NZ]);
        let mut contiguous_scratch = Array3::zeros([NX, NY, NZ_C]);
        fft.inverse_c2r_into(
            &contiguous_half,
            &mut contiguous_real,
            &mut contiguous_scratch,
        );
        let mut strided_real = Array3::zeros([NX, NY, NZ]);
        let mut strided_scratch = Array3::zeros([NX, NY, NZ_C]);
        fft.inverse_c2r_into(&strided_half, &mut strided_real, &mut strided_scratch);
        for (actual, expected) in strided_real.iter().zip(contiguous_real.iter()) {
            assert_eq!(actual, expected);
        }
    }
}
