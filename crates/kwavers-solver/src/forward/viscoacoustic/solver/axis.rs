//! Allocation-free pseudospectral derivatives for viscoacoustic stepping.

use super::ViscoacousticMemorySolver;
use kwavers_math::fft::{
    fft_3d_axis_complex_inplace, ifft_3d_axis_complex_inplace, Complex64, Fft3d,
};
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Compute `∂field/∂xₐ → out` with a retained per-axis FFT plan and scratch.
    ///
    /// A singleton axis has no spatial variation, so its derivative is exactly
    /// positive zero. That path clears `out` without reading `field`, touching
    /// the complex scratch, or dispatching either transform. For finite fields
    /// this is bitwise equal to the former FFT route; non-finite samples cannot
    /// contaminate a derivative along an inactive axis.
    pub(super) fn axis_derivative(
        fft: &Fft3d,
        k: &[f64],
        axis: usize,
        field: &Array3<f64>,
        cbuf: &mut Array3<Complex64>,
        out: &mut Array3<f64>,
    ) {
        let [nx, ny, nz] = cbuf.shape();
        assert_eq!(
            field.shape(),
            [nx, ny, nz],
            "invariant: viscoacoustic FFT scratch shape matches input field"
        );
        assert_eq!(
            out.shape(),
            [nx, ny, nz],
            "invariant: viscoacoustic derivative output shape matches input field"
        );
        let axis_len = match axis {
            0 => nx,
            1 => ny,
            2 => nz,
            _ => unreachable!("invariant: derivative axis is 0, 1, or 2"),
        };
        assert_eq!(
            k.len(),
            axis_len,
            "invariant: derivative wavenumbers match the selected axis"
        );
        if axis_len == 1 {
            out.fill(0.0);
            return;
        }

        if let (Some(dst), Some(src)) = (cbuf.as_slice_mut(), field.as_slice()) {
            for (dst, &src) in dst.iter_mut().zip(src) {
                *dst = Complex64::new(src, 0.0);
            }
        } else {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        cbuf[[x, y, z]] = Complex64::new(field[[x, y, z]], 0.0);
                    }
                }
            }
        }
        fft_3d_axis_complex_inplace(fft, cbuf, axis);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mode = match axis {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => unreachable!("invariant: derivative axis is 0, 1, or 2"),
                    };
                    cbuf[[x, y, z]] *= Complex64::new(0.0, k[mode]);
                }
            }
        }
        ifft_3d_axis_complex_inplace(fft, cbuf, axis);
        if let (Some(dst), Some(src)) = (out.as_slice_mut(), cbuf.as_slice()) {
            for (dst, src) in dst.iter_mut().zip(src) {
                *dst = src.re;
            }
        } else {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        out[[x, y, z]] = cbuf[[x, y, z]].re;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_axis_derivative(
        fft: &Fft3d,
        k: &[f64],
        axis: usize,
        field: &Array3<f64>,
        scratch: &mut Array3<Complex64>,
        out: &mut Array3<f64>,
    ) {
        let [nx, ny, nz] = field.shape();
        for (dst, &src) in scratch
            .as_slice_mut()
            .expect("test scratch is dense")
            .iter_mut()
            .zip(field.as_slice().expect("test field is dense"))
        {
            *dst = Complex64::new(src, 0.0);
        }
        fft_3d_axis_complex_inplace(fft, scratch, axis);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mode = [x, y, z][axis];
                    scratch[[x, y, z]] *= Complex64::new(0.0, k[mode]);
                }
            }
        }
        ifft_3d_axis_complex_inplace(fft, scratch, axis);
        for (dst, src) in out
            .as_slice_mut()
            .expect("test output is dense")
            .iter_mut()
            .zip(scratch.as_slice().expect("test scratch is dense"))
        {
            *dst = src.re;
        }
    }

    fn assert_complex_bits_eq(actual: &Array3<Complex64>, expected: &Array3<Complex64>) {
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }

    fn check_singleton_axis(shape: [usize; 3], axis: usize) {
        let [nx, ny, nz] = shape;
        let mut solver = ViscoacousticMemorySolver::new(
            nx,
            ny,
            nz,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
            1_000.0,
            2.25e9,
            &[],
        )
        .expect("test solver parameters are valid");
        let fft = solver.fft.clone();
        let k = match axis {
            0 => solver.kx.clone(),
            1 => solver.ky.clone(),
            2 => solver.kz.clone(),
            _ => unreachable!("test cases use a valid axis"),
        };

        for seed in 0..3 {
            let field = Array3::from_shape_fn((nx, ny, nz), |[x, y, z]| {
                ((17 * x + 11 * y + 5 * z + seed) as f64 * 0.37).sin()
            });
            let mut reference_scratch = Array3::zeros((nx, ny, nz));
            let mut expected = Array3::from_elem((nx, ny, nz), f64::NAN);
            reference_axis_derivative(
                &fft,
                &k,
                axis,
                &field,
                &mut reference_scratch,
                &mut expected,
            );

            solver.cbuf.fill(Complex64::new(seed as f64 + 1.0, -3.0));
            let scratch_before = solver.cbuf.clone();
            let mut actual = Array3::from_elem((nx, ny, nz), f64::NAN);
            ViscoacousticMemorySolver::axis_derivative(
                &fft,
                &k,
                axis,
                &field,
                &mut solver.cbuf,
                &mut actual,
            );

            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.to_bits(), expected.to_bits());
                assert_eq!(actual.to_bits(), 0.0_f64.to_bits());
            }
            assert_complex_bits_eq(&solver.cbuf, &scratch_before);
        }
    }

    #[test]
    fn singleton_axes_match_fft_reference_without_touching_scratch() {
        check_singleton_axis([8, 1, 1], 1);
        check_singleton_axis([8, 1, 1], 2);
        check_singleton_axis([4, 3, 1], 2);
    }

    #[test]
    fn singleton_axis_isolates_non_finite_samples_without_touching_scratch() {
        let mut solver = ViscoacousticMemorySolver::new(
            4,
            1,
            1,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
            1_000.0,
            2.25e9,
            &[],
        )
        .expect("test solver parameters are valid");
        let field = Array3::from_shape_fn((4, 1, 1), |[x, _, _]| match x {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            _ => -0.0,
        });
        solver.cbuf.fill(Complex64::new(7.0, -11.0));
        let scratch_before = solver.cbuf.clone();
        let mut derivative = Array3::from_elem((4, 1, 1), f64::NAN);

        ViscoacousticMemorySolver::axis_derivative(
            &solver.fft,
            &solver.ky,
            1,
            &field,
            &mut solver.cbuf,
            &mut derivative,
        );

        for value in &derivative {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }
        assert_complex_bits_eq(&solver.cbuf, &scratch_before);
    }
}
