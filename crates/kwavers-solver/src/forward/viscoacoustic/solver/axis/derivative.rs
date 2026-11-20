//! Allocation-free pseudospectral derivatives for viscoacoustic stepping.

use super::super::ViscoacousticMemorySolver;
use crate::forward::lanes::for_each_x_plane;
use kwavers_math::fft::{
    fft_3d_axis_complex_inplace, ifft_3d_axis_complex_inplace, Complex64, Fft3d,
};
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Compute `∂field/∂xₐ → out` with a retained per-axis FFT plan and scratch.
    ///
    /// A singleton axis has no spatial variation, so its derivative is exactly
    /// positive zero. That path clears `out` without reading `field`, touching
    /// the complex scratch, or dispatching either transform. The FFT reference
    /// may represent the same exact zero with either IEEE sign. A non-finite
    /// sample cannot contaminate an inactive-axis derivative.
    pub(in crate::forward::viscoacoustic::solver) fn axis_derivative(
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

        // Every pass walks the C-order volume in storage order, x plane by x
        // plane: a plane of `ny·nz` elements is contiguous, so no step strides
        // across the buffer, and the axis is resolved once per pass.
        let plane_len = ny * nz;
        let copy_bytes = size_of::<Complex64>() + size_of::<f64>();
        if let (Some(dst), Some(src)) = (cbuf.as_slice_mut(), field.as_slice()) {
            for_each_x_plane(dst, plane_len, copy_bytes, |x, plane| {
                let start = x * plane_len;
                for (dst, &src) in plane.iter_mut().zip(&src[start..start + plane_len]) {
                    *dst = Complex64::new(src, 0.0);
                }
            });
        } else {
            for x in 0..nx {
                for y in 0..ny {
                    for z in 0..nz {
                        cbuf[[x, y, z]] = Complex64::new(field[[x, y, z]], 0.0);
                    }
                }
            }
        }
        fft_3d_axis_complex_inplace(fft, cbuf, axis);
        apply_axis_wavenumbers(cbuf, k, axis);
        ifft_3d_axis_complex_inplace(fft, cbuf, axis);
        if let (Some(dst), Some(src)) = (out.as_slice_mut(), cbuf.as_slice()) {
            for_each_x_plane(dst, plane_len, copy_bytes, |x, plane| {
                let start = x * plane_len;
                for (dst, src) in plane.iter_mut().zip(&src[start..start + plane_len]) {
                    *dst = src.re;
                }
            });
        } else {
            for x in 0..nx {
                for y in 0..ny {
                    for z in 0..nz {
                        out[[x, y, z]] = cbuf[[x, y, z]].re;
                    }
                }
            }
        }
    }
}

/// Multiplies each element of an axis-transformed spectrum by `i·k[mode]`,
/// `mode` its index along `axis`: once per x plane for x, once per z lane for
/// y, and zipped along each z lane for z.
fn apply_axis_wavenumbers(spectrum: &mut Array3<Complex64>, k: &[f64], axis: usize) {
    let [nx, ny, nz] = spectrum.shape();
    let Some(values) = spectrum.as_slice_mut() else {
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    spectrum[[x, y, z]] *= Complex64::new(0.0, k[[x, y, z][axis]]);
                }
            }
        }
        return;
    };
    let plane_len = ny * nz;
    let element_bytes = size_of::<Complex64>();
    match axis {
        0 => for_each_x_plane(values, plane_len, element_bytes, |x, plane| {
            let factor = Complex64::new(0.0, k[x]);
            for value in plane {
                *value *= factor;
            }
        }),
        1 => for_each_x_plane(values, plane_len, element_bytes, |_, plane| {
            for (lane, &ky) in plane.chunks_exact_mut(nz).zip(k) {
                let factor = Complex64::new(0.0, ky);
                for value in lane {
                    *value *= factor;
                }
            }
        }),
        2 => for_each_x_plane(values, plane_len, element_bytes, |_, plane| {
            for lane in plane.chunks_exact_mut(nz) {
                for (value, &kz) in lane.iter_mut().zip(k) {
                    *value *= Complex64::new(0.0, kz);
                }
            }
        }),
        _ => unreachable!("invariant: derivative axis is 0, 1, or 2"),
    }
}
