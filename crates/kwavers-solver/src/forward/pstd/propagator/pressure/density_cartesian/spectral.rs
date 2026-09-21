//! The staggered-grid gradient step: a velocity spectrum times the shift
//! operator and the kappa correction, written into the gradient spectrum.

use crate::forward::lanes::{axis_index, for_each_z_lane, LaneAxis};
use kwavers_math::fft::Complex64;
use leto::Array1;
use leto::Array3 as LetoArray3;

pub(super) fn apply_shifted_kappa(
    grad_k: &mut LetoArray3<Complex64>,
    spectrum: &LetoArray3<Complex64>,
    kappa: &LetoArray3<f64>,
    shift: &Array1<Complex64>,
    axis: LaneAxis,
) {
    assert_eq!(
        grad_k.shape(),
        spectrum.shape(),
        "invariant: PSTD gradient spectrum shape matches velocity spectrum"
    );
    assert_eq!(
        grad_k.shape(),
        kappa.shape(),
        "invariant: PSTD gradient spectrum shape matches kappa"
    );

    let [_nx, ny, nz] = grad_k.shape();
    if let (Some(grad_values), Some(spectrum_values), Some(kappa_values), Some(shift_values)) = (
        grad_k.as_slice_mut(),
        spectrum.as_slice(),
        kappa.as_slice(),
        shift.as_slice(),
    ) {
        let element_bytes = 2 * size_of::<Complex64>() + size_of::<f64>();
        for_each_z_lane(grad_values, [ny, nz], element_bytes, |start, i, j, grad| {
            let spectrum = &spectrum_values[start..start + nz];
            let kappa = &kappa_values[start..start + nz];
            let lane = grad.iter_mut().zip(spectrum).zip(kappa);
            match axis {
                LaneAxis::X | LaneAxis::Y => {
                    let shift = shift_values[axis_index(axis, i, j, 0)];
                    for ((grad, &spectrum), &kappa) in lane {
                        *grad = (shift * spectrum) * kappa;
                    }
                }
                LaneAxis::Z => {
                    for (((grad, &spectrum), &kappa), &shift) in lane.zip(&shift_values[..nz]) {
                        *grad = (shift * spectrum) * kappa;
                    }
                }
            }
        });
        return;
    }

    let [nx, ny, nz] = grad_k.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                grad_k[[i, j, k]] =
                    (shift[axis_index(axis, i, j, k)] * spectrum[[i, j, k]]) * kappa[[i, j, k]];
            }
        }
    }
}
