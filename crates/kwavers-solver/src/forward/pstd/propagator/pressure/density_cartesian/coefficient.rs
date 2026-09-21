//! The density update coefficient field, and the dense-field access the
//! update reads it through.

use leto::Array3 as LetoArray3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

pub(super) trait DenseRealField {
    fn shape3(&self) -> [usize; 3];
    fn as_dense_slice(&self) -> Option<&[f64]>;
    fn value(&self, i: usize, j: usize, k: usize) -> f64;
}

impl DenseRealField for LetoArray3<f64> {
    fn shape3(&self) -> [usize; 3] {
        self.shape()
    }

    fn as_dense_slice(&self) -> Option<&[f64]> {
        self.as_slice()
    }

    fn value(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

pub(super) fn compute_nonlinear_density_coefficient(
    coefficient: &mut LetoArray3<f64>,
    rho0: &LetoArray3<f64>,
    rhox: &LetoArray3<f64>,
    rhoy: &LetoArray3<f64>,
    rhoz: &LetoArray3<f64>,
) {
    assert_eq!(
        coefficient.shape(),
        rho0.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rho0"
    );
    assert_eq!(
        coefficient.shape(),
        rhox.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rhox"
    );
    assert_eq!(
        coefficient.shape(),
        rhoy.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rhoy"
    );
    assert_eq!(
        coefficient.shape(),
        rhoz.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rhoz"
    );

    if let (
        Some(coef_values),
        Some(rho0_values),
        Some(rx_values),
        Some(ry_values),
        Some(rz_values),
    ) = (
        coefficient.as_slice_mut(),
        rho0.as_slice(),
        rhox.as_slice(),
        rhoy.as_slice(),
        rhoz.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(coef_values, |index, coefficient| {
            *coefficient = 2.0f64.mul_add(
                rx_values[index] + ry_values[index] + rz_values[index],
                rho0_values[index],
            );
        });
        return;
    }

    let [nx, ny, nz] = coefficient.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                coefficient[[i, j, k]] = 2.0f64.mul_add(
                    rhox[[i, j, k]] + rhoy[[i, j, k]] + rhoz[[i, j, k]],
                    rho0[[i, j, k]],
                );
            }
        }
    }
}
