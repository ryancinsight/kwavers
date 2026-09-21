//! The density updates: the fused form that folds the PML factor into the
//! lane pass, and the unfused form for the interior.

use super::coefficient::DenseRealField;
use crate::forward::lanes::{axis_index, for_each_z_lane, LaneAxis};
use leto::Array3 as LetoArray3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

pub(super) fn update_density_fused(
    density: &mut LetoArray3<f64>,
    divergence: &LetoArray3<f64>,
    coefficient: &impl DenseRealField,
    pml: &[f64],
    axis: LaneAxis,
    dt: f64,
) {
    assert_eq!(
        density.shape(),
        divergence.shape(),
        "invariant: PSTD density shape matches divergence"
    );
    assert_eq!(
        density.shape(),
        coefficient.shape3(),
        "invariant: PSTD density shape matches update coefficient"
    );

    let [_nx, ny, nz] = density.shape();
    if let (Some(density_values), Some(div_values), Some(coef_values)) = (
        density.as_slice_mut(),
        divergence.as_slice(),
        coefficient.as_dense_slice(),
    ) {
        for_each_z_lane(
            density_values,
            [ny, nz],
            3 * size_of::<f64>(),
            |start, i, j, density| {
                let coefficient = &coef_values[start..start + nz];
                let divergence = &div_values[start..start + nz];
                let lane = density.iter_mut().zip(coefficient).zip(divergence);
                match axis {
                    LaneAxis::X | LaneAxis::Y => {
                        let p = pml[axis_index(axis, i, j, 0)];
                        for ((density, &coefficient), &divergence) in lane {
                            *density = p * (p * *density - dt * coefficient * divergence);
                        }
                    }
                    LaneAxis::Z => {
                        for (((density, &coefficient), &divergence), &p) in lane.zip(&pml[..nz]) {
                            *density = p * (p * *density - dt * coefficient * divergence);
                        }
                    }
                }
            },
        );
        return;
    }

    let [nx, ny, nz] = density.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = pml[axis_index(axis, i, j, k)];
                density[[i, j, k]] = p
                    * (p * density[[i, j, k]]
                        - dt * coefficient.value(i, j, k) * divergence[[i, j, k]]);
            }
        }
    }
}

pub(super) fn update_density_unfused(
    density: &mut LetoArray3<f64>,
    divergence: &LetoArray3<f64>,
    coefficient: &impl DenseRealField,
    dt: f64,
) {
    assert_eq!(
        density.shape(),
        divergence.shape(),
        "invariant: PSTD density shape matches divergence"
    );
    assert_eq!(
        density.shape(),
        coefficient.shape3(),
        "invariant: PSTD density shape matches update coefficient"
    );

    if let (Some(density_values), Some(div_values), Some(coef_values)) = (
        density.as_slice_mut(),
        divergence.as_slice(),
        coefficient.as_dense_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(density_values, |index, density| {
            *density -= dt * coef_values[index] * div_values[index];
        });
        return;
    }

    let [nx, ny, nz] = density.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                density[[i, j, k]] -= dt * coefficient.value(i, j, k) * divergence[[i, j, k]];
            }
        }
    }
}
