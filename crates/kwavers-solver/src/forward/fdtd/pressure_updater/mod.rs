//! FDTD pressure field update — SRP extraction from solver.rs.
//!
//! Pressure-related `impl FdtdSolver` extension blocks:
//! - `update`: dispatch, CPU, SIMD, GPU paths
//! - `nonlinear`: Westervelt correction and history rotation
//! - `divergence`: staggered-grid velocity divergence

use leto::Array3 as LetoArray3;
use leto::{Array3, ArrayView3};
use moirai_parallel::{enumerate_mut_with, Adaptive};

pub mod divergence;
pub mod nonlinear;
#[cfg(test)]
mod tests;
pub mod update;

pub(super) fn accumulate_two_fields(target: &mut Array3<f64>, x: &Array3<f64>, y: &Array3<f64>) {
    assert_eq!(
        target.shape(),
        x.shape(),
        "invariant: FDTD divergence x-gradient shape matches target"
    );
    assert_eq!(
        target.shape(),
        y.shape(),
        "invariant: FDTD divergence y-gradient shape matches target"
    );

    if let (Some(target_values), Some(x_values), Some(y_values)) =
        (target.as_slice_mut(), x.as_slice(), y.as_slice())
    {
        enumerate_mut_with::<Adaptive, _, _>(target_values, |idx, target_value| {
            *target_value += x_values[idx] + y_values[idx];
        });
    } else {
        leto_ops::zip2_mut_with(
            &mut target.view_mut(),
            &x.view(),
            &y.view(),
            |target_value, x_value, y_value| *target_value += *x_value + *y_value,
        )
        .expect("invariant: accumulate_two_fields shapes asserted equal above");
    }
}

pub(super) fn apply_pressure_update(
    pressure: &mut Array3<f64>,
    divergence: ArrayView3<'_, f64>,
    rho_c_squared: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        pressure.shape(),
        divergence.shape(),
        "invariant: FDTD divergence shape matches pressure field"
    );
    assert_eq!(
        pressure.shape(),
        rho_c_squared.shape(),
        "invariant: FDTD rho*c^2 shape matches pressure field"
    );

    if let (Some(pressure_values), Some(divergence_values), Some(rho_values)) = (
        pressure.as_slice_mut(),
        divergence.as_slice(),
        rho_c_squared.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |idx, pressure_value| {
            *pressure_value -= dt * rho_values[idx] * divergence_values[idx];
        });
    } else {
        for ((pressure_value, &divergence_value), &rho_value) in pressure
            .iter_mut()
            .zip(divergence.iter())
            .zip(rho_c_squared.iter())
        {
            *pressure_value -= dt * rho_value * divergence_value;
        }
    }
}

pub(super) fn add_nonlinear_pressure_delta(pressure: &mut LetoArray3<f64>, delta: &Array3<f64>) {
    assert_eq!(
        pressure.shape(),
        delta.shape(),
        "invariant: FDTD nonlinear pressure delta shape matches pressure field"
    );

    if let (Some(pressure_values), Some(delta_values)) = (pressure.as_slice_mut(), delta.as_slice())
    {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |idx, pressure_value| {
            *pressure_value += delta_values[idx];
        });
    } else {
        for (pressure_value, delta_value) in pressure
            .as_slice_mut()
            .expect("FDTD leto pressure field must be contiguous")
            .iter_mut()
            .zip(delta.iter())
        {
            *pressure_value += delta_value;
        }
    }
}
