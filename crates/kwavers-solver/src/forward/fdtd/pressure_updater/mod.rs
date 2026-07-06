//! FDTD pressure field update — SRP extraction from solver.rs.
//!
//! Pressure-related `impl FdtdSolver` extension blocks:
//! - `update`: dispatch, CPU, SIMD, GPU paths
//! - `nonlinear`: Westervelt correction and history rotation
//! - `divergence`: staggered-grid velocity divergence

use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{Array3, ArrayView3, Zip};

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

    if let (Some(target_values), Some(x_values), Some(y_values)) = (
        target.as_slice_memory_order_mut(),
        x.as_slice_memory_order(),
        y.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(target_values, |idx, target_value| {
            *target_value += x_values[idx] + y_values[idx];
        });
    } else {
        Zip::from(target)
            .and(x)
            .and(y)
            .for_each(|target_value, &x_value, &y_value| *target_value += x_value + y_value);
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
        pressure.as_slice_memory_order_mut(),
        divergence.as_slice_memory_order(),
        rho_c_squared.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |idx, pressure_value| {
            *pressure_value -= dt * rho_values[idx] * divergence_values[idx];
        });
    } else {
        Zip::from(pressure)
            .and(divergence)
            .and(rho_c_squared)
            .for_each(|pressure_value, &divergence_value, &rho_value| {
                *pressure_value -= dt * rho_value * divergence_value;
            });
    }
}

pub(super) fn add_nonlinear_pressure_delta(pressure: &mut Array3<f64>, delta: &Array3<f64>) {
    assert_eq!(
        pressure.shape(),
        delta.shape(),
        "invariant: FDTD nonlinear pressure delta shape matches pressure field"
    );

    if let (Some(pressure_values), Some(delta_values)) = (
        pressure.as_slice_memory_order_mut(),
        delta.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |idx, pressure_value| {
            *pressure_value += delta_values[idx];
        });
    } else {
        Zip::from(pressure)
            .and(delta)
            .for_each(|pressure_value, &delta_value| *pressure_value += delta_value);
    }
}
