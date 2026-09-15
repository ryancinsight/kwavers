//! FDTD pressure field update — SRP extraction from solver.rs.
//!
//! Pressure-related `impl FdtdSolver` extension blocks:
//! - `update`: dispatch, CPU, SIMD, GPU paths
//! - `nonlinear`: Westervelt correction and history rotation
//! - `divergence`: staggered-grid velocity divergence

use leto::Array3 as LetoArray3;
use leto::{Array3, ArrayView3};

use crate::forward::lanes::for_each_z_lane;

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

    let [_, ny, nz] = target.shape();
    if let (Some(target_values), Some(x_values), Some(y_values)) =
        (target.as_slice_mut(), x.as_slice(), y.as_slice())
    {
        for_each_z_lane(
            target_values,
            [ny, nz],
            3 * size_of::<f64>(),
            |start, _, _, lane| {
                let inputs = x_values[start..start + nz]
                    .iter()
                    .zip(&y_values[start..start + nz]);
                for (target_value, (&x_value, &y_value)) in lane.iter_mut().zip(inputs) {
                    *target_value += x_value + y_value;
                }
            },
        );
    } else {
        leto_ops::zip_mut_with(
            target.view_mut(),
            (&x.view(), &y.view()),
            |target_value, (x_value, y_value)| *target_value += *x_value + *y_value,
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

    let [_, ny, nz] = pressure.shape();
    if let (Some(pressure_values), Some(divergence_values), Some(rho_values)) = (
        pressure.as_slice_mut(),
        divergence.as_slice(),
        rho_c_squared.as_slice(),
    ) {
        for_each_z_lane(
            pressure_values,
            [ny, nz],
            3 * size_of::<f64>(),
            |start, _, _, lane| {
                let inputs = rho_values[start..start + nz]
                    .iter()
                    .zip(&divergence_values[start..start + nz]);
                for (pressure_value, (&rho_value, &divergence_value)) in lane.iter_mut().zip(inputs)
                {
                    *pressure_value -= dt * rho_value * divergence_value;
                }
            },
        );
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

/// Pressure update with relaxation absorption:
/// `p -= dt * (M_U * div(v) + relaxation)`.
///
/// The lossless form multiplies the divergence by `rho_0*c_0^2`; here the
/// coefficient is the **unrelaxed** modulus `M_U`, which is stiffer, and the
/// memory-variable term is subtracted alongside it. Using the relaxed modulus
/// would run the medium at its low-frequency speed while the arms also supply
/// dispersion -- the error would look like a wrong sound speed, not a wrong
/// absorption.
pub(super) fn apply_absorbing_pressure_update(
    pressure: &mut LetoArray3<f64>,
    divergence: ArrayView3<'_, f64>,
    unrelaxed_modulus: &Array3<f64>,
    relaxation: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        pressure.shape(),
        divergence.shape(),
        "invariant: FDTD absorbing divergence shape matches pressure field"
    );
    assert_eq!(
        pressure.shape(),
        unrelaxed_modulus.shape(),
        "invariant: FDTD unrelaxed modulus shape matches pressure field"
    );
    assert_eq!(
        pressure.shape(),
        relaxation.shape(),
        "invariant: FDTD relaxation term shape matches pressure field"
    );

    let [_, ny, nz] = pressure.shape();
    if let (
        Some(pressure_values),
        Some(divergence_values),
        Some(modulus_values),
        Some(relax_values),
    ) = (
        pressure.as_slice_mut(),
        divergence.as_slice(),
        unrelaxed_modulus.as_slice(),
        relaxation.as_slice(),
    ) {
        for_each_z_lane(
            pressure_values,
            [ny, nz],
            4 * size_of::<f64>(),
            |start, _, _, lane| {
                let inputs = modulus_values[start..start + nz]
                    .iter()
                    .zip(&divergence_values[start..start + nz])
                    .zip(&relax_values[start..start + nz]);
                for (pressure_value, ((&modulus_value, &divergence_value), &relax_value)) in
                    lane.iter_mut().zip(inputs)
                {
                    *pressure_value -= dt * modulus_value.mul_add(divergence_value, relax_value);
                }
            },
        );
    } else {
        for (((pressure_value, &divergence_value), &modulus_value), &relax_value) in pressure
            .iter_mut()
            .zip(divergence.iter())
            .zip(unrelaxed_modulus.iter())
            .zip(relaxation.iter())
        {
            *pressure_value -= dt * modulus_value.mul_add(divergence_value, relax_value);
        }
    }
}

pub(super) fn add_nonlinear_pressure_delta(pressure: &mut LetoArray3<f64>, delta: &Array3<f64>) {
    assert_eq!(
        pressure.shape(),
        delta.shape(),
        "invariant: FDTD nonlinear pressure delta shape matches pressure field"
    );

    let [_, ny, nz] = pressure.shape();
    if let (Some(pressure_values), Some(delta_values)) = (pressure.as_slice_mut(), delta.as_slice())
    {
        for_each_z_lane(
            pressure_values,
            [ny, nz],
            2 * size_of::<f64>(),
            |start, _, _, lane| {
                for (pressure_value, &delta_value) in
                    lane.iter_mut().zip(&delta_values[start..start + nz])
                {
                    *pressure_value += delta_value;
                }
            },
        );
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
