use anyhow::Result;
use kwavers_solver::forward::viscoacoustic::ViscoacousticMemorySolver;
use leto::Array3;

use super::configuration::{
    excitation, ABSORBER_CELLS, ABSORBER_GAMMA, C0, DX, F_MAX, F_MIN, F_REF, N, N_ARMS, RHO,
    SENSOR_FAR, SENSOR_NEAR, SOURCE_INDEX, STEPS,
};

/// Propagate a broadband pulse and return the two sensor traces.
pub(crate) fn run_pulse(
    alpha_field: &Array3<f64>,
    gamma_field: &Array3<f64>,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let rho_field = Array3::from_elem([N, 1, 1], RHO);
    let c_field = Array3::from_elem([N, 1, 1], C0);
    let mut solver = ViscoacousticMemorySolver::from_power_law_fields(
        N,
        1,
        1,
        DX,
        1.0,
        1.0,
        dt,
        &rho_field,
        &c_field,
        alpha_field,
        gamma_field,
        F_MIN,
        F_MAX,
        N_ARMS,
        F_REF,
    )?;
    solver.enable_absorbing_layer(ABSORBER_CELLS, ABSORBER_GAMMA);
    solver.add_pressure_source((SOURCE_INDEX, 0, 0), excitation(dt))?;
    let near = solver.add_pressure_sensor((SENSOR_NEAR, 0, 0))?;
    let far = solver.add_pressure_sensor((SENSOR_FAR, 0, 0))?;

    for _ in 0..STEPS {
        solver.step();
    }
    Ok((
        solver.sensor_trace(near).to_vec(),
        solver.sensor_trace(far).to_vec(),
    ))
}
