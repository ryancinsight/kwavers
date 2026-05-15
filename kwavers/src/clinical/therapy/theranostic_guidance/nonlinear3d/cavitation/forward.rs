//! Rayleigh-Plesset cavitation forward map.
//!
//! Each active voxel receives the local Westervelt peak pressure as the
//! acoustic forcing amplitude in the Rayleigh-Plesset ODE. The source density
//! is the maximum period-doubled radius response normalised by the peak pressure.

use ndarray::Array3;

use crate::physics::acoustics::bubble_dynamics::{
    BubbleParameters, BubbleState, RayleighPlessetSolver,
};

use super::super::types::{Nonlinear3dConfig, Nonlinear3dVolume};

pub(super) fn cavitation_source(
    volume: &Nonlinear3dVolume,
    peak_pressure: &Array3<f64>,
    config: &Nonlinear3dConfig,
) -> Array3<f64> {
    let max_pressure = peak_pressure.iter().copied().fold(0.0, f64::max).max(1.0);
    Array3::from_shape_fn(peak_pressure.dim(), |idx| {
        if !volume.body_mask[idx] {
            return 0.0;
        }
        let pressure = peak_pressure[idx];
        let response = rayleigh_plesset_subharmonic_response(pressure, config);
        response * (pressure / max_pressure).clamp(0.0, 1.0)
    })
}

fn rayleigh_plesset_subharmonic_response(pressure_pa: f64, config: &Nonlinear3dConfig) -> f64 {
    if pressure_pa <= 0.0 {
        return 0.0;
    }
    let mut params = BubbleParameters {
        r0: config.bubble_radius_m,
        driving_frequency: config.frequency_hz,
        driving_amplitude: pressure_pa,
        use_thermal_effects: false,
        use_mass_transfer: false,
        use_compressibility: false,
        ..BubbleParameters::default()
    };
    params.initial_gas_pressure = params.p0;
    let solver = RayleighPlessetSolver::new(params.clone());
    let mut state = BubbleState::at_equilibrium(&params);
    let steps_per_period = config.bubble_time_steps_per_period;
    let total_steps = (config.cycles.ceil() as usize + 2) * steps_per_period;
    let dt = 1.0 / (config.frequency_hz * steps_per_period as f64);
    let mut radii = Vec::with_capacity(total_steps + 1);
    radii.push(state.radius);
    let mut max_subharmonic: f64 = 0.0;
    let mut max_compression: f64 = 1.0;
    for step in 0..total_steps {
        state = rk4_step(&solver, &state, pressure_pa, step as f64 * dt, dt, params.r0);
        state.update_compression(params.r0);
        max_compression = max_compression.max(state.compression_ratio);
        radii.push(state.radius);
        if radii.len() > steps_per_period {
            let previous_period = radii[radii.len() - 1 - steps_per_period];
            max_subharmonic =
                max_subharmonic.max((state.radius - previous_period).abs() / params.r0);
        }
        if state.radius <= 0.05 * params.r0 {
            max_subharmonic = max_subharmonic.max(max_compression);
            break;
        }
    }
    max_subharmonic.max((max_compression - 1.0).max(0.0))
}

fn rk4_step(
    solver: &RayleighPlessetSolver,
    state: &BubbleState,
    pressure_pa: f64,
    t: f64,
    dt: f64,
    r0: f64,
) -> BubbleState {
    let k1 = derivative(solver, state, pressure_pa, t);
    let s2 = shifted_state(state, k1, 0.5 * dt, r0);
    let k2 = derivative(solver, &s2, pressure_pa, t + 0.5 * dt);
    let s3 = shifted_state(state, k2, 0.5 * dt, r0);
    let k3 = derivative(solver, &s3, pressure_pa, t + 0.5 * dt);
    let s4 = shifted_state(state, k3, dt, r0);
    let k4 = derivative(solver, &s4, pressure_pa, t + dt);
    let mut out = state.clone();
    out.radius =
        (state.radius + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0).max(0.05 * r0);
    out.wall_velocity =
        state.wall_velocity + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
    out.wall_acceleration = solver.calculate_acceleration(&out, pressure_pa, t + dt);
    out
}

fn derivative(
    solver: &RayleighPlessetSolver,
    state: &BubbleState,
    pressure_pa: f64,
    t: f64,
) -> (f64, f64) {
    (
        state.wall_velocity,
        solver.calculate_acceleration(state, pressure_pa, t),
    )
}

fn shifted_state(state: &BubbleState, derivative: (f64, f64), dt: f64, r0: f64) -> BubbleState {
    let mut shifted = state.clone();
    shifted.radius = (state.radius + dt * derivative.0).max(0.05 * r0);
    shifted.wall_velocity = state.wall_velocity + dt * derivative.1;
    shifted
}
