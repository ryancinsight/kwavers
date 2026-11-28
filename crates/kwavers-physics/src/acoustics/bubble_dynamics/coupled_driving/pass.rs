use super::coupling::history_row;
use super::CoupledBubble;
use crate::acoustics::bubble_dynamics::acoustic_pressure::InstantaneousPressure;
use crate::acoustics::bubble_dynamics::adaptive_integration::AdaptiveBubbleModel;
use crate::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::KwaversResult;
use leto::Array2;
use std::f64::consts::PI;

/// Linear interpolation of a grid row at fractional index `x`.
fn interp_grid(row: &[f64], x: f64) -> f64 {
    if row.is_empty() {
        return 0.0;
    }
    let i0 = x.floor().max(0.0) as usize;
    if i0 >= row.len() - 1 {
        return row[row.len() - 1];
    }
    let frac = x - i0 as f64;
    row[i0] * (1.0 - frac) + row[i0 + 1] * frac
}

/// One bubble's KM acceleration at a trial state under the grid
/// drive, through the existing model (no KM duplication).
fn accel<M: AdaptiveBubbleModel>(
    model: &M,
    params: &BubbleParameters,
    radius: f64,
    velocity: f64,
    p_ac: f64,
    dp_dt: f64,
    t: f64,
) -> KwaversResult<f64> {
    let mut state = BubbleState::new(params);
    state.radius = radius;
    state.wall_velocity = velocity;
    model.calculate_acceleration(
        &mut state,
        InstantaneousPressure::from_pascals(p_ac),
        PressureRate::from_base(dp_dt),
        t,
    )
}

/// Per-bubble histories produced by one Picard pass, one row per
/// bubble on the integration grid.
#[derive(Debug)]
pub(super) struct PassHistories {
    /// Radius [m].
    pub(super) radius: Array2<f64>,
    /// Wall velocity [m/s].
    pub(super) velocity: Array2<f64>,
    /// Volume acceleration `V̈` [m³/s²].
    pub(super) volume_accel: Array2<f64>,
}

/// Integrate one Picard pass for the whole cloud under the given
/// per-bubble total-drive grids. RK4 in the Python reference's
/// stage arrangement.
pub(super) fn integrate_pass<M: AdaptiveBubbleModel>(
    models: &[M],
    bubbles: &[CoupledBubble],
    drive_p: &Array2<f64>,
    drive_dp: &Array2<f64>,
    dt: f64,
) -> KwaversResult<PassHistories> {
    let nb = bubbles.len();
    let nsteps = drive_p.shape()[1] - 1;
    let mut radius = Array2::zeros([nb, nsteps + 1]);
    let mut velocity = Array2::zeros([nb, nsteps + 1]);
    for (i, bubble) in bubbles.iter().enumerate() {
        let model = &models[i];
        let params = model.params();
        let (p_row, dp_row) = (history_row(drive_p, i), history_row(drive_dp, i));
        radius[[i, 0]] = bubble.radius0_m;
        for k in 0..nsteps {
            let rk = radius[[i, k]];
            let vk = velocity[[i, k]];
            let drive = |x: f64| (interp_grid(p_row, x), interp_grid(dp_row, x));
            let (p, dp) = drive(k as f64);
            let a1 = accel(model, params, rk, vk, p, dp, k as f64 * dt)?;
            let (p, dp) = drive(k as f64 + 0.5);
            let a2 = accel(
                model,
                params,
                rk + 0.5 * dt * vk,
                vk + 0.5 * dt * a1,
                p,
                dp,
                (k as f64 + 0.5) * dt,
            )?;
            let a3 = accel(
                model,
                params,
                rk + 0.5 * dt * (vk + 0.5 * dt * a1),
                vk + 0.5 * dt * a2,
                p,
                dp,
                (k as f64 + 0.5) * dt,
            )?;
            let (p, dp) = drive((k + 1) as f64);
            let a4 = accel(
                model,
                params,
                rk + dt * (vk + 0.5 * dt * a2),
                vk + dt * a3,
                p,
                dp,
                (k + 1) as f64 * dt,
            )?;
            velocity[[i, k + 1]] = vk + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
            radius[[i, k + 1]] = rk + dt * vk + dt * dt / 6.0 * (a1 + a2 + a3);
        }
    }
    // Volume acceleration on the grid from the grid drive:
    // V̈ = 4π(2RV̇² + R²R̈).
    let mut volume_accel = Array2::zeros([nb, nsteps + 1]);
    for i in 0..nb {
        let model = &models[i];
        let params = model.params();
        for k in 0..=nsteps {
            let a = accel(
                model,
                params,
                radius[[i, k]],
                velocity[[i, k]],
                drive_p[[i, k]],
                drive_dp[[i, k]],
                k as f64 * dt,
            )?;
            let r = radius[[i, k]];
            let v = velocity[[i, k]];
            volume_accel[[i, k]] = 4.0 * PI * (2.0 * r * v * v + r * r * a);
        }
    }
    Ok(PassHistories {
        radius,
        velocity,
        volume_accel,
    })
}
