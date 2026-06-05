//! Single Störmer-Verlet step (2nd-order symplectic) for bubble wall mechanics.

use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Single Störmer-Verlet step for bubble wall mechanics.
///
/// # Algorithm
///
/// Half-kick → full drift → half-kick (velocity Verlet):
/// ```text
/// V_{n+½} = V_n   + (h/2) f(R_n, V_n)
/// R_{n+1} = R_n   + h · V_{n+½}         [with hard collapse floor]
/// V_{n+1} = V_{n+½} + (h/2) f(R_{n+1}, V_{n+½})
/// ```
/// where f = R̈ is evaluated by `KellerMiksisModel::calculate_acceleration`.
///
/// Only the mechanical state (R, Ṙ) is updated. Thermal variables (T, n_v)
/// are deliberately left unchanged — they belong to the IMEX integrator.
///
/// # Arguments
///
/// * `state` — bubble state mutated in-place
/// * `model` — Keller-Miksis equation model (provides f = R̈)
/// * `p_acoustic` — acoustic pressure amplitude (Pa)
/// * `dp_dt` — time derivative of acoustic pressure [Pa/s]
/// * `dt` — time step (s) (may be negative for Yoshida composition)
/// * `t` — current time (s)
/// * `max_mach` — Mach number limit (error if exceeded after first half-kick)
/// * `r_min_fraction` — hard floor = r_min_fraction · params.r0
/// # Errors
/// - Returns [`KwaversError::Physics`] if the precondition for a Physics-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
#[allow(clippy::too_many_arguments)]
pub fn stormer_verlet_step(
    state: &mut BubbleState,
    model: &KellerMiksisModel,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
    max_mach: f64,
    r_min_fraction: f64,
) -> KwaversResult<()> {
    let r0 = model.params().r0;
    let c = model.params().c_liquid;
    let r_min = r_min_fraction * r0;

    // ── First half-kick: V_{n+½} = V_n + (h/2) f(R_n, V_n) ──────────────────
    let accel_0 = model.calculate_acceleration(state, p_acoustic, dp_dt, t)?;
    let v_half = (0.5 * dt).mul_add(accel_0, state.wall_velocity);

    // Mach-number guard on the half-step velocity (only for positive dt steps)
    if dt > 0.0 && v_half.abs() / c > max_mach {
        return Err(KwaversError::Physics(PhysicsError::NumericalInstability {
            timestep: dt,
            cfl_limit: v_half.abs() / c,
        }));
    }

    // ── Full drift: R_{n+1} = R_n + h · V_{n+½} ─────────────────────────────
    let r_new = dt.mul_add(v_half, state.radius);

    // Hard collapse floor: if radius would go below r_min, clamp and zero velocity.
    // This represents a fully inelastic collapse — energy is deposited at the wall.
    if r_new < r_min {
        state.radius = r_min;
        state.wall_velocity = 0.0;
        state.update_compression(r0);
        state.update_collapse_state();
        return Ok(());
    }
    state.radius = r_new;
    // Temporarily store v_half so that the second K-M call sees the correct velocity
    state.wall_velocity = v_half;
    state.update_compression(r0);

    // ── Second half-kick: V_{n+1} = V_{n+½} + (h/2) f(R_{n+1}, V_{n+½}) ────
    let t_new = t + dt;
    let accel_1 = model.calculate_acceleration(state, p_acoustic, dp_dt, t_new)?;
    state.wall_velocity = (0.5 * dt).mul_add(accel_1, v_half);

    // Update derived fields
    state.mach_number = state.wall_velocity.abs() / c;
    state.update_compression(r0);
    state.update_collapse_state();

    Ok(())
}
