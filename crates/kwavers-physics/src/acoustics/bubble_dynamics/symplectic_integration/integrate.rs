//! Time-span integration wrapper using Störmer-Verlet steps.

use super::{stormer_verlet_step, SymplecticConfig};
use crate::acoustics::bubble_dynamics::{
    acoustic_pressure::InstantaneousPressure, bubble_state::BubbleState,
    keller_miksis::KellerMiksisModel,
};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Integrate bubble dynamics over a time span using Störmer-Verlet.
///
/// This is a convenience wrapper around repeated `stormer_verlet_step` calls
/// suitable for use from the physics layer (`integration.rs`).
///
/// # Arguments
///
/// * `initial_state` — initial bubble state
/// * `model` — Keller-Miksis model
/// * `time_span` — (t_start, t_end)
/// * `dt` — time step (s)
/// * `p_acoustic` — constant INSTANTANEOUS acoustic pressure `p_∞ − p0` (Pa),
///   held over the span (a step input)
/// * `dp_dt` — constant time derivative of the instantaneous acoustic
///   pressure [Pa/s]
/// # Errors
/// - Returns [`KwaversError::Physics`] if the precondition for a Physics-class constraint is violated.
/// - Propagates any `KwaversError` returned by called functions.
///
pub fn integrate_bubble_dynamics_symplectic(
    initial_state: BubbleState,
    model: &KellerMiksisModel,
    time_span: (f64, f64),
    dt: f64,
    p_acoustic: InstantaneousPressure,
    dp_dt: PressureRate,
) -> KwaversResult<BubbleState> {
    if dt <= 0.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "dt".to_owned(),
            value: dt,
            reason: "Time step must be positive".to_owned(),
        }));
    }
    if time_span.1 <= time_span.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "time_span.1".to_owned(),
            value: time_span.1,
            reason: "End time must be greater than start time".to_owned(),
        }));
    }

    let config = SymplecticConfig::default();
    let mut state = initial_state;
    let mut t = time_span.0;

    while t < time_span.1 {
        let h = dt.min(time_span.1 - t);
        stormer_verlet_step(
            &mut state,
            model,
            p_acoustic,
            dp_dt,
            h,
            t,
            config.max_mach,
            config.r_min_fraction,
        )?;
        t += h;
    }

    Ok(state)
}
