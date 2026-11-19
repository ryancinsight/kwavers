//! Adaptive integration wrapper for bubble dynamics equations

use super::super::acoustic_pressure::InstantaneousPressure;
use super::super::bubble_state::BubbleState;
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::KwaversResult;

/// Integrate bubble dynamics with proper handling of stiff ODEs
///
/// This is the recommended integration method that uses adaptive time-stepping
/// with sub-cycling to handle the stiff nature of bubble dynamics equations.
///
/// `p_acoustic` is the INSTANTANEOUS acoustic pressure `p_∞(t) − p0` at
/// the step start and `dp_dt` its time derivative; both are held constant
/// across the step (zero-order hold — see the contract on
/// `AdaptiveBubbleModel::calculate_acceleration`). Keep `dt` small against
/// the acoustic period (≲ T/200).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn integrate_bubble_dynamics_stable(
    solver: &crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: InstantaneousPressure,
    dp_dt: PressureRate,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    use super::super::adaptive_integration::integrate_bubble_dynamics_adaptive;

    // Use adaptive integration with sub-cycling
    integrate_bubble_dynamics_adaptive(solver, state, p_acoustic, dp_dt, dt, t)
}
