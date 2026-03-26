//! Adaptive integration wrapper for bubble dynamics equations

use super::super::bubble_state::BubbleState;
use crate::core::error::KwaversResult;

/// Integrate bubble dynamics with proper handling of stiff ODEs
///
/// This is the recommended integration method that uses adaptive time-stepping
/// with sub-cycling to handle the stiff nature of bubble dynamics equations.
pub fn integrate_bubble_dynamics_stable(
    solver: &crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    use super::super::adaptive_integration::integrate_bubble_dynamics_adaptive;

    // Use adaptive integration with sub-cycling
    integrate_bubble_dynamics_adaptive(solver, state, p_acoustic, dp_dt, dt, t)
}
