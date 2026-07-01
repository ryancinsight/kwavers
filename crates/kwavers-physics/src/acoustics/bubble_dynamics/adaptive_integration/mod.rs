//! Adaptive time-stepping for bubble dynamics
//!
//! This module provides adaptive integration methods specifically designed
//! for the stiff ODEs in bubble dynamics, particularly during violent collapse.
//!
//! ## Literature References
//!
//! 1. **Hairer & Wanner (1996)**. "Solving Ordinary Differential Equations II:
//!    Stiff and Differential-Algebraic Problems"
//!    - Adaptive time-stepping strategies for stiff ODEs
//!
//! 2. **Storey & Szeri (2000)**. "Water vapour, sonoluminescence and sonochemistry"
//!    - Time scales in bubble dynamics
//!
//! 3. **Lauterborn & Kurz (2010)**. "Physics of bubble oscillations"
//!    - Numerical challenges in bubble dynamics

mod config;
mod integrator;
mod statistics;

#[cfg(test)]
mod tests;

pub use config::AdaptiveBubbleConfig;
pub use integrator::AdaptiveBubbleIntegrator;
pub use statistics::IntegrationStatistics;

use super::{bubble_state::BubbleParameters, BubbleState};
use kwavers_core::error::KwaversResult;

/// Contract implemented by bubble-equation models used by the adaptive RK4
/// engine in this module.
pub trait AdaptiveBubbleModel {
    /// Shared model parameters (sound speed, density, equilibrium radius, etc.).
    fn params(&self) -> &BubbleParameters;

    /// Bubble-wall acceleration R̈ at the current state.
    fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        t: f64,
    ) -> KwaversResult<f64>;

    /// Optional temperature update hook. A no-op implementation is valid when
    /// thermal effects are disabled for the selected model.
    fn update_temperature(&self, state: &mut BubbleState, dt: f64) -> KwaversResult<()>;

    /// Optional vapor transfer update hook. A no-op implementation is valid when
    /// mass transfer is disabled for the selected model.
    fn update_mass_transfer(&self, state: &mut BubbleState, dt: f64) -> KwaversResult<()>;
}

/// Replace the old fixed-timestep integration with adaptive version
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn integrate_bubble_dynamics_adaptive<Model>(
    solver: &Model,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()>
where
    Model: AdaptiveBubbleModel,
{
    let config = AdaptiveBubbleConfig::default();
    let mut integrator = AdaptiveBubbleIntegrator::new(solver, config);
    integrator.integrate_adaptive(state, p_acoustic, dp_dt, dt, t)
}
