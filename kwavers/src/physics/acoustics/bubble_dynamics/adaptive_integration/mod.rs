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

use super::keller_miksis::KellerMiksisModel;
use super::BubbleState;
use crate::core::error::KwaversResult;

/// Replace the old fixed-timestep integration with adaptive version
pub fn integrate_bubble_dynamics_adaptive(
    solver: &KellerMiksisModel,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    let config = AdaptiveBubbleConfig::default();
    let mut integrator = AdaptiveBubbleIntegrator::new(solver, config);
    integrator.integrate_adaptive(state, p_acoustic, dp_dt, dt, t)
}
