//! IMEX (Implicit-Explicit) Time Integration for Bubble Dynamics
//!
//! ## References
//!
//! - Ascher et al. (1997). IMEX-RK schemes for stiff systems
//! - Kennedy & Carpenter (2003). ARK methods for mixed stiff/non-stiff systems
//! - Prosperetti & Lezzi (1986). Thermal effects in bubble dynamics

pub mod config;
pub mod integrator;
pub mod stiffness;
pub mod thermal_mass_transfer;

#[cfg(test)]
mod tests;

pub use config::BubbleIMEXConfig;
pub use integrator::BubbleIMEXIntegrator;

use super::{BubbleState, KellerMiksisModel};
use crate::core::error::KwaversResult;
use std::sync::Arc;

/// Main function to integrate bubble dynamics using IMEX method
pub fn integrate_bubble_dynamics_imex(
    solver: Arc<KellerMiksisModel>,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    let mut integrator = BubbleIMEXIntegrator::with_defaults(solver);
    integrator.step(state, p_acoustic, dp_dt, dt, t)?;
    Ok(())
}
