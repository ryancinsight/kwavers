//! Microbubble Dynamics Application Service
//!
//! Orchestrates the simulation of therapeutic microbubble dynamics by integrating:
//! - Keller-Miksis ODE solver for bubble oscillation
//! - Marmottant shell model for lipid coating mechanics
//! - Primary Bjerknes radiation forces
//! - Drug release kinetics
//!
//! ## References
//!
//! - Clean Architecture (Martin 2017)
//! - Keller JB, Miksis M (1980). *J Acoust Soc Am* 68(2):628–633.

use crate::core::error::KwaversResult;
use crate::domain::therapy::microbubble::MicrobubbleState;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

/// Microbubble dynamics simulation service
///
/// Application service coordinating microbubble physics simulation.
#[derive(Debug)]
pub struct MicrobubbleDynamicsService {
    /// Keller-Miksis ODE solver
    pub(super) keller_miksis: KellerMiksisModel,
}

impl MicrobubbleDynamicsService {
    /// Create new microbubble dynamics service
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn new(bubble_params: BubbleParameters) -> Self {
        let keller_miksis = KellerMiksisModel::new(bubble_params);
        Self { keller_miksis }
    }

    /// Create service from microbubble state
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_microbubble_state(state: &MicrobubbleState) -> KwaversResult<Self> {
        let params = Self::extract_bubble_parameters(state)?;
        Ok(Self::new(params))
    }
}

mod dynamics;
mod sampling;
mod state;
#[cfg(test)]
mod tests;

pub use sampling::sample_acoustic_field_at_position;
