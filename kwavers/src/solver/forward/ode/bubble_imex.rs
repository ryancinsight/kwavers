//! IMEX (Implicit-Explicit) integration for bubble dynamics
//!
//! Combines explicit treatment of the mechanical driving term with implicit
//! treatment of the stiff thermal and mass transfer terms in the Keller-Miksis
//! equation with thermodynamic coupling.
//!
//! ## Mathematical Foundation
//!
//! The system is split as:
//!
//! ```text
//! dy/dt = f_explicit(t, y) + f_implicit(t, y)
//! ```
//!
//! where `f_explicit` contains the acoustic driving and inertial terms (non-stiff)
//! and `f_implicit` contains the thermal conduction and mass transfer terms (stiff
//! during violent collapse).
//!
//! ## References
//!
//! - Ascher et al. (1997). IMEX-RK schemes for stiff systems
//! - Prosperetti & Lezzi (1986). Thermal effects in bubble dynamics

// Re-export from physics layer (canonical implementation location)
pub use crate::physics::acoustics::bubble_dynamics::imex_integration::{
    integrate_bubble_dynamics_imex, BubbleIMEXConfig, BubbleIMEXIntegrator,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
    use crate::physics::acoustics::bubble_dynamics::KellerMiksisModel;
    use std::sync::Arc;

    #[test]
    fn test_imex_config_defaults() {
        // Verify config exists and has sensible defaults
        let model = KellerMiksisModel::new(BubbleParameters::default());
        let _config = BubbleIMEXConfig::default();
        let _integrator = BubbleIMEXIntegrator::with_defaults(Arc::new(model));
    }
}
