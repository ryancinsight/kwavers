//! Lithotripsy physics submodules.
//!
//! This module provides the physics components for extracorporeal shock wave
//! lithotripsy (ESWL) simulation, including shock wave generation, stone
//! fracture mechanics, cavitation dynamics, and bioeffects assessment.
//!
//! ## Current Status
//!
//! This module is under active development. Submodules will be implemented
//! incrementally as part of the therapeutic ultrasound physics expansion.
//!
//! ## Planned Components
//!
//! - Shock wave generation and propagation
//! - Stone fracture mechanics
//! - Cavitation cloud dynamics
//! - Bioeffects assessment and safety monitoring

// Solver and bioeffects components remain separate implementation work:
// - Shock wave generation and nonlinear propagation (KZK/Westervelt solver integration)
// - Stone fracture mechanics: σ > σ_critical (Coleman et al. 1987 J Urol)
// - Cavitation cloud dynamics with bubble-bubble interactions
// - Bioeffects assessment and safety monitoring
//
// References for future implementation:
// - Coleman AJ et al. (1987) J Urol 137(3):504-507. (stone fragmentation)
// - Cleveland RO et al. (2007) J Acoust Soc Am 122(5):2672-2682. (lithotripsy acoustics)
// - Szabo TL (2004) Diagnostic Ultrasound Imaging. §14 (shock wave therapy)

use aequitas::systems::si::quantities::{Frequency, Pressure, Time};

/// Lithotripsy shock-wave configuration.
#[derive(Debug, Clone)]
pub struct LithotripsyConfig {
    /// Shock-wave peak pressure.
    pub peak_pressure: Pressure<f64>,
    /// Shock-wave pulse duration.
    pub pulse_duration: Time<f64>,
    /// Shock-wave repetition rate.
    pub repetition_rate: Frequency<f64>,
}

impl Default for LithotripsyConfig {
    fn default() -> Self {
        Self {
            peak_pressure: Pressure::from_base(50e6),
            pulse_duration: Time::from_base(1e-6),
            repetition_rate: Frequency::from_base(1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lithotripsy_config_default() {
        let config = LithotripsyConfig::default();
        assert!(config.peak_pressure.into_base() > 0.0);
        assert!(config.pulse_duration.into_base() > 0.0);
        assert!(config.repetition_rate.into_base() > 0.0);
    }
}
