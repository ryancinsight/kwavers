//! Microbubble radial dynamics for contrast-enhanced ultrasound.

mod integration;
mod scattering;

#[cfg(test)]
mod tests;

use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use kwavers_imaging::ultrasound::ceus::Microbubble;

/// Linearized total damping constant for micron-scale bubbles at resonance.
///
/// Prosperetti (1977) reports that air bubbles in water at 20 C with radii in
/// the 1-10 um CEUS range have combined damping near 0.1 around resonance.
pub(crate) const PROSPERETTI_TOTAL_DAMPING_COEFFICIENT: f64 = 0.1;

/// Microbubble dynamics simulator.
#[derive(Debug, Clone, Copy)]
pub struct BubbleDynamics {
    /// Time step for integration (s).
    pub(crate) dt: f64,
    /// Ambient pressure (Pa).
    pub(crate) ambient_pressure: f64,
    /// Liquid density [kg/m^3].
    pub(crate) liquid_density: f64,
    /// Dimensionless damping coefficient.
    pub(crate) damping_coefficient: f64,
}

impl BubbleDynamics {
    /// Create a microbubble dynamics simulator with CEUS-scale defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dt: 1e-9,
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            liquid_density: DENSITY_WATER_NOMINAL,
            damping_coefficient: PROSPERETTI_TOTAL_DAMPING_COEFFICIENT,
        }
    }

    pub(super) fn equilibrium_gas_pressure(&self, bubble: &Microbubble, r0: f64) -> f64 {
        self.ambient_pressure
            + 2.0 * bubble.surface_tension / r0
            + 4.0 * bubble.shell_elasticity * bubble.shell_thickness / r0
    }
}

impl Default for BubbleDynamics {
    fn default() -> Self {
        Self::new()
    }
}
