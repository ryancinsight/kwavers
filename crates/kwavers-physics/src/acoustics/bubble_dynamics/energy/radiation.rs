//! Stefan-Boltzmann radiation losses

use aequitas::systems::si::{
    quantities::Power,
    units::{Kelvin, Watt},
};

use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;
use kwavers_core::constants::fundamental::STEFAN_BOLTZMANN;
use kwavers_core::constants::numerical::FOUR_PI;

impl EnergyBalanceCalculator {
    /// Calculate Stefan-Boltzmann radiation losses
    ///
    /// At extreme temperatures (T > 5000 K), thermal radiation becomes significant:
    /// ```text
    /// Q_rad = 4πR² ε σ (T⁴ - T_∞⁴)
    /// ```
    ///
    /// Where:
    /// - ε: Emissivity (≈ 1 for blackbody)
    /// - σ: Stefan-Boltzmann constant = 5.67×10⁻⁸ W/(m²·K⁴)
    /// - R: Bubble radius
    /// - T: Bubble temperature
    /// - T_∞: Ambient temperature
    ///
    /// # References
    ///
    /// - Prosperetti (1991) J Fluid Mech 222:587-616
    /// - Hilgenfeldt et al. (1999) J Fluid Mech 365:171-204
    #[must_use]
    pub fn calculate_radiation_losses(&self, state: &BubbleState) -> Power {
        if !self.enable_radiation || state.temperature < 5000.0 {
            return Power::from_unit::<Watt>(0.0);
        }

        // Emissivity (assume blackbody for high-temperature plasma)
        const EMISSIVITY: f64 = 1.0;

        // Surface area
        let area = FOUR_PI * state.radius * state.radius;

        // Temperature to the fourth power
        let t_bubble_4 = state.temperature.powi(4);
        let t_ambient_4 = self.ambient_temperature.in_unit::<Kelvin>().powi(4);

        // Radiation power (positive = energy loss from bubble)
        let radiation_power = area * EMISSIVITY * STEFAN_BOLTZMANN * (t_bubble_4 - t_ambient_4);

        Power::from_unit::<Watt>(-radiation_power)
    }
}
