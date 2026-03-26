//! Chemical reaction enthalpy effects

use uom::si::f64::Power;
use uom::si::power::watt;

use crate::core::constants::R_GAS;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::physics::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;

impl EnergyBalanceCalculator {
    /// Calculate chemical reaction energy rate
    ///
    /// Simplified model for sonochemistry reactions (H2O dissociation, OH radical formation)
    ///
    /// # Theory
    ///
    /// During extreme compression (T > 2000 K), water vapor dissociates:
    /// ```text
    /// H2O → H + OH      ΔH = +498 kJ/mol (endothermic)
    /// 2OH → H2O + O     ΔH = -70 kJ/mol (exothermic)
    /// ```
    ///
    /// Net energy absorption depends on temperature and pressure.
    ///
    /// # References
    ///
    /// - Storey & Szeri (2000) J Fluid Mech 396:203-229
    /// - Yasui (1997) Phys Rev E 56(6):6750-6760
    #[must_use]
    pub fn calculate_chemical_reaction_rate(&self, state: &BubbleState) -> Power {
        if !self.enable_chemical_reactions || state.temperature < 2000.0 {
            return Power::new::<watt>(0.0);
        }

        // Water dissociation enthalpy: ΔH_diss = 498 kJ/mol
        const H_DISSOCIATION: f64 = 498_000.0; // J/mol

        // Estimate reaction rate based on Arrhenius equation
        // k = A exp(-E_a / RT) where E_a ≈ 500 kJ/mol for H2O dissociation
        const ACTIVATION_ENERGY: f64 = 500_000.0; // J/mol
        const PRE_EXPONENTIAL: f64 = 1e13; // 1/s

        let rate_constant =
            PRE_EXPONENTIAL * (-ACTIVATION_ENERGY / (R_GAS * state.temperature)).exp();

        // Number of vapor molecules that can react
        let n_vapor_moles = state.n_vapor / crate::core::constants::AVOGADRO;

        // Reaction rate (mol/s) - limited by vapor content and kinetics
        let reaction_rate = rate_constant * n_vapor_moles * 0.01; // 1% per time constant

        // Energy rate (positive = endothermic, absorbs energy from bubble)
        Power::new::<watt>(-reaction_rate * H_DISSOCIATION)
    }
}
