//! Plasma ionization energy rate during cavitation collapses

use uom::si::f64::Power;
use uom::si::power::watt;

use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleState, GasSpecies};
use crate::physics::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;

impl EnergyBalanceCalculator {
    /// Calculate plasma ionization energy rate
    ///
    /// During extreme compression (T > 10000 K), gas ionization occurs:
    /// ```text
    /// Ar → Ar⁺ + e⁻     E_ion = 15.76 eV
    /// ```
    ///
    /// Ionization absorbs energy, reducing peak temperature and light emission.
    ///
    /// # Saha Equation
    ///
    /// Ionization fraction at thermal equilibrium:
    /// ```text
    /// n_e n_i / n_0 = (2πm_e kT/h²)^(3/2) exp(-E_ion/kT) / n_gas
    /// ```
    ///
    /// # References
    ///
    /// - Moss et al. (1997) Phys Fluids 9(6):1535-1538
    /// - Hilgenfeldt et al. (1999) J Fluid Mech 365:171-204
    #[must_use]
    pub fn calculate_plasma_ionization_rate(&self, state: &BubbleState) -> Power {
        if !self.enable_plasma_effects || state.temperature < 10_000.0 {
            return Power::new::<watt>(0.0);
        }

        // Ionization energy for common gases (in eV, convert to J)
        const E_V_TO_JOULES: f64 = 1.602_176_634e-19;
        let ionization_energy = match state.gas_species {
            GasSpecies::Argon => 15.76 * E_V_TO_JOULES,
            GasSpecies::Xenon => 12.13 * E_V_TO_JOULES,
            GasSpecies::Air | GasSpecies::Nitrogen => 14.53 * E_V_TO_JOULES,
            _ => 15.0 * E_V_TO_JOULES, // Default
        };

        // Saha equation for ionization fraction
        // Simplified: α ≈ exp(-E_ion / kT) for low ionization
        const K_BOLTZMANN: f64 = 1.380_649e-23; // J/K
        let ionization_fraction = (-ionization_energy / (K_BOLTZMANN * state.temperature)).exp();

        // Ionization rate (molecules/s)
        let n_total = state.n_gas + state.n_vapor;
        let ionization_rate = n_total * ionization_fraction * 1e12; // Time scale ~1 ps

        // Energy rate (positive = endothermic, absorbs energy)
        Power::new::<watt>(-ionization_rate * ionization_energy)
    }
}
