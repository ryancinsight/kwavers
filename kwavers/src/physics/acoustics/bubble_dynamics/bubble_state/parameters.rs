use super::gas_dynamics::{GasSpecies, GasType};
use crate::core::constants::thermodynamic::T_AMBIENT;
use std::collections::HashMap;

/// Physical parameters for bubble dynamics
#[derive(Debug, Clone)]
pub struct BubbleParameters {
    // Equilibrium properties
    pub r0: f64, // Equilibrium radius [m]
    pub p0: f64, // Ambient pressure [Pa]

    // Liquid properties
    pub rho_liquid: f64, // Liquid density [kg/m³]
    pub c_liquid: f64,   // Sound speed in liquid [m/s]
    pub mu_liquid: f64,  // Dynamic viscosity [Pa·s]
    pub sigma: f64,      // Surface tension [N/m]
    pub pv: f64,         // Vapor pressure [Pa]

    // Thermal properties
    pub thermal_conductivity: f64, // k [W/(m·K)]
    pub specific_heat_liquid: f64, // cp [J/(kg·K)]
    pub accommodation_coeff: f64,  // Thermal accommodation

    // Gas properties
    pub gas_species: GasSpecies,
    pub initial_gas_pressure: f64, // Initial gas pressure [Pa]
    /// Gas composition: maps gas type to mole fraction
    /// Default is air (79% N2, 21% O2)
    pub gas_composition: HashMap<GasType, f64>,
    pub gamma: f64, // Adiabatic index (ratio of specific heats)
    pub t0: f64,    // Ambient temperature [K]

    // Acoustic forcing parameters
    pub driving_frequency: f64, // Driving frequency [Hz]
    pub driving_amplitude: f64, // Pressure amplitude [Pa]

    // Numerical parameters
    pub use_compressibility: bool, // Use Keller-Miksis
    pub use_thermal_effects: bool, // Include heat transfer
    pub use_mass_transfer: bool,   // Include evaporation/condensation
}

impl Default for BubbleParameters {
    fn default() -> Self {
        // Default air composition
        let mut gas_composition = HashMap::new();
        gas_composition.insert(GasType::N2, 0.79);
        gas_composition.insert(GasType::O2, 0.21);

        Self {
            // Water at 20°C with 5 μm air bubble
            r0: 5e-6,
            p0: 101325.0,
            rho_liquid: 998.0,
            c_liquid: 1482.0,
            gamma: 1.4, // Air adiabatic index
            t0: T_AMBIENT, // 20°C in Kelvin (293.15 K)
            mu_liquid: 1.002e-3,
            sigma: 0.0728,
            pv: 2.33e3,
            thermal_conductivity: 0.6,
            specific_heat_liquid: 4182.0,
            accommodation_coeff: 0.04,
            gas_species: GasSpecies::Air,
            initial_gas_pressure: 101325.0,
            gas_composition,
            driving_frequency: 26.5e3, // 26.5 kHz (typical medical ultrasound)
            driving_amplitude: 1e5,    // 100 kPa acoustic pressure
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }
}

impl BubbleParameters {
    /// Create parameters for pure gas bubble
    #[must_use]
    pub fn with_pure_gas(mut self, gas_type: GasType) -> Self {
        self.gas_composition.clear();
        self.gas_composition.insert(gas_type, 1.0);
        self
    }

    /// Calculate effective Van der Waals constants for gas mixture
    #[must_use]
    pub fn effective_vdw_constants(&self) -> (f64, f64) {
        let mut a_mix = 0.0;
        let mut b_mix = 0.0;

        // Use mixing rules for Van der Waals constants
        // a_mix = (Σ x_i * sqrt(a_i))^2 (geometric mean for a)
        // b_mix = Σ x_i * b_i (arithmetic mean for b)
        for (gas, &fraction) in &self.gas_composition {
            a_mix += fraction * gas.vdw_a().sqrt();
            b_mix += fraction * gas.vdw_b();
        }
        a_mix = a_mix.powi(2);

        (a_mix, b_mix)
    }
}
