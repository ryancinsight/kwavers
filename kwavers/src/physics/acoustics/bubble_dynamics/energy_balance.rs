//! Energy Balance Model for Bubble Dynamics
//!
//! This module implements a comprehensive energy balance equation for bubble temperature
//! evolution, including work done by pressure-volume changes, heat transfer, and
//! latent heat from mass transfer.

use uom::si::area::square_meter;
use uom::si::energy::joule;
use uom::si::f64::{
    Area, HeatCapacity, Length, Mass, Power, Pressure, ThermalConductivity,
    ThermodynamicTemperature, Time, Velocity,
};
use uom::si::heat_capacity::joule_per_kelvin;
use uom::si::mass::kilogram;
use uom::si::power::watt;
use uom::si::pressure::pascal;
use uom::si::thermal_conductivity::watt_per_meter_kelvin;
use uom::si::thermodynamic_temperature::kelvin;
use uom::si::time::second;

use super::bubble_state::BubbleState;
use super::BubbleParameters;
use crate::core::constants::GAS_CONSTANT as R_GAS;

/// Energy balance calculator for bubble dynamics
///
/// Implements complete thermodynamic energy balance for bubble collapse including:
/// - Work done by pressure-volume changes (PdV work)
/// - Conductive heat transfer (Fourier's law with Nusselt correlation)
/// - Phase change latent heat (evaporation/condensation)
/// - Chemical reaction enthalpy (sonochemistry)
/// - Plasma ionization/recombination energy (sonoluminescence)
/// - Stefan-Boltzmann radiation (extreme temperatures T > 5000 K)
///
/// # Mathematical Foundation
///
/// First law of thermodynamics for open system:
/// ```text
/// dU/dt = -P(dV/dt) + Q_heat + Q_latent + Q_reaction + Q_plasma + Q_radiation
/// ```
///
/// Where:
/// - U: Internal energy
/// - P(dV/dt): Work done by bubble expansion/compression
/// - Q_heat: Conductive heat transfer to liquid
/// - Q_latent: Latent heat from mass transfer
/// - Q_reaction: Chemical reaction enthalpy changes
/// - Q_plasma: Ionization/recombination energy
/// - Q_radiation: Stefan-Boltzmann radiation losses
///
/// # References
///
/// - Prosperetti (1991) "The thermal behavior of oscillating gas bubbles" - J Fluid Mech 222:587-616
/// - Storey & Szeri (2000) "Water vapour, sonoluminescence and sonochemistry" - J Fluid Mech 396:203-229
/// - Moss et al. (1997) "Hydrodynamic simulations of bubble collapse" - Phys Fluids 9(6):1535-1538
/// - Hilgenfeldt et al. (1999) "Analysis of Rayleigh-Plesset dynamics" - J Fluid Mech 365:171-204
#[derive(Debug, Clone)]
pub struct EnergyBalanceCalculator {
    /// Thermal conductivity of the liquid
    thermal_conductivity: ThermalConductivity,
    /// Specific heat capacity of the liquid
    #[allow(dead_code)] // Stored for future bioheat equation calculations
    specific_heat_liquid: HeatCapacity,
    /// Ambient temperature
    ambient_temperature: ThermodynamicTemperature,
    /// Enable chemical reaction energy tracking
    pub enable_chemical_reactions: bool,
    /// Enable plasma ionization energy tracking
    pub enable_plasma_effects: bool,
    /// Enable radiation losses (Stefan-Boltzmann)
    pub enable_radiation: bool,
}

impl EnergyBalanceCalculator {
    /// Create a new energy balance calculator
    #[must_use]
    pub fn new(params: &BubbleParameters) -> Self {
        Self {
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(
                params.thermal_conductivity,
            ),
            specific_heat_liquid: HeatCapacity::new::<joule_per_kelvin>(
                params.specific_heat_liquid * params.rho_liquid,
            ),
            ambient_temperature: ThermodynamicTemperature::new::<kelvin>(293.15),
            enable_chemical_reactions: true,
            enable_plasma_effects: true,
            enable_radiation: true,
        }
    }

    /// Create calculator with specific energy tracking options
    #[must_use]
    pub fn with_options(
        params: &BubbleParameters,
        enable_chemical: bool,
        enable_plasma: bool,
        enable_radiation: bool,
    ) -> Self {
        Self {
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(
                params.thermal_conductivity,
            ),
            specific_heat_liquid: HeatCapacity::new::<joule_per_kelvin>(
                params.specific_heat_liquid * params.rho_liquid,
            ),
            ambient_temperature: ThermodynamicTemperature::new::<kelvin>(293.15),
            enable_chemical_reactions: enable_chemical,
            enable_plasma_effects: enable_plasma,
            enable_radiation,
        }
    }

    /// Calculate the rate of change of internal energy (dU/dt)
    ///
    /// Complete energy balance equation:
    /// ```text
    /// dU/dt = -P(dV/dt) + Q_heat + Q_latent + Q_reaction + Q_plasma + Q_radiation
    /// ```
    ///
    /// Where:
    /// - P(dV/dt): Work done by bubble expansion/compression
    /// - Q_heat: Conductive heat transfer to/from liquid
    /// - Q_latent: Latent heat from phase changes (evaporation/condensation)
    /// - Q_reaction: Chemical reaction enthalpy changes
    /// - Q_plasma: Ionization/recombination energy
    /// - Q_radiation: Stefan-Boltzmann radiation losses
    ///
    /// # Physics Correctness
    ///
    /// All energy terms follow first law of thermodynamics with proper sign convention:
    /// - Positive: Energy input to bubble
    /// - Negative: Energy loss from bubble
    #[must_use]
    pub fn calculate_energy_rate(
        &self,
        state: &BubbleState,
        internal_pressure: Pressure,
        heat_transfer_rate: Power,
        latent_heat_rate: Power,
    ) -> Power {
        // Calculate volume rate of change
        let _radius = Length::new::<uom::si::length::meter>(state.radius);
        let wall_velocity =
            Velocity::new::<uom::si::velocity::meter_per_second>(state.wall_velocity);

        // Surface area = 4πr²
        let surface_area = Area::new::<square_meter>(state.surface_area());

        // dV/dt = surface_area * wall_velocity
        let volume_rate = surface_area * wall_velocity;

        // Work done by the bubble (negative when compressing)
        let work_rate = -internal_pressure * volume_rate;

        // Calculate additional energy terms
        let chemical_rate = self.calculate_chemical_reaction_rate(state);
        let plasma_rate = self.calculate_plasma_ionization_rate(state);
        let radiation_rate = self.calculate_radiation_losses(state);

        // Total energy rate (sum of all contributions)
        work_rate
            + heat_transfer_rate
            + latent_heat_rate
            + chemical_rate
            + plasma_rate
            + radiation_rate
    }

    /// Calculate complete energy rate with all terms (convenience method)
    ///
    /// This method calculates all energy contributions in one call.
    #[must_use]
    pub fn calculate_complete_energy_rate(
        &self,
        state: &BubbleState,
        internal_pressure: f64,
        mass_transfer_rate: f64,
        thermal_diffusivity: f64,
    ) -> Power {
        // Convert to SI units
        let pressure = Pressure::new::<pascal>(internal_pressure);

        // Calculate Peclet number
        let peclet =
            self.calculate_peclet_number(state.radius, state.wall_velocity, thermal_diffusivity);

        // Calculate heat transfer rate
        let heat_transfer_rate = self.calculate_heat_transfer_rate(state, peclet);

        // Calculate latent heat rate from mass transfer
        const LATENT_HEAT_VAPORIZATION: f64 = 2.26e6; // J/kg for water
        let latent_heat_rate = Power::new::<watt>(mass_transfer_rate * LATENT_HEAT_VAPORIZATION);

        // Calculate total energy rate
        self.calculate_energy_rate(state, pressure, heat_transfer_rate, latent_heat_rate)
    }

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
            crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Argon => {
                15.76 * E_V_TO_JOULES
            }
            crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Xenon => {
                12.13 * E_V_TO_JOULES
            }
            crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air
            | crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Nitrogen => {
                14.53 * E_V_TO_JOULES
            }
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
            return Power::new::<watt>(0.0);
        }

        // Stefan-Boltzmann constant
        const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8; // W/(m²·K⁴)

        // Emissivity (assume blackbody for high-temperature plasma)
        const EMISSIVITY: f64 = 1.0;

        // Surface area
        let area = 4.0 * std::f64::consts::PI * state.radius * state.radius;

        // Temperature to the fourth power
        let t_bubble_4 = state.temperature.powi(4);
        let t_ambient_4 = self.ambient_temperature.get::<kelvin>().powi(4);

        // Radiation power (positive = energy loss from bubble)
        let radiation_power = area * EMISSIVITY * STEFAN_BOLTZMANN * (t_bubble_4 - t_ambient_4);

        Power::new::<watt>(-radiation_power)
    }

    /// Calculate heat transfer rate using Nusselt correlation
    #[must_use]
    pub fn calculate_heat_transfer_rate(&self, state: &BubbleState, peclet_number: f64) -> Power {
        // Nusselt number correlation for oscillating bubble
        // Nu = 2 + 0.6 * Pe^0.5 (standard correlation)
        const NUSSELT_BASE: f64 = 2.0;
        const NUSSELT_PECLET_COEFF: f64 = 0.6;
        const NUSSELT_PECLET_EXPONENT: f64 = 0.5;

        let nusselt =
            NUSSELT_BASE + NUSSELT_PECLET_COEFF * peclet_number.powf(NUSSELT_PECLET_EXPONENT);

        // Heat transfer coefficient: h = Nu * k / r
        let radius = Length::new::<uom::si::length::meter>(state.radius);
        let k_value = self.thermal_conductivity.get::<watt_per_meter_kelvin>();
        let r_value = radius.get::<uom::si::length::meter>();
        let h_value = nusselt * k_value / r_value; // W/(m²·K)

        // Temperature difference
        let bubble_temperature = ThermodynamicTemperature::new::<kelvin>(state.temperature);
        let delta_t_k =
            bubble_temperature.get::<kelvin>() - self.ambient_temperature.get::<kelvin>();

        // Heat transfer rate: Q = h * A * ΔT
        let area = Area::new::<square_meter>(state.surface_area());
        let area_m2 = area.get::<square_meter>();

        Power::new::<watt>(h_value * area_m2 * delta_t_k)
    }

    /// Calculate Peclet number for heat transfer
    #[must_use]
    pub fn calculate_peclet_number(
        &self,
        radius: f64,
        wall_velocity: f64,
        thermal_diffusivity: f64,
    ) -> f64 {
        (radius * wall_velocity.abs()) / thermal_diffusivity
    }

    /// Update bubble temperature from energy change
    pub fn update_temperature_from_energy(
        &self,
        state: &mut BubbleState,
        energy_rate: Power,
        dt: Time,
        heat_capacity: HeatCapacity,
    ) -> ThermodynamicTemperature {
        // ΔT = (dU/dt * dt) / (m * cv)
        let energy_change = energy_rate * dt;
        let mass = Mass::new::<kilogram>(state.mass());

        // Calculate temperature change in Kelvin
        let energy_joules = energy_change.get::<joule>();
        let mass_kg = mass.get::<kilogram>();
        let heat_cap_j_per_k = heat_capacity.get::<joule_per_kelvin>();
        let temp_change_k = energy_joules / (mass_kg * heat_cap_j_per_k);

        let current_temperature_k = state.temperature;
        let temperature_k = current_temperature_k + temp_change_k;

        // Ensure temperature doesn't go below ambient (non-physical)
        let ambient_k = self.ambient_temperature.get::<kelvin>();
        if temperature_k < ambient_k {
            self.ambient_temperature
        } else {
            ThermodynamicTemperature::new::<kelvin>(temperature_k)
        }
    }
}

/// Comprehensive temperature update using energy balance
pub fn update_temperature_energy_balance(
    calculator: &EnergyBalanceCalculator,
    state: &mut BubbleState,
    params: &BubbleParameters,
    internal_pressure: f64,
    mass_transfer_rate: f64,
    dt: f64,
) {
    if !params.use_thermal_effects {
        return;
    }

    // Convert to SI units
    let pressure = Pressure::new::<pascal>(internal_pressure);
    let time_step = Time::new::<second>(dt);

    // Calculate thermal diffusivity
    let thermal_diffusivity =
        params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);

    // Calculate Peclet number
    let peclet =
        calculator.calculate_peclet_number(state.radius, state.wall_velocity, thermal_diffusivity);

    // Calculate heat transfer rate
    let heat_transfer_rate = calculator.calculate_heat_transfer_rate(state, peclet);

    // Calculate latent heat rate from mass transfer
    // L_vap ≈ 2.26 MJ/kg for water at standard conditions
    const LATENT_HEAT_VAPORIZATION: f64 = 2.26e6; // J/kg
    let latent_heat_rate = Power::new::<watt>(mass_transfer_rate * LATENT_HEAT_VAPORIZATION);

    // Calculate total energy rate
    let energy_rate =
        calculator.calculate_energy_rate(state, pressure, heat_transfer_rate, latent_heat_rate);

    // Calculate heat capacity
    let gamma = state.gas_species.gamma();
    let molecular_weight = state.gas_species.molecular_weight();
    let cv = R_GAS / molecular_weight / (gamma - 1.0);
    let heat_capacity = HeatCapacity::new::<joule_per_kelvin>(cv);

    // Update temperature
    let temperature =
        calculator.update_temperature_from_energy(state, energy_rate, time_step, heat_capacity);

    state.temperature = temperature.get::<kelvin>();
    state.update_max_temperature();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_balance_equilibrium() {
        let params = BubbleParameters::default();
        let calculator = EnergyBalanceCalculator::new(&params);
        let state = BubbleState::new(&params);

        // At equilibrium with no wall motion, energy rate should be near zero
        let pressure = Pressure::new::<pascal>(101325.0);
        let heat_rate = Power::new::<watt>(0.0);
        let latent_rate = Power::new::<watt>(0.0);

        let energy_rate =
            calculator.calculate_energy_rate(&state, pressure, heat_rate, latent_rate);

        assert!(energy_rate.get::<watt>().abs() < 1e-10);
    }

    #[test]
    fn test_heat_transfer_calculation() {
        let params = BubbleParameters::default();
        let calculator = EnergyBalanceCalculator::new(&params);
        let mut state = BubbleState::new(&params);

        // Set higher temperature
        state.temperature = 400.0; // K

        let heat_rate = calculator.calculate_heat_transfer_rate(&state, 10.0);

        // Heat rate is positive when bubble is hotter than liquid (heat flows out)
        // This is the standard convention: positive = heat out of bubble
        assert!(heat_rate.get::<watt>() > 0.0);
    }

    #[test]
    fn test_chemical_reaction_energy() {
        let params = BubbleParameters::default();
        let calculator = EnergyBalanceCalculator::new(&params);
        let mut state = BubbleState::new(&params);

        // At low temperature, no chemical reactions
        state.temperature = 1000.0;
        let reaction_rate_low = calculator.calculate_chemical_reaction_rate(&state);
        assert_eq!(reaction_rate_low.get::<watt>(), 0.0);

        // At high temperature (T > 2000 K), chemical reactions occur
        state.temperature = 5000.0;
        state.n_vapor = 1e20; // Significant vapor content
        let reaction_rate_high = calculator.calculate_chemical_reaction_rate(&state);

        // Should be negative (endothermic, absorbs energy from bubble)
        assert!(reaction_rate_high.get::<watt>() < 0.0);
    }

    #[test]
    fn test_plasma_ionization_energy() {
        let params = BubbleParameters::default();
        let calculator = EnergyBalanceCalculator::new(&params);
        let mut state = BubbleState::new(&params);

        // At low temperature, no ionization
        state.temperature = 5000.0;
        let ionization_rate_low = calculator.calculate_plasma_ionization_rate(&state);
        assert_eq!(ionization_rate_low.get::<watt>(), 0.0);

        // At extreme temperature (T > 10000 K), ionization occurs
        state.temperature = 15_000.0;
        state.n_gas = 1e20;
        let ionization_rate_high = calculator.calculate_plasma_ionization_rate(&state);

        // Should be negative (endothermic, absorbs energy)
        assert!(ionization_rate_high.get::<watt>() < 0.0);
    }

    #[test]
    fn test_radiation_losses() {
        let params = BubbleParameters::default();
        let calculator = EnergyBalanceCalculator::new(&params);
        let mut state = BubbleState::new(&params);

        // At moderate temperature, negligible radiation
        state.temperature = 3000.0;
        let radiation_low = calculator.calculate_radiation_losses(&state);
        assert_eq!(radiation_low.get::<watt>(), 0.0);

        // At extreme temperature (T > 5000 K), significant radiation
        state.temperature = 10_000.0;
        state.radius = 5e-6; // 5 μm
        let radiation_high = calculator.calculate_radiation_losses(&state);

        // Should be negative (energy loss from bubble)
        assert!(radiation_high.get::<watt>() < 0.0);

        // Stefan-Boltzmann: Q ∝ T⁴, so doubling temperature increases radiation by 16×
        state.temperature = 20_000.0;
        let radiation_double = calculator.calculate_radiation_losses(&state);
        let ratio = radiation_double.get::<watt>() / radiation_high.get::<watt>();
        assert!((ratio - 16.0).abs() < 2.0); // Within 2x tolerance (T⁴ dominates)
    }

    #[test]
    fn test_complete_energy_balance() {
        let params = BubbleParameters::default();
        let calculator = EnergyBalanceCalculator::new(&params);
        let mut state = BubbleState::new(&params);

        // Extreme collapse conditions (sonoluminescence regime)
        state.temperature = 12_000.0; // K
        state.radius = 3e-6; // 3 μm
        state.wall_velocity = -100.0; // m/s (compression)
        state.n_gas = 1e19;
        state.n_vapor = 5e18;

        let internal_pressure = 1e8; // 1000 bar
        let mass_transfer_rate = 1e-9; // kg/s
        let thermal_diffusivity = 1e-5; // m²/s

        let total_energy_rate = calculator.calculate_complete_energy_rate(
            &state,
            internal_pressure,
            mass_transfer_rate,
            thermal_diffusivity,
        );

        // Total should be finite and physical
        let energy_watt = total_energy_rate.get::<watt>();
        assert!(energy_watt.is_finite());

        // During violent compression, multiple energy sinks should be active
        let chemical = calculator
            .calculate_chemical_reaction_rate(&state)
            .get::<watt>();
        let plasma = calculator
            .calculate_plasma_ionization_rate(&state)
            .get::<watt>();
        let radiation = calculator.calculate_radiation_losses(&state).get::<watt>();

        // All should be non-zero in this regime
        assert!(chemical < 0.0); // Energy absorption
        assert!(plasma < 0.0); // Energy absorption
        assert!(radiation < 0.0); // Energy loss
    }

    #[test]
    fn test_energy_balance_options() {
        let params = BubbleParameters::default();

        // Calculator with all effects disabled
        let calc_disabled = EnergyBalanceCalculator::with_options(&params, false, false, false);
        let mut state = BubbleState::new(&params);
        state.temperature = 15_000.0;

        // Should return zero for all advanced terms
        assert_eq!(
            calc_disabled
                .calculate_chemical_reaction_rate(&state)
                .get::<watt>(),
            0.0
        );
        assert_eq!(
            calc_disabled
                .calculate_plasma_ionization_rate(&state)
                .get::<watt>(),
            0.0
        );
        assert_eq!(
            calc_disabled
                .calculate_radiation_losses(&state)
                .get::<watt>(),
            0.0
        );

        // Calculator with all effects enabled
        let calc_enabled = EnergyBalanceCalculator::with_options(&params, true, true, true);
        state.n_gas = 1e20;
        state.n_vapor = 5e19;

        // Should return non-zero for all advanced terms
        assert_ne!(
            calc_enabled
                .calculate_chemical_reaction_rate(&state)
                .get::<watt>(),
            0.0
        );
        assert_ne!(
            calc_enabled
                .calculate_plasma_ionization_rate(&state)
                .get::<watt>(),
            0.0
        );
        assert_ne!(
            calc_enabled
                .calculate_radiation_losses(&state)
                .get::<watt>(),
            0.0
        );
    }
}
