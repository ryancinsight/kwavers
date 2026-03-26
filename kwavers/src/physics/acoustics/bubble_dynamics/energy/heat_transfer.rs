//! Fourier heat transfer integration and coordinate energy balances

use uom::si::area::square_meter;
use uom::si::energy::joule;
use uom::si::f64::{
    Area, HeatCapacity, Length, Mass, Power, Pressure, ThermodynamicTemperature, Time, Velocity,
};
use uom::si::heat_capacity::joule_per_kelvin;
use uom::si::mass::kilogram;
use uom::si::power::watt;
use uom::si::pressure::pascal;
use uom::si::thermal_conductivity::watt_per_meter_kelvin;
use uom::si::thermodynamic_temperature::kelvin;
use uom::si::time::second;

use crate::core::constants::GAS_CONSTANT as R_GAS;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::physics::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;
use crate::physics::acoustics::bubble_dynamics::BubbleParameters;

impl EnergyBalanceCalculator {
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
