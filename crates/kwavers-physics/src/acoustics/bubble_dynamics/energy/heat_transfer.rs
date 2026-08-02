//! Fourier heat transfer integration and coordinate energy balances

use aequitas::systems::si::{
    quantities::{
        Area, Length, Mass, Power, Pressure, SpecificHeatCapacity, TemperatureDifference,
        ThermodynamicTemperature, Time, Velocity,
    },
    units::{
        JoulePerKilogramKelvin, Kelvin, Kilogram, Meter, MeterPerSecond, Pascal, Second,
        SquareMeter, Watt,
    },
};

use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;
use crate::acoustics::bubble_dynamics::BubbleParameters;
use kwavers_core::constants::thermodynamic::H_VAP_WATER_100C;
use kwavers_core::constants::GAS_CONSTANT as R_GAS;

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
        let wall_velocity = Velocity::from_unit::<MeterPerSecond>(state.wall_velocity);

        // Surface area = 4πr²
        let surface_area = Area::from_unit::<SquareMeter>(state.surface_area());

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
        let pressure = Pressure::from_unit::<Pascal>(internal_pressure);

        // Calculate Peclet number
        let peclet =
            self.calculate_peclet_number(state.radius, state.wall_velocity, thermal_diffusivity);

        // Calculate heat transfer rate
        let heat_transfer_rate = self.calculate_heat_transfer_rate(state, peclet);

        // Calculate latent heat rate from mass transfer (H_VAP_WATER_100C = 2.257 MJ/kg at 100°C)
        let latent_heat_rate = Power::from_unit::<Watt>(mass_transfer_rate * H_VAP_WATER_100C);

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

        let nusselt = NUSSELT_PECLET_COEFF.mul_add(peclet_number.sqrt(), NUSSELT_BASE);

        // Heat transfer coefficient: h = Nu * k / r
        let radius = Length::from_unit::<Meter>(state.radius);
        let heat_transfer_coefficient = (self.thermal_conductivity / radius) * nusselt;

        // Temperature difference
        let bubble_temperature = ThermodynamicTemperature::from_unit::<Kelvin>(state.temperature);
        let delta_temperature = bubble_temperature - self.ambient_temperature;

        // Heat transfer rate: Q = h * A * ΔT
        let area = Area::from_unit::<SquareMeter>(state.surface_area());
        heat_transfer_coefficient * area * delta_temperature
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
        specific_heat_capacity: SpecificHeatCapacity,
    ) -> ThermodynamicTemperature {
        // ΔT = (dU/dt * dt) / (m * cv)
        let energy_change = energy_rate * dt;
        let mass = Mass::from_unit::<Kilogram>(state.mass());
        let temperature_change = TemperatureDifference::from_base(
            (energy_change / (mass * specific_heat_capacity)).into_base(),
        );
        let current_temperature = ThermodynamicTemperature::from_unit::<Kelvin>(state.temperature);
        let temperature = current_temperature + temperature_change;

        // Ensure temperature doesn't go below ambient (non-physical)
        if temperature < self.ambient_temperature {
            self.ambient_temperature
        } else {
            temperature
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
    let pressure = Pressure::from_unit::<Pascal>(internal_pressure);
    let time_step = Time::from_unit::<Second>(dt);

    // Calculate thermal diffusivity
    let thermal_diffusivity =
        params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);

    // Calculate Peclet number
    let peclet =
        calculator.calculate_peclet_number(state.radius, state.wall_velocity, thermal_diffusivity);

    // Calculate heat transfer rate
    let heat_transfer_rate = calculator.calculate_heat_transfer_rate(state, peclet);

    // Calculate latent heat rate from mass transfer (H_VAP_WATER_100C = 2.257 MJ/kg at 100°C)
    let latent_heat_rate = Power::from_unit::<Watt>(mass_transfer_rate * H_VAP_WATER_100C);

    // Calculate total energy rate
    let energy_rate =
        calculator.calculate_energy_rate(state, pressure, heat_transfer_rate, latent_heat_rate);

    // Calculate heat capacity
    let gamma = state.gas_species.gamma();
    let molecular_weight = state.gas_species.molecular_weight();
    let cv = R_GAS / molecular_weight / (gamma - 1.0);
    let specific_heat_capacity = SpecificHeatCapacity::from_unit::<JoulePerKilogramKelvin>(cv);

    // Update temperature
    let temperature = calculator.update_temperature_from_energy(
        state,
        energy_rate,
        time_step,
        specific_heat_capacity,
    );

    state.temperature = temperature.in_unit::<Kelvin>();
    state.update_max_temperature();
}
