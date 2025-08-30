//! Energy Balance Model for Bubble Dynamics
//!
//! This module implements a comprehensive energy balance equation for bubble temperature
//! evolution, including work done by pressure-volume changes, heat transfer, and
//! latent heat from mass transfer.

use uom::si::area::square_meter;
use uom::si::energy::joule;
use uom::si::f64::*;
use uom::si::heat_capacity::joule_per_kelvin;
use uom::si::mass::kilogram;
use uom::si::power::watt;
use uom::si::pressure::pascal;
use uom::si::thermal_conductivity::watt_per_meter_kelvin;
use uom::si::thermodynamic_temperature::kelvin;
use uom::si::time::second;

use super::bubble_state::BubbleState;
use super::BubbleParameters;
use crate::constants::thermodynamics::R_GAS;

/// Energy balance calculator for bubble dynamics
#[derive(Clone, Debug)]
pub struct EnergyBalanceCalculator {
    /// Thermal conductivity of the liquid
    thermal_conductivity: ThermalConductivity,
    /// Specific heat capacity of the liquid
    specific_heat_liquid: HeatCapacity,
    /// Ambient temperature
    ambient_temperature: ThermodynamicTemperature,
}

impl EnergyBalanceCalculator {
    /// Create a new energy balance calculator
    pub fn new(params: &BubbleParameters) -> Self {
        Self {
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(
                params.thermal_conductivity,
            ),
            specific_heat_liquid: HeatCapacity::new::<joule_per_kelvin>(
                params.specific_heat_liquid * params.rho_liquid,
            ),
            ambient_temperature: ThermodynamicTemperature::new::<kelvin>(293.15),
        }
    }

    /// Calculate the rate of change of internal energy (dU/dt)
    ///
    /// The energy balance equation is:
    /// dU/dt = -P dV/dt + Q_heat + Q_latent
    ///
    /// Where:
    /// - P dV/dt is the work done by the bubble
    /// - Q_heat is heat transfer to/from the liquid
    /// - Q_latent is latent heat from mass transfer
    pub fn calculate_energy_rate(
        &self,
        state: &BubbleState,
        internal_pressure: Pressure,
        heat_transfer_rate: Power,
        latent_heat_rate: Power,
    ) -> Power {
        // Calculate volume rate of change
        let radius = Length::new::<uom::si::length::meter>(state.radius);
        let wall_velocity =
            Velocity::new::<uom::si::velocity::meter_per_second>(state.wall_velocity);

        // Surface area = 4πr²
        let surface_area = Area::new::<square_meter>(state.surface_area());

        // dV/dt = surface_area * wall_velocity
        let volume_rate = surface_area * wall_velocity;

        // Work done by the bubble (negative when compressing)
        let work_rate = -internal_pressure * volume_rate;

        // Total energy rate
        work_rate + heat_transfer_rate + latent_heat_rate
    }

    /// Calculate heat transfer rate using Nusselt correlation
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
}
