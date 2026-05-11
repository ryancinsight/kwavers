use std::collections::HashMap;

use crate::core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, AVOGADRO, DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE,
};
use crate::core::error::KwaversResult;
use crate::domain::therapy::microbubble::{MarmottantShellProperties, MicrobubbleState};
use crate::physics::acoustics::bubble_dynamics::bubble_state::{
    BubbleParameters, BubbleState, GasSpecies,
};

use super::MicrobubbleDynamicsService;

impl MicrobubbleDynamicsService {
    /// Extract Keller-Miksis parameters from microbubble state
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn extract_bubble_parameters(
        state: &MicrobubbleState,
    ) -> KwaversResult<BubbleParameters> {
        const DYNAMIC_VISCOSITY: f64 = 0.001; // Water at 37°C [Pa·s]
        const SURFACE_TENSION: f64 = 0.072; // [N/m]
        const VAPOR_PRESSURE: f64 = 3169.0; // Water at 37°C [Pa]
        const POLYTROPIC_INDEX: f64 = 1.4;
        const BODY_TEMP: f64 = 310.0; // 37°C [K]
        const THERMAL_CONDUCTIVITY: f64 = 0.6; // Water [W/(m·K)]
        const SPECIFIC_HEAT: f64 = 4186.0; // Water [J/(kg·K)]

        let mut gas_composition = HashMap::new();
        gas_composition.insert(
            crate::physics::acoustics::bubble_dynamics::bubble_state::GasType::N2,
            0.79,
        );
        gas_composition.insert(
            crate::physics::acoustics::bubble_dynamics::bubble_state::GasType::O2,
            0.21,
        );

        Ok(BubbleParameters {
            r0: state.radius_equilibrium,
            p0: ATMOSPHERIC_PRESSURE,
            rho_liquid: DENSITY_WATER_NOMINAL,
            c_liquid: SOUND_SPEED_TISSUE,
            mu_liquid: DYNAMIC_VISCOSITY,
            sigma: SURFACE_TENSION,
            pv: VAPOR_PRESSURE,
            thermal_conductivity: THERMAL_CONDUCTIVITY,
            specific_heat_liquid: SPECIFIC_HEAT,
            accommodation_coeff: 0.4,
            gas_species: GasSpecies::Air,
            initial_gas_pressure: ATMOSPHERIC_PRESSURE,
            gas_composition,
            gamma: POLYTROPIC_INDEX,
            t0: BODY_TEMP,
            driving_frequency: 1e6,
            driving_amplitude: 0.0,
            use_compressibility: true,
            use_thermal_effects: false,
            use_mass_transfer: false,
        })
    }

    /// Convert domain state to Keller-Miksis bubble state
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn domain_to_km_state(
        bubble: &MicrobubbleState,
        _shell: &MarmottantShellProperties,
    ) -> KwaversResult<BubbleState> {
        let params = Self::extract_bubble_parameters(bubble)?;
        let mut km_state = BubbleState::new(&params);

        km_state.radius = bubble.radius;
        km_state.wall_velocity = bubble.wall_velocity;
        km_state.wall_acceleration = bubble.wall_acceleration;
        km_state.temperature = bubble.temperature;
        km_state.pressure_internal = bubble.pressure_internal;
        km_state.pressure_liquid = bubble.pressure_liquid;
        km_state.n_gas = bubble.gas_moles * AVOGADRO;

        Ok(km_state)
    }

    /// Convert Keller-Miksis state back to domain state
    pub(super) fn km_to_domain_state(
        km_state: &BubbleState,
        bubble: &mut MicrobubbleState,
        shell: &MarmottantShellProperties,
    ) {
        bubble.radius = km_state.radius;
        bubble.wall_velocity = km_state.wall_velocity;
        bubble.wall_acceleration = km_state.wall_acceleration;
        bubble.temperature = km_state.temperature;
        bubble.pressure_internal = km_state.pressure_internal;
        bubble.pressure_liquid = km_state.pressure_liquid;
        bubble.surface_tension = shell.surface_tension(bubble.radius);
        bubble.shell_is_ruptured = shell.is_ruptured();
    }

    /// Calculate effective added mass for bubble translation
    ///
    /// For a sphere in incompressible fluid: m_eff = (2π/3)ρR³
    pub(super) fn effective_bubble_mass(radius: f64) -> f64 {
        (2.0 / 3.0) * std::f64::consts::PI * DENSITY_WATER_NOMINAL * radius.powi(3)
    }
}
