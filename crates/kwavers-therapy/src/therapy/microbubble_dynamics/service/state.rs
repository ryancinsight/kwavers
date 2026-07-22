use std::collections::HashMap;

use kwavers_core::constants::cavitation::{
    POLYTROPIC_EXPONENT_AIR, SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER_25C, VISCOSITY_WATER,
};
use kwavers_core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, AVOGADRO, DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::{
    BODY_TEMPERATURE_K, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
};
use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::bubble_dynamics::bubble_state::{
    BubbleParameters, BubbleState, GasSpecies,
};
use kwavers_physics::therapy::microbubble::{MarmottantShellProperties, MicrobubbleState};

use super::MicrobubbleDynamicsService;

impl MicrobubbleDynamicsService {
    /// Extract Keller-Miksis parameters from microbubble state
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn extract_bubble_parameters(
        state: &MicrobubbleState,
    ) -> KwaversResult<BubbleParameters> {
        // All fluid properties sourced from core::constants SSOT.
        // VISCOSITY_WATER = 1.002e-3 Pa·s at 20°C (NIST); nearest available constant.
        // VAPOR_PRESSURE_WATER_25C = 3169.0 Pa at 25°C (CRC Handbook Table 6-5).
        let mu_liquid = VISCOSITY_WATER;
        let sigma = SURFACE_TENSION_WATER;
        let pv = VAPOR_PRESSURE_WATER_25C;
        let gamma = POLYTROPIC_EXPONENT_AIR;
        let t0 = BODY_TEMPERATURE_K;
        let thermal_cond = THERMAL_CONDUCTIVITY_WATER;
        let cp_liquid = SPECIFIC_HEAT_WATER;

        let mut gas_composition = HashMap::new();
        gas_composition.insert(
            kwavers_physics::acoustics::bubble_dynamics::bubble_state::GasType::N2,
            0.79,
        );
        gas_composition.insert(
            kwavers_physics::acoustics::bubble_dynamics::bubble_state::GasType::O2,
            0.21,
        );

        Ok(BubbleParameters {
            r0: state.radius_equilibrium,
            p0: ATMOSPHERIC_PRESSURE,
            rho_liquid: DENSITY_WATER_NOMINAL,
            c_liquid: SOUND_SPEED_TISSUE,
            mu_liquid,
            sigma,
            pv,
            thermal_conductivity: thermal_cond,
            specific_heat_liquid: cp_liquid,
            accommodation_coeff: 0.4,
            gas_species: GasSpecies::Air,
            initial_gas_pressure: ATMOSPHERIC_PRESSURE,
            gas_composition,
            gamma,
            t0,
            driving_frequency: MHZ_TO_HZ,
            driving_amplitude: 0.0,
            use_compressibility: true,
            use_thermal_effects: false,
            use_mass_transfer: false,
        })
    }

    /// Convert domain state to Keller-Miksis bubble state
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
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