//! Thermal and mass transfer rate calculations for IMEX implicit step
//!
//! ## Mathematical Foundation
//!
//! ### Thermal rate (dT/dt)
//! Compression heating: -(γ-1) T v / R
//! Heat transfer: -h (T - T_ambient) / (n_gas + n_vapor)
//! Nusselt correlation: Nu = 2 + 0.6 Pe^(1/3)
//!
//! ### Mass transfer rate (dn_vapor/dt)
//! Antoine equation for equilibrium vapor pressure
//! Sherwood correlation for mass transfer coefficient

use super::integrator::BubbleIMEXIntegrator;
use crate::acoustics::bubble_dynamics::BubbleState;
use kwavers_core::constants::fundamental::GAS_CONSTANT as R_GAS;
use kwavers_core::constants::numerical::MMHG_TO_PA;
use kwavers_core::constants::thermodynamic::{
    NUSSELT_CONSTANT, NUSSELT_PECLET_COEFF, SHERWOOD_PECLET_EXPONENT, T_AMBIENT,
    VAPOR_DIFFUSION_COEFFICIENT, WATER_ANTOINE_A, WATER_ANTOINE_B, WATER_ANTOINE_C,
};
use kwavers_core::error::KwaversResult;

impl BubbleIMEXIntegrator {
    /// Calculate thermal and mass transfer rates without modifying state
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn calculate_thermal_mass_transfer_rates(
        &self,
        state: &BubbleState,
    ) -> KwaversResult<(f64, f64)> {
        let r = state.radius;
        let v = state.wall_velocity;
        let temperature = state.temperature;
        let n_vapor = state.n_vapor;
        let n_gas = state.n_gas;

        let params = self.solver.params();
        let dt_dt = if params.use_thermal_effects {
            let gamma = self.calculate_effective_polytropic_index(state);

            let thermal_diffusivity =
                params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);
            let peclet = (2.0 * r * v.abs()) / thermal_diffusivity;
            let nusselt = NUSSELT_PECLET_COEFF.mul_add(peclet.sqrt(), NUSSELT_CONSTANT);
            let h = nusselt * params.thermal_conductivity / (2.0 * r);

            let compression_heating = -(gamma - 1.0) * temperature * v / r;
            let heat_transfer = -h * (temperature - T_AMBIENT) / (n_gas + n_vapor);

            compression_heating + heat_transfer
        } else {
            0.0
        };

        let dn_vapor_dt = if params.use_mass_transfer {
            let p_vapor_eq = self.calculate_equilibrium_vapor_pressure(temperature);
            let p_vapor_actual = n_vapor * R_GAS * temperature / state.volume();

            let d_vapor = VAPOR_DIFFUSION_COEFFICIENT;
            let thermal_diffusivity =
                params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);
            let peclet = (2.0 * r * v.abs()) / thermal_diffusivity;
            let sherwood = NUSSELT_PECLET_COEFF
                .mul_add(peclet.powf(SHERWOOD_PECLET_EXPONENT), NUSSELT_CONSTANT);
            let k_mass = sherwood * d_vapor / (2.0 * r);

            let driving_force = p_vapor_eq - p_vapor_actual;
            k_mass * driving_force * state.surface_area() / (R_GAS * temperature)
        } else {
            0.0
        };

        Ok((dt_dt, dn_vapor_dt))
    }

    /// Calculate effective polytropic index for thermal model
    pub(crate) fn calculate_effective_polytropic_index(&self, state: &BubbleState) -> f64 {
        use kwavers_core::constants::cavitation::{MIN_PECLET_NUMBER, PECLET_SCALING_FACTOR};

        let params = self.solver.params();
        let thermal_diffusivity =
            params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);
        let peclet = (2.0 * state.radius * state.wall_velocity.abs()) / thermal_diffusivity;
        let peclet_eff = peclet.max(MIN_PECLET_NUMBER);

        let gamma_gas = state.gas_species.gamma();
        1.0 + (gamma_gas - 1.0) / (1.0 + PECLET_SCALING_FACTOR / peclet_eff)
    }

    /// Calculate equilibrium vapor pressure at given temperature (Antoine equation).
    ///
    /// log₁₀(P_mmHg) = A − B / (C + T_celsius); coefficients from Stull (1947).
    pub(crate) fn calculate_equilibrium_vapor_pressure(&self, temperature: f64) -> f64 {
        let t_celsius = kwavers_core::constants::thermodynamic::kelvin_to_celsius(temperature);
        let log10_p = WATER_ANTOINE_A - WATER_ANTOINE_B / (WATER_ANTOINE_C + t_celsius);
        10.0_f64.powf(log10_p) * MMHG_TO_PA
    }
}
