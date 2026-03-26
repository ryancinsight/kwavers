//! Vapor thermal and transport properties

use std::f64::consts::PI;

use super::vapor_pressure::ThermodynamicsCalculator;
use crate::core::constants::{M_WATER, P_CRITICAL_WATER, R_GAS, T_CRITICAL_WATER};

impl ThermodynamicsCalculator {
    /// Calculate enthalpy of vaporization at given temperature
    ///
    /// Uses Watson correlation for temperature dependence
    #[must_use]
    pub fn enthalpy_vaporization(&self, temperature: f64) -> f64 {
        // Reference values at normal boiling point
        const H_VAP_NBP: f64 = 40660.0; // J/mol at 100°C
        const T_NBP: f64 = 373.15; // Normal boiling point

        if temperature >= T_CRITICAL_WATER {
            return 0.0;
        }

        // Watson correlation
        let tr = temperature / T_CRITICAL_WATER;
        let tr_nbp = T_NBP / T_CRITICAL_WATER;

        H_VAP_NBP * ((1.0 - tr) / (1.0 - tr_nbp)).powf(0.38)
    }

    /// Calculate specific heat capacity of water vapor
    #[must_use]
    pub fn heat_capacity_vapor(&self, temperature: f64) -> f64 {
        // Shomate equation coefficients for water vapor
        // Valid 500-1700 K (extended range)
        const A: f64 = 30.092;
        const B: f64 = 6.832514;
        const C: f64 = 6.793435;
        const D: f64 = -2.534480;
        const E: f64 = 0.082139;

        let t = temperature / 1000.0;
        A + B * t + C * t * t + D * t * t * t + E / (t * t)
    }

    /// Calculate thermal conductivity of water vapor
    #[must_use]
    pub fn thermal_conductivity_vapor(&self, temperature: f64, pressure: f64) -> f64 {
        // Water vapor correlation calculation
        // From IAPWS formulation
        let t_reduced = temperature / T_CRITICAL_WATER;
        let p_reduced = pressure / P_CRITICAL_WATER;

        // Base conductivity at low pressure
        let k0 = 0.0181 * (temperature / 300.0).powf(0.76);

        // Pressure correction
        let k_corr = 1.0 + 0.003 * p_reduced / t_reduced;

        k0 * k_corr
    }

    /// Calculate dynamic viscosity of water vapor
    #[must_use]
    pub fn viscosity_vapor(&self, temperature: f64) -> f64 {
        // Sutherland's formula for water vapor
        const MU_REF: f64 = 1.12e-5; // Pa·s at 373 K
        const T_REF: f64 = 373.0; // K
        const S: f64 = 961.0; // Sutherland constant for water vapor

        MU_REF * (temperature / T_REF).powf(1.5) * (T_REF + S) / (temperature + S)
    }

    /// Calculate mass transfer coefficient for evaporation/condensation
    #[must_use]
    pub fn mass_transfer_coefficient(&self, temperature: f64, accommodation_coeff: f64) -> f64 {
        // Hertz-Knudsen equation coefficient
        let molecular_speed = (8.0 * R_GAS * temperature / (PI * M_WATER)).sqrt();
        accommodation_coeff * molecular_speed / 4.0
    }

    /// Calculate equilibrium vapor concentration
    #[must_use]
    pub fn vapor_concentration(&self, temperature: f64, pressure: f64) -> f64 {
        let p_sat = self.vapor_pressure(temperature);
        let mole_fraction = p_sat / pressure;

        // Convert to mass concentration (kg/m³)
        mole_fraction * pressure * M_WATER / (R_GAS * temperature)
    }
}
