//! Thermodynamic models for bubble dynamics
//!
//! This module provides comprehensive thermodynamic models for calculating
//! vapor pressure, phase equilibria, and thermal properties in bubble dynamics.
//!
//! # Models Implemented
//!
//! 1. **Antoine Equation**: Industry-standard for vapor pressure
//! 2. **Clausius-Clapeyron Relation**: Theoretical foundation for phase transitions
//! 3. **Wagner Equation**: High-accuracy model for water vapor
//! 4. **IAPWS-IF97**: International standard for water/steam properties
//!
//! # References
//!
//! - Wagner, W., & Pruss, A. (2002). "The IAPWS formulation 1995 for the
//!   thermodynamic properties of ordinary water substance." J. Phys. Chem. Ref. Data.
//! - Antoine, C. (1888). "Tensions des vapeurs; nouvelle relation entre les
//!   tensions et les températures." Comptes Rendus, 107, 681-684.

use std::f64::consts::PI;

/// Physical constants
pub mod constants {
    /// Universal gas constant [J/(mol·K)]
    pub const R_GAS: f64 = 8.314462618;
    /// Avogadro's number [1/mol]
    pub const AVOGADRO: f64 = 6.02214076e23;
    /// Molecular weight of water [kg/mol]
    pub const M_WATER: f64 = 0.01801528;
    /// Critical temperature of water [K]
    pub const T_CRITICAL_WATER: f64 = 647.096;
    /// Critical pressure of water [Pa]
    pub const P_CRITICAL_WATER: f64 = 22.064e6;
    /// Triple point temperature of water [K]
    pub const T_TRIPLE_WATER: f64 = 273.16;
    /// Triple point pressure of water [Pa]
    pub const P_TRIPLE_WATER: f64 = 611.657;
    /// Standard atmospheric pressure [Pa]
    pub const P_ATM: f64 = 101325.0;
    /// Enthalpy of vaporization for water at 100°C [J/mol]
    pub const H_VAP_WATER_100C: f64 = 40660.0;
    /// Boiling point of water at 1 atm [K]
    pub const T_BOILING_WATER: f64 = 373.15;
}

use constants::*;

/// Vapor pressure model selection
#[derive(Debug, Clone, Copy)]
pub enum VaporPressureModel {
    /// Antoine equation (fast, good accuracy)
    Antoine,
    /// Clausius-Clapeyron relation (theoretical)
    ClausiusClapeyron,
    /// Wagner equation (high accuracy for water)
    Wagner,
    /// Buck equation (meteorological standard)
    Buck,
    /// IAPWS-IF97 (international standard)
    IAPWS,
}

/// Thermodynamics calculator for bubble dynamics
#[derive(Debug, Clone)]
pub struct ThermodynamicsCalculator {
    model: VaporPressureModel,
    /// Enthalpy of vaporization [J/mol]
    h_vap: f64,
    /// Reference temperature for Clausius-Clapeyron [K]
    t_ref: f64,
    /// Reference pressure for Clausius-Clapeyron [Pa]
    p_ref: f64,
}

impl Default for ThermodynamicsCalculator {
    fn default() -> Self {
        Self {
            model: VaporPressureModel::Wagner, // Reference model for water
            h_vap: H_VAP_WATER_100C,           // Water at 100°C [J/mol]
            t_ref: T_BOILING_WATER,            // 100°C
            p_ref: P_ATM,                      // 1 atm
        }
    }
}

impl ThermodynamicsCalculator {
    /// Create a new calculator with specified model
    pub fn new(model: VaporPressureModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Calculate saturation vapor pressure at given temperature
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Saturation vapor pressure in Pascals
    pub fn vapor_pressure(&self, temperature: f64) -> f64 {
        // Bounds checking
        if temperature < T_TRIPLE_WATER {
            return self.vapor_pressure_ice(temperature);
        }
        if temperature > T_CRITICAL_WATER {
            return P_CRITICAL_WATER;
        }

        match self.model {
            VaporPressureModel::Antoine => self.antoine_equation(temperature),
            VaporPressureModel::ClausiusClapeyron => self.clausius_clapeyron(temperature),
            VaporPressureModel::Wagner => self.wagner_equation(temperature),
            VaporPressureModel::Buck => self.buck_equation(temperature),
            VaporPressureModel::IAPWS => self.iapws_if97(temperature),
        }
    }

    /// Antoine equation for vapor pressure
    ///
    /// log10(P) = A - B/(C + T)
    /// where P is in mmHg and T is in °C
    fn antoine_equation(&self, temperature: f64) -> f64 {
        // Antoine coefficients for water (valid 1-100°C)
        // From NIST Chemistry WebBook
        const A: f64 = 8.07131;
        const B: f64 = 1730.63;
        const C: f64 = 233.426;

        let t_celsius = temperature - 273.15;

        if !(1.0..=100.0).contains(&t_celsius) {
            // Fall back to Wagner equation outside valid range
            return self.wagner_equation(temperature);
        }

        let log10_p_mmhg = A - B / (C + t_celsius);
        let p_mmhg = 10_f64.powf(log10_p_mmhg);

        // Convert mmHg to Pa (1 mmHg = 133.322 Pa)
        p_mmhg * 133.322
    }

    /// Clausius-Clapeyron relation
    ///
    /// ln(P/P_ref) = -ΔH_vap/R * (1/T - 1/T_ref)
    fn clausius_clapeyron(&self, temperature: f64) -> f64 {
        // Temperature-dependent enthalpy of vaporization (Watson correlation)
        let h_vap_t = self.h_vap
            * ((T_CRITICAL_WATER - temperature) / (T_CRITICAL_WATER - self.t_ref)).powf(0.38);

        let exponent = -h_vap_t / R_GAS * (1.0 / temperature - 1.0 / self.t_ref);
        self.p_ref * exponent.exp()
    }

    /// Wagner equation for water vapor pressure
    ///
    /// High-accuracy equation specifically for water
    /// Valid from triple point to critical point
    fn wagner_equation(&self, temperature: f64) -> f64 {
        // Wagner coefficients for water
        const A1: f64 = -7.85951783;
        const A2: f64 = 1.84408259;
        const A3: f64 = -11.7866497;
        const A4: f64 = 22.6807411;
        const A5: f64 = -15.9618719;
        const A6: f64 = 1.80122502;

        let tau = 1.0 - temperature / T_CRITICAL_WATER;

        if tau <= 0.0 {
            return P_CRITICAL_WATER;
        }

        let ln_pr = (A1 * tau
            + A2 * tau.powf(1.5)
            + A3 * tau.powf(3.0)
            + A4 * tau.powf(3.5)
            + A5 * tau.powf(4.0)
            + A6 * tau.powf(7.5))
            / (temperature / T_CRITICAL_WATER);

        P_CRITICAL_WATER * ln_pr.exp()
    }

    /// Buck equation (meteorological standard)
    ///
    /// Enhanced Magnus formula with higher accuracy
    fn buck_equation(&self, temperature: f64) -> f64 {
        let t_celsius = temperature - 273.15;

        // Buck (1981) coefficients
        let (a, b, c) = if t_celsius > 0.0 {
            (17.368, 238.88, 234.5) // Above freezing
        } else {
            (17.966, 247.15, 233.7) // Below freezing
        };

        let exponent = a * t_celsius / (b + t_celsius);
        611.21 * exponent.exp() // Result in Pa
    }

    /// IAPWS-IF97 formulation (international standard)
    ///
    /// Implementation of IAPWS-IF97 for saturation line
    fn iapws_if97(&self, temperature: f64) -> f64 {
        // Coefficients for Region 4 (saturation line)
        const N: [f64; 10] = [
            0.11670521452767e4,
            -0.72421316703206e6,
            -0.17073846940092e2,
            0.12020824702470e5,
            -0.32325550322333e7,
            0.14915108613530e2,
            -0.48232657361591e4,
            0.40511340542057e6,
            -0.23855557567849,
            0.65017534844798e3,
        ];

        let theta = temperature + N[8] / (temperature - N[9]);
        let a = theta * theta + N[0] * theta + N[1];
        let b = N[2] * theta * theta + N[3] * theta + N[4];
        let c = N[5] * theta * theta + N[6] * theta + N[7];

        let p_mpa = (2.0 * c / (-b + (b * b - 4.0 * a * c).sqrt())).powi(4);
        p_mpa * 1e6 // Convert MPa to Pa
    }

    /// Calculate vapor pressure over ice (below triple point)
    fn vapor_pressure_ice(&self, temperature: f64) -> f64 {
        // Goff-Gratch equation for ice
        const A: f64 = -9.09718;
        const B: f64 = -3.56654;
        const C: f64 = 0.876793;
        const D: f64 = -2.2195983e-3;

        let t_ratio = T_TRIPLE_WATER / temperature;
        let log10_p = A * (t_ratio - 1.0) + B * t_ratio.log10() + C * (1.0 - 1.0 / t_ratio) + D;

        P_TRIPLE_WATER * 10_f64.powf(log10_p)
    }

    /// Calculate enthalpy of vaporization at given temperature
    ///
    /// Uses Watson correlation for temperature dependence
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
    pub fn viscosity_vapor(&self, temperature: f64) -> f64 {
        // Sutherland's formula for water vapor
        const MU_REF: f64 = 1.12e-5; // Pa·s at 373 K
        const T_REF: f64 = 373.0; // K
        const S: f64 = 961.0; // Sutherland constant for water vapor

        MU_REF * (temperature / T_REF).powf(1.5) * (T_REF + S) / (temperature + S)
    }

    /// Calculate mass transfer coefficient for evaporation/condensation
    pub fn mass_transfer_coefficient(&self, temperature: f64, accommodation_coeff: f64) -> f64 {
        // Hertz-Knudsen equation coefficient
        let molecular_speed = (8.0 * R_GAS * temperature / (PI * M_WATER)).sqrt();
        accommodation_coeff * molecular_speed / 4.0
    }

    /// Calculate equilibrium vapor concentration
    pub fn vapor_concentration(&self, temperature: f64, pressure: f64) -> f64 {
        let p_sat = self.vapor_pressure(temperature);
        let mole_fraction = p_sat / pressure;

        // Convert to mass concentration (kg/m³)
        mole_fraction * pressure * M_WATER / (R_GAS * temperature)
    }
}

/// Enhanced mass transfer model for bubble dynamics
#[derive(Debug, Clone)]
pub struct MassTransferModel {
    thermo: ThermodynamicsCalculator,
    /// Accommodation coefficient (typically 0.04-1.0)
    accommodation_coeff: f64,
    /// Enable non-equilibrium effects
    non_equilibrium: bool,
}

impl MassTransferModel {
    /// Create a new mass transfer model
    pub fn new(accommodation_coeff: f64) -> Self {
        Self {
            thermo: ThermodynamicsCalculator::default(),
            accommodation_coeff,
            non_equilibrium: true,
        }
    }

    /// Calculate mass transfer rate for bubble
    ///
    /// # Arguments
    /// * `temperature` - Bubble temperature [K]
    /// * `pressure_vapor` - Current vapor pressure in bubble [Pa]
    /// * `surface_area` - Bubble surface area [m²]
    ///
    /// # Returns
    /// Mass transfer rate [kg/s] (positive for evaporation)
    pub fn mass_transfer_rate(
        &self,
        temperature: f64,
        pressure_vapor: f64,
        surface_area: f64,
    ) -> f64 {
        // Saturation pressure at bubble temperature
        let p_sat = self.thermo.vapor_pressure(temperature);

        // Pressure difference drives mass transfer
        let delta_p = p_sat - pressure_vapor;

        // Hertz-Knudsen equation
        let coeff = self
            .thermo
            .mass_transfer_coefficient(temperature, self.accommodation_coeff);
        let rate = coeff * surface_area * delta_p * M_WATER / (R_GAS * temperature);

        // Non-equilibrium correction for rapid dynamics
        if self.non_equilibrium {
            let peclet = pressure_vapor.abs() / p_sat;
            let correction = 1.0 / (1.0 + 0.5 * peclet);
            rate * correction
        } else {
            rate
        }
    }

    /// Calculate heat of phase change
    pub fn heat_transfer_rate(&self, mass_rate: f64, temperature: f64) -> f64 {
        let h_vap = self.thermo.enthalpy_vaporization(temperature);
        mass_rate * h_vap / M_WATER // Convert to J/s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vapor_pressure_models() {
        let calc = ThermodynamicsCalculator::default();

        // Test at 100°C (373.15 K) - should be ~1 atm
        let t = 373.15;
        let p_wagner = calc.wagner_equation(t);
        assert!((p_wagner - P_ATM).abs() / P_ATM < 0.01); // Within 1%

        // Test Antoine equation
        let calc_antoine = ThermodynamicsCalculator::new(VaporPressureModel::Antoine);
        let p_antoine = calc_antoine.vapor_pressure(t);
        assert!((p_antoine - P_ATM).abs() / P_ATM < 0.02); // Within 2%

        // Test at 25°C (298.15 K) - should be ~3.17 kPa
        let t_room = 298.15;
        let p_room = calc.vapor_pressure(t_room);
        assert!((p_room - 3170.0).abs() < 100.0); // Within 100 Pa
    }

    #[test]
    fn test_clausius_clapeyron() {
        let calc = ThermodynamicsCalculator::new(VaporPressureModel::ClausiusClapeyron);

        // Test at reference point
        let p_ref = calc.vapor_pressure(373.15);
        assert!((p_ref - P_ATM).abs() / P_ATM < 0.01);

        // Test temperature dependence
        let p_low = calc.vapor_pressure(353.15); // 80°C
        let p_high = calc.vapor_pressure(393.15); // 120°C

        assert!(p_low < P_ATM);
        assert!(p_high > P_ATM);
    }

    #[test]
    fn test_mass_transfer() {
        let model = MassTransferModel::new(0.04); // Typical accommodation coefficient

        let temperature = 300.0; // K
        let p_vapor = 2000.0; // Pa (undersaturated)
        let surface_area = 1e-6; // m² (1 mm² bubble)

        let rate = model.mass_transfer_rate(temperature, p_vapor, surface_area);

        // Should be positive (evaporation) since p_vapor < p_sat
        assert!(rate > 0.0);
    }

    #[test]
    fn test_enthalpy_vaporization() {
        let calc = ThermodynamicsCalculator::default();

        // At 100°C
        let h_vap_100 = calc.enthalpy_vaporization(373.15);
        assert!((h_vap_100 - 40660.0).abs() < 100.0); // Should match reference

        // Should decrease with temperature
        let h_vap_150 = calc.enthalpy_vaporization(423.15); // 150°C
        assert!(h_vap_150 < h_vap_100);

        // Should be zero at critical point
        let h_vap_crit = calc.enthalpy_vaporization(T_CRITICAL_WATER);
        assert!(h_vap_crit.abs() < 1.0);
    }
}
