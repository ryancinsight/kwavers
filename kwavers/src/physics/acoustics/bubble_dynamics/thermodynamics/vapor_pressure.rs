//! Vapor pressure models for phase equilibrium

use crate::core::constants::{
    H_VAP_WATER_100C, P_ATM, P_CRITICAL_WATER, P_TRIPLE_WATER, R_GAS, T_BOILING_WATER,
    T_CRITICAL_WATER, T_TRIPLE_WATER,
};

/// Vapor pressure model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub(crate) model: VaporPressureModel,
    /// Enthalpy of vaporization [J/mol]
    pub(crate) h_vap: f64,
    /// Reference temperature for Clausius-Clapeyron \[K\]
    pub(crate) t_ref: f64,
    /// Reference pressure for Clausius-Clapeyron \[Pa\]
    pub(crate) p_ref: f64,
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
    #[must_use]
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
    #[must_use]
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

        let t_celsius = crate::physics::constants::kelvin_to_celsius(temperature);

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
    /// `ln(P/P_ref)` = -`ΔH_vap/R` * (1/T - `1/T_ref`)
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
    pub(crate) fn wagner_equation(&self, temperature: f64) -> f64 {
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
    /// Magnus formula for vapor pressure calculation
    fn buck_equation(&self, temperature: f64) -> f64 {
        let t_celsius = crate::physics::constants::kelvin_to_celsius(temperature);

        // Buck (1981) coefficients
        let (a, b, _c) = if t_celsius > 0.0 {
            (17.368, 238.88, 234.5) // Above freezing
        } else {
            (17.966, 247.15, 233.7) // Below freezing
        };

        let exponent: f64 = a * t_celsius / (b + t_celsius);
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
}
