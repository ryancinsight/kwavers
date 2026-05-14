use super::StateDependentConstants;
use crate::core::constants::water::WaterProperties;

impl StateDependentConstants {
    /// Calculate speed of sound in water with temperature and pressure dependence
    ///
    /// Uses Del Grosso (1972) temperature model combined with Holton (1951) pressure model.
    /// Valid range: 0-100°C, 0-10 MPa
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `pressure` - Absolute pressure (Pa)
    ///
    /// # Returns
    /// Speed of sound (m/s)
    #[must_use]
    pub fn sound_speed_water(&self, temperature: f64, pressure: f64) -> f64 {
        let c0 = WaterProperties::sound_speed(temperature);

        // Add pressure dependence: c(p) = c₀(1 + β·Δp)
        // Pressure coefficient β ≈ 1.0e-10 Pa⁻¹ (Holton 1951)
        let beta = 1.0e-10; // Pa⁻¹
        let delta_p = pressure - self.reference_pressure;

        c0 * (1.0 + beta * delta_p)
    }

    /// Calculate nonlinear parameter B/A for water with temperature dependence
    ///
    /// Delegates to [`WaterProperties::nonlinear_parameter`] which uses the
    /// Beyer (1960) 4th-order polynomial fit. Valid range: 0-100°C
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Nonlinear parameter B/A (dimensionless)
    #[must_use]
    pub fn nonlinear_parameter_water(&self, temperature: f64) -> f64 {
        WaterProperties::nonlinear_parameter(temperature)
    }

    /// Calculate frequency-dependent attenuation coefficient for water
    ///
    /// Uses power law model: α(f) = α₀·f^b with Francois & Garrison (1982) model.
    ///
    /// # Arguments
    /// * `frequency` - Frequency (Hz)
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Attenuation coefficient [Np/m]
    #[must_use]
    pub fn attenuation_coefficient_water(&self, frequency: f64, temperature: f64) -> f64 {
        WaterProperties::absorption_coefficient(
            frequency,
            temperature,
            0.0, // depth
            0.0, // salinity (fresh water)
            7.0, // pH (neutral)
        )
    }

    /// Calculate frequency-dependent attenuation coefficient for soft tissue
    ///
    /// Power law model: α(f) = α₀·f^b
    ///
    /// For soft tissue (Duck 1990):
    /// - α₀ = 0.5 dB/(cm·MHz^b)
    /// - b = 1.5 (frequency exponent, tissue-specific)
    ///
    /// Temperature dependence (Q10 = 1.3):
    /// α(T) = α(T₀)·Q10^((T - T₀)/10)
    ///
    /// # Arguments
    /// * `frequency` - Frequency (Hz)
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Attenuation coefficient [Np/m]
    #[must_use]
    pub fn attenuation_coefficient_tissue(&self, frequency: f64, temperature: f64) -> f64 {
        const ALPHA_0: f64 = 0.5; // dB/(cm·MHz^b)
        const B: f64 = 1.5; // Frequency exponent
        const T_REF: f64 = 37.0; // Body temperature reference
        const Q10: f64 = 1.3; // Temperature sensitivity

        let f_mhz = frequency / 1e6;
        let alpha_db_cm = ALPHA_0 * f_mhz.powf(B);
        let temp_factor = Q10.powf((temperature - T_REF) / 10.0);
        let alpha_db_cm_corrected = alpha_db_cm * temp_factor;

        alpha_db_cm_corrected * crate::core::constants::DB_TO_NP * 100.0
    }

    /// Calculate acoustic impedance Z = ρ·c
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `pressure` - Absolute pressure (Pa)
    ///
    /// # Returns
    /// Acoustic impedance [kg/(m²·s)] or (Rayl)
    #[must_use]
    pub fn acoustic_impedance_water(&self, temperature: f64, pressure: f64) -> f64 {
        let rho = WaterProperties::density(temperature);
        let c = self.sound_speed_water(temperature, pressure);
        rho * c
    }

    /// Calculate bulk modulus K = ρ·c²
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `pressure` - Absolute pressure (Pa)
    ///
    /// # Returns
    /// Bulk modulus (Pa)
    #[must_use]
    pub fn bulk_modulus_water(&self, temperature: f64, pressure: f64) -> f64 {
        let rho = WaterProperties::density(temperature);
        let c = self.sound_speed_water(temperature, pressure);
        rho * c * c
    }
}
