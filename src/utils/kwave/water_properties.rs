//! Water properties and attenuation models
//!
//! Implements water property calculations based on:
//! - Bilaniuk & Wong (1993): Sound speed model
//! - Kell (1975): Density model
//! - Francois & Garrison (1982): Absorption model
//! - Pinkerton (1949): Simplified absorption model

/// Water properties and attenuation models
#[derive(Debug, Debug))]
pub struct WaterProperties;

impl WaterProperties {
    // Sound speed polynomial coefficients (Bilaniuk & Wong, 1993)
    const SOUND_SPEED_C0: f64 = 1402.385;
    const SOUND_SPEED_C1: f64 = 5.03830;
    const SOUND_SPEED_C2: f64 = -5.81090e-2;
    const SOUND_SPEED_C3: f64 = 3.34320e-4;
    const SOUND_SPEED_C4: f64 = -1.48259e-6;
    const SOUND_SPEED_C5: f64 = 3.16090e-9;

    // Water density polynomial coefficients (Kell, 1975)
    // Reference: Kell, G. S. (1975). "Density, thermal expansivity, and compressibility of liquid water from 0° to 150°C"
    // Units: All coefficients in kg/m³, temperature in °C
    /// Constant term [kg/m³]
    const KELL_A: f64 = 999.83952;
    /// Linear coefficient [kg/m³·°C⁻¹]
    const KELL_B: f64 = 16.945176;
    /// Quadratic coefficient [kg/m³·°C⁻²]
    const KELL_C: f64 = -7.9870401e-3;
    /// Cubic coefficient [kg/m³·°C⁻³]
    const KELL_D: f64 = -46.170461e-6;
    /// Quartic coefficient [kg/m³·°C⁻⁴]
    const KELL_E: f64 = 105.56302e-9;
    /// Quintic coefficient [kg/m³·°C⁻⁵]
    const KELL_F: f64 = -280.54253e-12;
    /// Denominator linear coefficient [dimensionless]
    const KELL_G: f64 = 16.879850e-3;

    /// Calculate water density as function of temperature
    /// Based on Kell (1975) formula
    pub fn density(temperature: f64) -> f64 {
        // Temperature in Celsius
        let t = temperature;

        // Kell's formula for water density (kg/m³)
        let numerator = Self::KELL_A
            + Self::KELL_B * t
            + Self::KELL_C * t.powi(2)
            + Self::KELL_D * t.powi(3)
            + Self::KELL_E * t.powi(4)
            + Self::KELL_F * t.powi(5);
        let denominator = 1.0 + Self::KELL_G * t;
        numerator / denominator
    }

    /// Calculate water sound speed as function of temperature
    /// Based on Bilaniuk & Wong (1993)
    pub fn sound_speed(temperature: f64) -> f64 {
        // Temperature in Celsius
        let t = temperature;

        // 5th order polynomial fit (Bilaniuk & Wong, 1993)
        Self::SOUND_SPEED_C0
            + Self::SOUND_SPEED_C1 * t
            + Self::SOUND_SPEED_C2 * t.powi(2)
            + Self::SOUND_SPEED_C3 * t.powi(3)
            + Self::SOUND_SPEED_C4 * t.powi(4)
            + Self::SOUND_SPEED_C5 * t.powi(5)
    }

    /// Calculate water absorption coefficient
    /// Based on Francois & Garrison (1982) model
    pub fn absorption_coefficient(
        frequency: f64,   // Hz
        temperature: f64, // Celsius
        depth: f64,       // meters
        salinity: f64,    // parts per thousand
        ph: f64,          // pH value
    ) -> f64 {
        let f = frequency / 1e3; // Convert to kHz
        let t = temperature;
        let s = salinity;
        let d = depth / 1e3; // Convert to km

        // Boric acid contribution
        let f1 = 0.78 * (s / 35.0).sqrt() * (t / 26.0).exp();
        let a1 = 8.86 / WaterProperties::sound_speed(t) * 10.0_f64.powf(0.78 * ph - 5.0);
        let p1 = 1.0;
        let boric = a1 * p1 * f1 * f * f / (f1 * f1 + f * f);

        // Magnesium sulfate contribution
        let f2 = 42.0 * (t / 17.0).exp();
        let a2 = 21.44 * s / WaterProperties::sound_speed(t) * (1.0 + 0.025 * t);
        let p2 = 1.0 - 1.37e-4 * d + 6.2e-9 * d * d;
        let magnesium = a2 * p2 * f2 * f * f / (f2 * f2 + f * f);

        // Pure water contribution
        let a3 = if t <= 20.0 {
            4.937e-4 - 2.59e-5 * t + 9.11e-7 * t * t - 1.50e-8 * t * t * t
        } else {
            3.964e-4 - 1.146e-5 * t + 1.45e-7 * t * t - 6.5e-10 * t * t * t
        };
        let p3 = 1.0 - 3.83e-5 * d + 4.9e-10 * d * d;
        let water = a3 * p3 * f * f;

        // Total absorption in dB/km, convert to Np/m
        let alpha_db_per_km = boric + magnesium + water;
        alpha_db_per_km * 0.1151 / 1e3 // Convert to Np/m
    }

    /// Pinkerton model for absorption calculations
    pub fn pinkerton_absorption(frequency: f64, temperature: f64) -> f64 {
        // Pinkerton (1949) model: α = A * f²
        // where A depends on temperature
        let f_mhz = frequency / 1e6;
        let a = 25.3 * ((-17.0 / (temperature + 273.15)).exp());

        a * f_mhz * f_mhz * 1e-3 // Convert to Np/m
    }
}
