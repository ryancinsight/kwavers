//! Water properties and attenuation models
//!
//! Implements water property calculations based on:
//! - Marczak (1997): Sound speed model (k-wave compatible)
//! - Jones & Harris (1992): Density model (k-wave compatible)
//! - Beyer (1960): Nonlinear parameter B/A (k-wave compatible)
//! - Francois & Garrison (1982): Absorption model
//! - Pinkerton (1949): Simplified absorption model

/// Water properties and attenuation models
#[derive(Debug)]
pub struct WaterProperties;

impl WaterProperties {
    // Sound speed polynomial coefficients — Marczak (1997)
    // Reference: R. Marczak (1997). "The sound velocity in water as a function of
    // temperature". Journal of Research of NIST, 102(6), 561-567.
    // Valid range: 0–95 °C.  Matches k-wave-python `water_sound_speed`.
    const SOUND_SPEED_C0: f64 = 1.402385e3;
    const SOUND_SPEED_C1: f64 = 5.038813;
    const SOUND_SPEED_C2: f64 = -5.779136e-2;
    const SOUND_SPEED_C3: f64 = 3.287156e-4;
    const SOUND_SPEED_C4: f64 = -1.398845e-6;
    const SOUND_SPEED_C5: f64 = 2.787860e-9;

    // Water density polynomial coefficients — Jones & Harris (1992)
    // Reference: F. E. Jones and G. L. Harris (1992) "ITS-90 Density of Water
    // Formulation for Volumetric Standards Calibration," Journal of Research of
    // NIST, 97(3), 335-340.
    // Valid range: 5–40 °C.  Matches k-wave-python `water_density`.
    const JONES_A: f64 = 999.84847;
    const JONES_B: f64 = 6.337563e-2;
    const JONES_C: f64 = -8.523829e-3;
    const JONES_D: f64 = 6.943248e-5;
    const JONES_E: f64 = -3.821216e-7;

    // Nonlinear parameter B/A polynomial coefficients — Beyer (1960)
    // Reference: R. T. Beyer (1960) "Parameter of nonlinearity in fluids,"
    // J. Acoust. Soc. Am., 32(6), 719-721.
    // Valid range: 0–100 °C.  Matches k-wave-python `water_non_linearity`.
    const BEYER_BA_C0: f64 = 4.186533937275504;
    const BEYER_BA_C1: f64 = 5.380874771364909e-2;
    const BEYER_BA_C2: f64 = -9.355518377254833e-4;
    const BEYER_BA_C3: f64 = 1.047843302423604e-5;
    const BEYER_BA_C4: f64 = -4.587913769504693e-8;

    /// Calculate water density as function of temperature.
    ///
    /// Uses Jones & Harris (1992) 4th-order polynomial for air-saturated water.
    /// Valid range: 5–40 °C.
    ///
    /// # Arguments
    /// * `temperature` — Temperature in degrees Celsius
    ///
    /// # Returns
    /// Density in kg/m³
    #[must_use]
    pub fn density(temperature: f64) -> f64 {
        let t = temperature;
        Self::JONES_A
            + Self::JONES_B * t
            + Self::JONES_C * t.powi(2)
            + Self::JONES_D * t.powi(3)
            + Self::JONES_E * t.powi(4)
    }

    /// Calculate water sound speed as function of temperature.
    ///
    /// Uses Marczak (1997) 5th-order polynomial.
    /// Valid range: 0–95 °C.
    ///
    /// # Arguments
    /// * `temperature` — Temperature in degrees Celsius
    ///
    /// # Returns
    /// Sound speed in m/s
    #[must_use]
    pub fn sound_speed(temperature: f64) -> f64 {
        let t = temperature;
        Self::SOUND_SPEED_C0
            + Self::SOUND_SPEED_C1 * t
            + Self::SOUND_SPEED_C2 * t.powi(2)
            + Self::SOUND_SPEED_C3 * t.powi(3)
            + Self::SOUND_SPEED_C4 * t.powi(4)
            + Self::SOUND_SPEED_C5 * t.powi(5)
    }

    /// Calculate water nonlinear parameter B/A as function of temperature.
    ///
    /// Uses Beyer (1960) 4th-order polynomial fit.
    /// Valid range: 0–100 °C.
    ///
    /// # Arguments
    /// * `temperature` — Temperature in degrees Celsius
    ///
    /// # Returns
    /// Dimensionless B/A parameter
    #[must_use]
    pub fn nonlinear_parameter(temperature: f64) -> f64 {
        let t = temperature;
        Self::BEYER_BA_C0
            + Self::BEYER_BA_C1 * t
            + Self::BEYER_BA_C2 * t.powi(2)
            + Self::BEYER_BA_C3 * t.powi(3)
            + Self::BEYER_BA_C4 * t.powi(4)
    }

    /// Calculate ultrasonic absorption in distilled water.
    ///
    /// Uses Pinkerton (1949) model: 7th-order polynomial in temperature,
    /// quadratic in frequency.  Matches k-wave-python `water_absorption`.
    ///
    /// # Arguments
    /// * `frequency_mhz` — Frequency in MHz
    /// * `temperature`   — Temperature in degrees Celsius (0–60 °C)
    ///
    /// # Returns
    /// Absorption in dB/cm
    ///
    /// # References
    /// J. M. M. Pinkerton (1949) "The Absorption of Ultrasonic Waves in
    /// Liquids and its Relation to Molecular Constitution," Proceedings of
    /// the Physical Society. Section B, 2, 129-141.
    #[must_use]
    pub fn absorption_pinkerton(frequency_mhz: f64, temperature: f64) -> f64 {
        const NEPER2DB: f64 = 8.686;
        const A: [f64; 8] = [
            56.723_531_840_522_71,
            -2.899633796917384,
            0.099253401567561,
            -0.002067402501557,
            2.189417428917596e-05,
            -6.210860973978427e-08,
            -6.402634551821596e-10,
            3.869387679459408e-12,
        ];

        let t = temperature;
        let a_on_fsqr = (A[0]
            + A[1] * t
            + A[2] * t.powi(2)
            + A[3] * t.powi(3)
            + A[4] * t.powi(4)
            + A[5] * t.powi(5)
            + A[6] * t.powi(6)
            + A[7] * t.powi(7))
            * 1e-17;

        NEPER2DB * 1e12 * frequency_mhz * frequency_mhz * a_on_fsqr
    }

    /// Calculate water absorption coefficient
    /// Based on Francois & Garrison (1982) model
    #[must_use]
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
}
