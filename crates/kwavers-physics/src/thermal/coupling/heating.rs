//! Acoustic heating source modeling

/// Acoustic heating source
///
/// Represents heat generation from ultrasound absorption (I²-type heating)
#[derive(Debug, Clone, Copy)]
pub struct AcousticHeatingSource {
    /// Absorption coefficient [Np/m]
    pub absorption_coefficient: f64,
    /// Acoustic intensity [W/m²]
    pub intensity: f64,
}

impl AcousticHeatingSource {
    /// Create acoustic heating source
    #[must_use]
    pub fn new(absorption_coefficient: f64, intensity: f64) -> Self {
        Self {
            absorption_coefficient,
            intensity,
        }
    }

    /// Heat source power [W/m³]
    ///
    /// Q = 2·α·I
    /// where:
    /// - α = absorption coefficient [Np/m]
    /// - I = acoustic intensity [W/m²]
    ///
    /// This comes from the acoustic power balance equation.
    #[must_use]
    pub fn power(&self) -> f64 {
        2.0 * self.absorption_coefficient * self.intensity
    }

    /// Heat source power at depth z
    ///
    /// Q(z) = 2·α·I·exp(-2αz)
    /// Accounts for attenuation of acoustic intensity with depth.
    #[must_use]
    pub fn power_at_depth(&self, depth: f64) -> f64 {
        2.0 * self.absorption_coefficient
            * self.intensity
            * (-2.0 * self.absorption_coefficient * depth).exp()
    }
}
