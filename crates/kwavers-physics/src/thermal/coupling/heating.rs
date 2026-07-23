//! Acoustic heating source modeling

use aequitas::systems::si::quantities::{
    Dimensionless, Intensity, Length, ReciprocalLength, VolumetricPowerDensity,
};

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
        let absorption = ReciprocalLength::from_base(self.absorption_coefficient);
        let intensity = Intensity::from_base(self.intensity);
        let power_density: VolumetricPowerDensity = absorption * intensity;
        (power_density * 2.0).into_base()
    }

    /// Heat source power at depth z
    ///
    /// Q(z) = 2·α·I·exp(-2αz)
    /// Accounts for attenuation of acoustic intensity with depth.
    #[must_use]
    pub fn power_at_depth(&self, depth: f64) -> f64 {
        let absorption = ReciprocalLength::from_base(self.absorption_coefficient);
        let intensity = Intensity::from_base(self.intensity);
        let power_density: VolumetricPowerDensity = absorption * intensity;
        let optical_depth: Dimensionless = absorption * Length::from_base(depth);
        (power_density * 2.0 * (-2.0 * optical_depth.into_base()).exp()).into_base()
    }
}
