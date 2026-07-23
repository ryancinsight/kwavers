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
    pub absorption_coefficient: ReciprocalLength<f64>,
    /// Acoustic intensity [W/m²]
    pub intensity: Intensity<f64>,
}

impl AcousticHeatingSource {
    /// Create acoustic heating source
    #[must_use]
    pub const fn new(
        absorption_coefficient: ReciprocalLength<f64>,
        intensity: Intensity<f64>,
    ) -> Self {
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
    pub fn power(&self) -> VolumetricPowerDensity<f64> {
        let power_density: VolumetricPowerDensity<f64> =
            self.absorption_coefficient * self.intensity;
        power_density * 2.0
    }

    /// Heat source power at depth z
    ///
    /// Q(z) = 2·α·I·exp(-2αz)
    /// Accounts for attenuation of acoustic intensity with depth.
    #[must_use]
    pub fn power_at_depth(&self, depth: Length<f64>) -> VolumetricPowerDensity<f64> {
        let power_density = self.power();
        let optical_depth: Dimensionless = self.absorption_coefficient * depth;
        power_density * (-2.0 * optical_depth.into_base()).exp()
    }
}
