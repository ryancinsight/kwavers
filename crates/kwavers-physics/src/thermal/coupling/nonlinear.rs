//! Nonlinear acoustic effects (Second-harmonic generation)
//!
//! Nonlinear acoustics contribute to heating through shock formation
//! and generation of higher harmonics that are more readily absorbed.

use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, MassDensity, Pressure, Velocity, VolumetricPowerDensityGradient,
};
use kwavers_core::constants::numerical::TWO_PI;

/// Nonlinear acoustic effects generating secondary absorption
#[derive(Debug, Clone, Copy)]
pub struct NonlinearHeating {
    /// Nonlinearity parameter (B/A)
    pub nonlinearity_parameter: Dimensionless<f64>,
    /// Acoustic pressure amplitude (Pa)
    pub pressure: Pressure<f64>,
    /// Sound speed (m/s)
    pub sound_speed: Velocity<f64>,
    /// Density [kg/m³]
    pub density: MassDensity<f64>,
    /// Driving frequency (Hz)
    pub frequency: Frequency<f64>,
}

impl NonlinearHeating {
    /// Create nonlinear heating source
    #[must_use]
    pub fn new(
        nonlinearity_parameter: Dimensionless<f64>,
        pressure: Pressure<f64>,
        sound_speed: Velocity<f64>,
        density: MassDensity<f64>,
        frequency: Frequency<f64>,
    ) -> Self {
        Self {
            nonlinearity_parameter,
            pressure,
            sound_speed,
            density,
            frequency,
        }
    }

    /// Spatial gradient of the nonlinear volumetric power-density term [W/m⁴].
    ///
    /// Q_nl = (B/A)·P²·ω² / (ρ·c³)
    ///
    /// Derived from the second-order Westervelt source term: the generated
    /// harmonics are absorbed proportional to ω² (Hamilton & Blackstock 1998,
    /// §4.3; Sehgal & Greenleaf 1984).
    #[must_use]
    pub fn power_density_gradient(&self) -> VolumetricPowerDensityGradient<f64> {
        let omega: Frequency<f64> = self.frequency * TWO_PI;
        let pressure_squared = self.pressure * self.pressure;
        let omega_squared = omega * omega;
        let sound_speed_cubed = self.sound_speed * self.sound_speed * self.sound_speed;
        let numerator = pressure_squared * omega_squared;
        let denominator = self.density * sound_speed_cubed;
        self.nonlinearity_parameter * numerator / denominator
    }

    /// Shock formation parameter (Mach number for acoustic waves)
    ///
    /// σ = (B/A)·P / (2·ρ·c²)
    /// Indicates propensity for shock formation
    #[must_use]
    pub fn shock_parameter(&self) -> Dimensionless<f64> {
        let sound_speed_squared = self.sound_speed * self.sound_speed;
        let denominator = self.density * sound_speed_squared;
        self.nonlinearity_parameter * self.pressure / denominator / 2.0
    }

    /// Is nonlinear regime significant?
    /// (σ > 1e-4 generally indicates significant nonlinear effects)
    /// This threshold is based on the acoustic nonlinearity coefficient (B/A)
    /// being about 5-8 for most tissues, combined with typical therapeutic pressures
    #[must_use]
    pub fn is_nonlinear_significant(&self) -> bool {
        self.shock_parameter().into_base() > 1e-4
    }
}
