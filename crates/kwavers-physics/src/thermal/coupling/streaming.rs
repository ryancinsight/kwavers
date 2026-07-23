//! Acoustic streaming effects
//!
//! Acoustic streaming is the steady flow of fluid induced by acoustic waves.
//! It contributes to heat mixing and can enhance thermal effects.
//!
//! Reference: Nyborg (1988) "Acoustic streaming in ultrasonic therapy"

use aequitas::systems::si::quantities::{
    AcousticImpedance, Intensity, MassDensity, Pressure, Velocity,
};

/// Acoustic streaming effects
#[derive(Debug, Clone, Copy)]
pub struct AcousticStreaming {
    /// Acoustic intensity [W/m²]
    pub intensity: Intensity<f64>,
    /// Sound speed [m/s]
    pub sound_speed: Velocity<f64>,
    /// Fluid density [kg/m³]
    pub density: MassDensity<f64>,
}

impl AcousticStreaming {
    /// Create streaming effect
    #[must_use]
    pub const fn new(
        intensity: Intensity<f64>,
        sound_speed: Velocity<f64>,
        density: MassDensity<f64>,
    ) -> Self {
        Self {
            intensity,
            sound_speed,
            density,
        }
    }

    /// Streaming velocity [m/s]
    ///
    /// v_stream ~ I / (ρ·c²)
    /// Derived from radiation body force I/c divided by acoustic impedance ρ·c
    #[must_use]
    pub fn velocity(&self) -> Velocity<f64> {
        let impedance: AcousticImpedance<f64> = self.density * self.sound_speed;
        let radiation_pressure: Pressure<f64> = self.intensity / self.sound_speed;
        radiation_pressure / impedance
    }

    /// Streaming power flux [W/m²]
    ///
    /// P_stream = v_stream · (I/c) = I² / (ρ·c³)
    #[must_use]
    pub fn power(&self) -> Intensity<f64> {
        let radiation_pressure: Pressure<f64> = self.intensity / self.sound_speed;
        self.velocity() * radiation_pressure
    }
}
