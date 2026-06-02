//! Acoustic streaming effects
//!
//! Acoustic streaming is the steady flow of fluid induced by acoustic waves.
//! It contributes to heat mixing and can enhance thermal effects.
//!
//! Reference: Nyborg (1988) "Acoustic streaming in ultrasonic therapy"

/// Acoustic streaming effects
#[derive(Debug, Clone, Copy)]
pub struct AcousticStreaming {
    /// Acoustic intensity [W/m²]
    pub intensity: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Fluid density [kg/m³]
    pub density: f64,
}

impl AcousticStreaming {
    /// Create streaming effect
    #[must_use]
    pub fn new(intensity: f64, sound_speed: f64, density: f64) -> Self {
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
    pub fn velocity(&self) -> f64 {
        self.intensity / (self.density * self.sound_speed.powi(2))
    }

    /// Streaming power flux [W/m²]
    ///
    /// P_stream = v_stream · (I/c) = I² / (ρ·c³)
    #[must_use]
    pub fn power(&self) -> f64 {
        self.intensity.powi(2) / (self.density * self.sound_speed.powi(3))
    }
}
