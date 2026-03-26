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
    /// v_stream ~ I / (ρ·c)²
    /// Derived from radiation stress tensor
    #[must_use]
    pub fn velocity(&self) -> f64 {
        self.intensity / ((self.density * self.sound_speed).powi(2))
    }

    /// Streaming power (energy flux) [W/m³]
    #[must_use]
    pub fn power(&self) -> f64 {
        self.intensity.powi(2) / (self.density * self.sound_speed)
    }

    /// Enhance thermal mixing (effective diffusivity increase)
    ///
    /// Acoustic streaming can enhance thermal transport.
    /// Effective diffusivity increase: Δα ~ (L·v_stream)²/t_acoustic
    /// where L is characteristic length scale
    #[must_use]
    pub fn enhanced_diffusivity(&self, characteristic_length: f64) -> f64 {
        let v = self.velocity();
        let period = 1.0 / 1e6; // Assume 1 MHz frequency
        (characteristic_length * v).powi(2) / period
    }
}
