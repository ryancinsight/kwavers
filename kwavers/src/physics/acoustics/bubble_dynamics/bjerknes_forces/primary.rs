//! Primary Bjerknes force calculation

use super::calculator::BjerknesCalculator;
use crate::core::error::{KwaversError, KwaversResult};

impl BjerknesCalculator {
    /// Calculate primary Bjerknes force on a bubble in acoustic field
    ///
    /// The primary Bjerknes force arises from the time-averaged radiation pressure:
    /// F_p = -4πR² ⟨p'(∂u'/∂z)⟩
    ///
    /// For a plane wave: F_p ≈ πR² I / c₀
    /// where I is acoustic intensity
    pub fn primary_bjerknes_force(
        &self,
        bubble_radius: f64,
        acoustic_pressure_amplitude: f64,
        pressure_gradient: f64,
    ) -> KwaversResult<f64> {
        if bubble_radius <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radius must be positive".to_string(),
            ));
        }

        // Primary Bjerknes force for acoustic wave
        // F_p = -4πR² ⟨p'(∂u'/∂z)⟩
        // Approximation: F_p ≈ -4πR² (p₀/(2ρc₀)) (∂p/∂z)
        // where the time-averaged acoustic intensity contribution is proportional to pressure amplitude and gradient

        let surface_area = 4.0 * std::f64::consts::PI * bubble_radius.powi(2);

        // Acoustic radiation pressure contribution
        let radiation_pressure =
            acoustic_pressure_amplitude.powi(2) / (2.0 * self.config.rho * self.config.c0);

        // Force from pressure gradient
        let force = -surface_area * radiation_pressure * (pressure_gradient / self.config.c0);

        Ok(force)
    }
}
