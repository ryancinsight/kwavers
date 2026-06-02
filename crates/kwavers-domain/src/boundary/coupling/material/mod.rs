//! Material interface boundary condition
//!
//! Handles wave propagation across material discontinuities with proper
//! transmission and reflection coefficients based on acoustic impedance mismatch.

mod boundary_impl;
#[cfg(test)]
mod tests;

use crate::medium::properties::AcousticPropertyData;

/// Material interface boundary condition for wave reflection and transmission.
///
/// # Physics
///
/// At a material interface:
/// ```text
/// R = (Z₂ - Z₁) / (Z₂ + Z₁)    (reflection coefficient for pressure)
/// T = 2Z₂ / (Z₂ + Z₁)          (transmission coefficient for pressure)
/// ```
///
/// Energy conservation: |R|² + (Z₁/Z₂)|T|² = 1
///
/// # References
///
/// - Kinsler et al., *Fundamentals of Acoustics* (4th ed.), Chapter 5
/// - Hamilton & Blackstock, *Nonlinear Acoustics* (1998), Chapter 2
#[derive(Debug, Clone)]
pub struct MaterialInterface {
    pub position: [f64; 3],
    pub normal: [f64; 3],
    pub material_1: AcousticPropertyData,
    pub material_2: AcousticPropertyData,
    pub thickness: f64,
}

impl MaterialInterface {
    #[must_use]
    pub fn new(
        position: [f64; 3],
        normal: [f64; 3],
        material_1: AcousticPropertyData,
        material_2: AcousticPropertyData,
        thickness: f64,
    ) -> Self {
        Self {
            position,
            normal,
            material_1,
            material_2,
            thickness,
        }
    }

    /// Compute reflection coefficient R = (Z2 - Z1)/(Z2 + Z1)
    #[must_use]
    pub fn reflection_coefficient(&self) -> f64 {
        let z1 = self.material_1.impedance();
        let z2 = self.material_2.impedance();
        (z2 - z1) / (z2 + z1)
    }

    /// Compute transmission coefficient T = 2Z2/(Z1 + Z2)
    #[must_use]
    pub fn transmission_coefficient(&self) -> f64 {
        let z1 = self.material_1.impedance();
        let z2 = self.material_2.impedance();
        2.0 * z2 / (z1 + z2)
    }

    #[must_use]
    pub fn transmitted_pressure(&self, incident_pressure: f64) -> f64 {
        incident_pressure * self.transmission_coefficient()
    }

    #[must_use]
    pub fn reflected_pressure(&self, incident_pressure: f64) -> f64 {
        incident_pressure * self.reflection_coefficient()
    }
}
