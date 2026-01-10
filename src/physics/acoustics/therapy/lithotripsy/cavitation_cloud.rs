//! Cavitation cloud dynamics for lithotripsy simulation.
//!
//! This module implements bubble cloud formation, growth, and collapse dynamics
//! relevant to shock wave lithotripsy, where cavitation plays a key role in
//! stone erosion and tissue bioeffects.

/// Parameters for cavitation cloud dynamics.
#[derive(Debug, Clone)]
pub struct CloudParameters {
    /// Initial bubble radius (m)
    pub initial_bubble_radius: f64,
    /// Bubble number density (#/m³)
    pub bubble_density: f64,
    /// Ambient pressure (Pa)
    pub ambient_pressure: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
    /// Viscosity (Pa·s)
    pub viscosity: f64,
}

impl Default for CloudParameters {
    fn default() -> Self {
        Self {
            initial_bubble_radius: 1e-6, // 1 micron
            bubble_density: 1e12,        // 10^12 bubbles/m³
            ambient_pressure: 101325.0,  // 1 atm
            surface_tension: 0.0728,     // Water at 20°C
            viscosity: 1e-3,             // Water at 20°C
        }
    }
}

/// Cavitation cloud dynamics model.
#[derive(Debug, Clone)]
pub struct CavitationCloudDynamics {
    /// Cloud parameters
    parameters: CloudParameters,
}

impl CavitationCloudDynamics {
    /// Create new cavitation cloud dynamics model.
    pub fn new(parameters: CloudParameters) -> Self {
        Self { parameters }
    }

    /// Get cloud parameters.
    pub fn parameters(&self) -> &CloudParameters {
        &self.parameters
    }
}

impl Default for CavitationCloudDynamics {
    fn default() -> Self {
        Self::new(CloudParameters::default())
    }
}
