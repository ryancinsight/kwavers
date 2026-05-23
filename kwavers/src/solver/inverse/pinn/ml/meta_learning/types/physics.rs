//! Physics parameters for meta-learning task governing equations.

use crate::core::constants::fundamental::{
    DENSITY_AIR, DENSITY_TISSUE, DENSITY_WATER_NOMINAL, SOUND_SPEED_AIR, SOUND_SPEED_TISSUE,
    SOUND_SPEED_WATER_SIM,
};

/// Physics parameters defining the task's governing equations
///
/// Different PDE types use different subsets of these parameters:
/// - **Wave/Acoustic**: `wave_speed`, `density`, `absorption`
/// - **Diffusion**: `density` (as diffusivity coefficient)
/// - **Navier-Stokes**: `density`, `viscosity`
/// - **Elastic**: `density`, `wave_speed` (as shear/longitudinal wave speeds)
#[derive(Debug, Clone)]
pub struct MetaLearningPhysicsParameters {
    /// Wave propagation speed (m/s)
    ///
    /// - Acoustic waves in air: ~343 m/s
    /// - Acoustic waves in water: ~1500 m/s
    /// - Acoustic waves in tissue: ~1540 m/s
    /// - Seismic P-waves: ~5000-8000 m/s
    pub wave_speed: f64,

    /// Material density (kg/m³)
    ///
    /// - Air: ~1.2 kg/m³
    /// - Water: ~1000 kg/m³
    /// - Soft tissue: ~1000-1100 kg/m³
    /// - Bone: ~1700-2000 kg/m³
    pub density: f64,

    /// Dynamic viscosity (Pa·s)
    ///
    /// Used for Navier-Stokes equations.
    /// - Air: ~1.8×10⁻⁵ Pa·s
    /// - Water: ~1.0×10⁻³ Pa·s
    /// - Blood: ~3-4×10⁻³ Pa·s
    pub viscosity: Option<f64>,

    /// Absorption coefficient (Np/m or dB/cm)
    ///
    /// Acoustic energy loss due to viscous friction and thermal conduction.
    /// - Air at 1 kHz: ~0.001 dB/m
    /// - Water at 1 MHz: ~0.025 dB/cm
    /// - Soft tissue at 1 MHz: ~0.5-1.0 dB/cm
    pub absorption: Option<f64>,

    /// Nonlinearity parameter (B/A or β)
    ///
    /// Characterizes nonlinear wave propagation (e.g., shock formation).
    /// - Water: B/A ≈ 5
    /// - Soft tissue: B/A ≈ 6-8
    /// - Used in Westervelt or KZK equations
    pub nonlinearity: Option<f64>,
}

impl Default for MetaLearningPhysicsParameters {
    fn default() -> Self {
        Self {
            wave_speed: SOUND_SPEED_AIR, // Speed of sound in air at 20°C
            density: DENSITY_AIR,        // Air density at 20°C, 1 atm (1.204 kg/m³)
            viscosity: None,
            absorption: None,
            nonlinearity: None,
        }
    }
}

impl MetaLearningPhysicsParameters {
    /// Create parameters for acoustic wave propagation in air
    pub fn acoustic_air() -> Self {
        Self {
            wave_speed: SOUND_SPEED_AIR,
            density: DENSITY_AIR,
            viscosity: None,
            absorption: Some(0.001),
            nonlinearity: None,
        }
    }

    /// Create parameters for acoustic wave propagation in water
    pub fn acoustic_water() -> Self {
        Self {
            wave_speed: SOUND_SPEED_WATER_SIM,
            density: DENSITY_WATER_NOMINAL,
            viscosity: None,
            absorption: Some(0.025),
            nonlinearity: Some(5.0), // B/A for water
        }
    }

    /// Create parameters for acoustic wave propagation in soft tissue
    pub fn acoustic_tissue() -> Self {
        Self {
            wave_speed: SOUND_SPEED_TISSUE,
            density: DENSITY_TISSUE,
            viscosity: None,
            absorption: Some(0.5),   // At 1 MHz
            nonlinearity: Some(7.0), // Typical B/A for tissue
        }
    }

    /// Create parameters for fluid flow (Navier-Stokes)
    pub fn fluid(density: f64, viscosity: f64) -> Self {
        Self {
            wave_speed: 0.0, // Not used for N-S
            density,
            viscosity: Some(viscosity),
            absorption: None,
            nonlinearity: None,
        }
    }
}
