//! Domain-specific boundary condition type enums (acoustic, EM, elastic).

use serde::{Deserialize, Serialize};

/// Acoustic-specific boundary condition types.
///
/// ## Mathematical Specifications
///
/// - **SoundSoft**: `p = 0` (pressure release)
/// - **SoundHard**: `∂p/∂n = 0` (rigid wall, zero normal velocity)
/// - **Impedance**: `Z·(∂p/∂n) + p = 0` where Z = ρc
/// - **Absorbing**: PML or ABC for non-reflecting boundaries
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DomainAcousticBoundaryType {
    /// Sound-soft boundary: `p = 0` (pressure release)
    SoundSoft,
    /// Sound-hard boundary: `∂p/∂n = 0` (rigid wall)
    SoundHard,
    /// Impedance boundary: `Z·(∂p/∂n) + p = 0`
    Impedance {
        /// Acoustic impedance (kg/m²s)
        impedance: f64,
    },
    /// Absorbing boundary (PML or ABC)
    Absorbing,
    /// Radiation condition (far-field)
    Radiation,
}

/// Electromagnetic-specific boundary condition types.
///
/// ## Mathematical Specifications
///
/// - **PEC**: `n × E = 0` (tangential E vanishes)
/// - **PMC**: `n × H = 0` (tangential H vanishes)
/// - **Absorbing**: PML or ABC for wave absorption
/// - **Periodic**: Bloch-periodic boundaries for photonic crystals
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ElectromagneticBoundaryType {
    /// Perfect Electric Conductor: `n × E = 0`
    PerfectElectricConductor,
    /// Perfect Magnetic Conductor: `n × H = 0`
    PerfectMagneticConductor,
    /// Absorbing boundary (PML or ABC)
    Absorbing,
    /// Periodic boundary with Bloch phase
    Periodic {
        /// Bloch wave vector k (rad/m)
        k_bloch: [f64; 3],
    },
    /// Impedance boundary for EM waves (surface impedance, Ω)
    Impedance {
        /// Surface impedance (Ω)
        impedance: f64,
    },
}

/// Elastic-specific boundary condition types.
///
/// ## Mathematical Specifications
///
/// - **Clamped**: `u = 0` (zero displacement)
/// - **Free**: `σ·n = 0` (zero traction/stress)
/// - **Roller**: `u·n = 0, (σ·n)×n = 0` (tangential slip allowed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElasticBoundaryType {
    /// Clamped boundary: `u = 0` (zero displacement)
    Clamped,
    /// Free boundary: `σ·n = 0` (zero traction)
    Free,
    /// Roller boundary: tangential slip allowed, normal fixed
    Roller,
    /// Absorbing boundary (PML or ABC)
    Absorbing,
    /// Periodic boundary
    Periodic,
}
