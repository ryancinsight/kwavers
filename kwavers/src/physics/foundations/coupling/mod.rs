//! Multi-Physics Coupling Bounded Context
//!
//! Provides trait-based abstractions for coupling between different physics domains:
//! - Acoustic-Elastic interfaces
//! - Acoustic-Thermal (thermoacoustic) effects
//! - Electromagnetic-Acoustic (photoacoustic) coupling
//! - Electromagnetic-Thermal (photothermal) effects
//! - Domain decomposition methods
//!
//! ## Mathematical Foundation
//!
//! For coupled systems of PDEs:
//! ```text
//! ∂u/∂t = L₁[u] + C₁₂[v]     in Ω₁  (Physics 1)
//! ∂v/∂t = L₂[v] + C₂₁[u]     in Ω₂  (Physics 2)
//! ```
//!
//! Conservation properties: dE/dt = 0, dP/dt = 0, dM/dt = 0

pub mod acoustic_elastic;
pub mod acoustic_thermal;
pub mod domain_decomposition;
pub mod electromagnetic_acoustic;
pub mod electromagnetic_thermal;

#[cfg(test)]
mod tests;

// Re-export all sub-module traits and types for backward compatibility
pub use acoustic_elastic::AcousticElasticCoupling;
pub use acoustic_thermal::AcousticThermalCoupling;
pub use domain_decomposition::{DomainDecomposition, SchwarzMethod, TransmissionCondition};
pub use electromagnetic_acoustic::ElectromagneticAcousticCoupling;
pub use electromagnetic_thermal::ElectromagneticThermalCoupling;

use std::fmt::Debug;

/// Coupling strength between physics domains
#[derive(Debug, Clone)]
pub struct CouplingStrength {
    /// Spatial coupling coefficient (dimensionless or with units)
    pub spatial_coefficient: f64,
    /// Temporal coupling coefficient (1/s)
    pub temporal_coefficient: f64,
    /// Energy transfer efficiency (dimensionless)
    pub energy_efficiency: f64,
}

/// Interface condition type
#[derive(Debug, Clone)]
pub enum InterfaceCondition {
    /// Dirichlet-type: field continuity u₁ = u₂
    Dirichlet { field_name: String },
    /// Neumann-type: flux continuity ∂u₁/∂n = ∂u₂/∂n
    Neumann { flux_name: String },
    /// Robin-type: weighted combination αu + β∂u/∂n = γu₂ + δ∂u₂/∂n
    Robin {
        alpha: f64,
        beta: f64,
        gamma: f64,
        delta: f64,
    },
    /// Transmission condition for wave propagation
    Transmission { impedance_ratio: f64 },
}

/// Multi-physics coupling trait
pub trait MultiPhysicsCoupling: Send + Sync {
    /// Get coupling strength between domains
    fn coupling_strength(&self) -> CouplingStrength;

    /// Get interface conditions for this coupling
    fn interface_conditions(&self) -> Vec<InterfaceCondition>;

    /// Compute energy transfer rate between domains (W/m³)
    fn energy_transfer_rate(&self, interface_position: &[f64]) -> f64;

    /// Check stability criteria for coupled time stepping
    fn stability_criteria(&self, dt: f64) -> Result<(), String>;

    /// Apply coupling at interface
    fn apply_coupling(&mut self, dt: f64) -> Result<(), String>;
}
