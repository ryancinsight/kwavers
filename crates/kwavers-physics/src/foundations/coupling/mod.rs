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
//! ∂u/∂t = L₁[u] + C₁₂`V`     in Ω₁  (Physics 1)
//! ∂v/∂t = L₂`V` + C₂₁[u]     in Ω₂  (Physics 2)
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
pub use domain_decomposition::{
    DomainDecompTransmissionCondition, DomainDecomposition, SchwarzMethod,
};
pub use electromagnetic_acoustic::ElectromagneticAcousticCoupling;
pub use electromagnetic_thermal::ElectromagneticThermalCoupling;

use aequitas::systems::si::quantities::{
    Dimensionless, Length, ReciprocalTime, Time, VolumetricPowerDensity,
};

/// Position of a sample in the coupling geometry.
pub type CouplingPosition = [Length; 3];

/// Coupling strength between physics domains
#[derive(Debug, Clone)]
pub struct CouplingStrength {
    /// Dimensionless spatial weight.
    pub spatial_weight: Dimensionless,
    /// Temporal coupling coefficient (1/s)
    pub temporal_coefficient: ReciprocalTime,
    /// Energy transfer efficiency (dimensionless)
    pub energy_efficiency: Dimensionless,
}

/// Interface condition type
#[derive(Debug, Clone)]
pub enum CouplingInterfaceCondition {
    /// Dirichlet-type: field continuity u₁ = u₂
    Dirichlet {
        /// Name of the field whose continuity is enforced across the interface.
        field_name: String,
    },
    /// Neumann-type: flux continuity ∂u₁/∂n = ∂u₂/∂n
    Neumann {
        /// Name of the flux quantity whose continuity is enforced.
        flux_name: String,
    },
    /// Robin-type: weighted combination αu + β∂u/∂n = γu₂ + δ∂u₂/∂n
    Robin {
        /// Coefficient α on the field value of side 1.
        alpha: Dimensionless,
        /// Coefficient β on the normal derivative of side 1.
        beta: Length,
        /// Coefficient γ on the field value of side 2.
        gamma: Dimensionless,
        /// Coefficient δ on the normal derivative of side 2.
        delta: Length,
    },
    /// Transmission condition for wave propagation
    Transmission {
        /// Acoustic impedance ratio Z₂/Z₁ across the interface.
        impedance_ratio: Dimensionless,
    },
}

/// Multi-physics coupling in physical coordinates.
///
/// A point in the coupling geometry has three length components.
/// ```compile_fail,E0308
/// use aequitas::systems::si::quantities::Time;
/// use kwavers_physics::foundations::coupling::{CouplingPosition, MultiPhysicsCoupling};
/// fn use_time_as_power(coupling: &impl MultiPhysicsCoupling, point: &CouplingPosition) {
///     let _: Time = coupling.energy_transfer_rate(point);
/// }
/// ```
pub trait MultiPhysicsCoupling: Send + Sync {
    /// Get coupling strength between domains
    fn coupling_strength(&self) -> CouplingStrength;

    /// Get interface conditions for this coupling
    fn interface_conditions(&self) -> Vec<CouplingInterfaceCondition>;

    /// Compute energy transfer rate between domains (W/m³)
    fn energy_transfer_rate(&self, interface_position: &CouplingPosition)
        -> VolumetricPowerDensity;

    /// Check stability criteria for coupled time stepping
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn stability_criteria(&self, dt: Time) -> Result<(), String>;

    /// Apply coupling at interface
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_coupling(&mut self, dt: Time) -> Result<(), String>;
}
