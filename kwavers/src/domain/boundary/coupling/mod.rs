//! Advanced Boundary Conditions for Multi-Physics Coupling
//!
//! This module provides sophisticated boundary condition types that enable
//! seamless coupling between different physics domains.
//!
//! ## Boundary Condition Categories
//!
//! ### Interface Boundaries
//! - **MaterialInterface**: Handles discontinuities between different materials
//! - **MultiPhysicsInterface**: Couples different physics (EM-acoustic, acoustic-elastic)
//!
//! ### Advanced Absorbing Boundaries
//! - **ImpedanceBoundary**: Frequency-dependent absorption
//! - **AdaptiveBoundary**: Dynamically adjusts absorption based on field energy
//!
//! ### Coupling Boundaries
//! - **SchwarzBoundary**: Domain decomposition coupling (✅ Neumann & Robin implemented)
//!
//! ## Schwarz Domain Decomposition (Sprint 210 Phase 1)
//!
//! The `SchwarzBoundary` type implements overlapping domain decomposition with
//! four transmission conditions:
//!
//! ### Dirichlet Transmission
//! Direct value copying: `u_interface = u_neighbor`
//!
//! ### Neumann Transmission (✅ Implemented)
//! Flux continuity: `∂u₁/∂n = ∂u₂/∂n`
//! - Uses centered finite differences for gradient computation
//! - Applies correction to match fluxes across interface
//! - Validated: gradient preservation, conservation, matching
//!
//! ### Robin Transmission (✅ Implemented)
//! Coupled condition: `∂u/∂n + αu = β`
//! - Combines field value and gradient (convection, impedance)
//! - Stable blending of interface, neighbor, and Robin contributions
//! - Validated: parameter sweep, stability, edge cases
//!
//! ### Optimized Schwarz
//! Relaxation-based: `u_new = (1-θ)u_old + θ·u_neighbor`
//!
//! ## Mathematical Foundations
//!
//! ### Gradient Computation
//! ```text
//! Interior: ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2Δx)    [O(Δx²)]
//! Boundary: ∂u/∂x ≈ (u[i+1] - u[i]) / Δx          [O(Δx)]
//! ```
//!
//! ### Energy Conservation
//! For lossless interfaces: `|R|² + |T|² = 1`
//!
//! ## Example Usage
//!
//! ```no_run
//! use kwavers::domain::boundary::coupling::{
//!     MaterialInterface,
//!     ImpedanceBoundary,
//!     AdaptiveBoundary,
//!     MultiPhysicsInterface,
//!     SchwarzBoundary,
//!     PhysicsDomain,
//!     CouplingType,
//!     TransmissionCondition,
//! };
//! use kwavers::domain::boundary::traits::BoundaryDirections;
//! use kwavers::domain::medium::properties::AcousticPropertyData;
//!
//! // Material interface example
//! let water = AcousticPropertyData {
//!     density: 1000.0,
//!     sound_speed: 1500.0,
//!     absorption_coefficient: 0.002,
//!     absorption_power: 2.0,
//!     nonlinearity: 5.0,
//! };
//!
//! let tissue = AcousticPropertyData {
//!     density: 1050.0,
//!     sound_speed: 1540.0,
//!     absorption_coefficient: 0.5,
//!     absorption_power: 1.1,
//!     nonlinearity: 6.5,
//! };
//!
//! let material_bc = MaterialInterface::new(
//!     [0.05, 0.0, 0.0],
//!     [1.0, 0.0, 0.0],
//!     water,
//!     tissue,
//!     0.001,
//! );
//!
//! // Schwarz domain decomposition example
//! let schwarz = SchwarzBoundary::new(0.01, BoundaryDirections::all())
//!     .with_transmission_condition(TransmissionCondition::Neumann);
//! ```
//!
//! ## References
//!
//! - Schwarz, H.A. (1870). "Über einen Grenzübergang durch alternierendes Verfahren"
//! - Dolean, V., et al. (2015). "An Introduction to Domain Decomposition Methods"
//! - Quarteroni, A. & Valli, A. (1999). "Domain Decomposition Methods for PDEs"
//! - Kinsler et al., *Fundamentals of Acoustics* (4th ed.), Chapter 5
//! - Hamilton & Blackstock, *Nonlinear Acoustics* (1998), Chapter 2

// Module declarations
pub mod adaptive;
pub mod impedance;
pub mod material;
pub mod multiphysics;
pub mod schwarz;
pub mod types;

// Re-export main types for convenient access
pub use adaptive::AdaptiveBoundary;
pub use impedance::ImpedanceBoundary;
pub use material::MaterialInterface;
pub use multiphysics::MultiPhysicsInterface;
pub use schwarz::SchwarzBoundary;

// Re-export shared types from types module
pub use types::{CouplingType, FrequencyProfile, PhysicsDomain, TransmissionCondition};
