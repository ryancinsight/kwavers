//! Physics Foundations - Wave Equation Specifications
//!
//! This module contains the canonical trait definitions for all wave physics
//! in the Kwavers system. These traits define the mathematical structure of
//! wave propagation PDEs without committing to specific numerical methods.
//!
//! # Design Principle: Specification vs Implementation
//!
//! The foundations module separates **what the physics is** (specifications)
//! from **how to solve it** (implementations). This enables:
//!
//! - Forward solvers (FDTD, PSTD, FEM) and inverse solvers (PINN, optimization)
//!   to share the same physics specifications
//! - Validation logic to be reused across different solver types
//! - Material properties and boundary conditions to be solver-agnostic
//! - Easy addition of new numerical methods without changing physics definitions
//!
//! # Architecture
//!
//! ```text
//! physics/foundations/          ← Physics specifications (THIS MODULE)
//! ├── wave_equation.rs          ← Core wave equation traits
//! └── coupling.rs               ← Multi-physics coupling interfaces
//!
//! physics/acoustics/            ← Acoustic wave implementations
//! physics/electromagnetic/      ← EM wave implementations
//! physics/nonlinear/            ← Nonlinear physics implementations
//!
//! solver/forward/               ← Numerical method implementations
//! solver/inverse/               ← Inverse problem solvers
//! ```
//!
//! # Mathematical Foundation
//!
//! All wave equations in this system are second-order hyperbolic PDEs:
//!
//! ```text
//! ∂²u/∂t² = L[u] + f(x,t)
//! ```
//!
//! where:
//! - `u`: Field variable (scalar pressure, vector displacement, tensor fields)
//! - `L`: Spatial differential operator (depends on wave type and material)
//! - `f`: Source term (external forcing, boundary conditions)
//!
//! Different wave types specialize this general form:
//! - **Acoustic**: `u = p` (pressure), `L = c²∇²` (Helmholtz operator)
//! - **Elastic**: `u = d⃗` (displacement), `L = (λ+μ)∇(∇·u) + μ∇²u` (Navier)
//! - **Electromagnetic**: Coupled `E`, `H` fields with Maxwell's equations
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::physics::foundations::{
//!     WaveEquation, AcousticWaveEquation, BoundaryCondition, Domain
//! };
//!
//! // Define physics-based validation that works for ANY solver
//! fn validate_wave_solver<S: WaveEquation>(solver: &S) -> Result<(), String> {
//!     let domain = solver.domain();
//!     let dt = solver.cfl_timestep();
//!
//!     // Physics validation is solver-agnostic
//!     if dt <= 0.0 {
//!         return Err("CFL timestep must be positive".to_string());
//!     }
//!
//!     Ok(())
//! }
//!
//! // Works for FDTD, PSTD, PINN, or any other solver implementation
//! validate_wave_solver(&fdtd_solver)?;
//! validate_wave_solver(&pinn_solver)?;
//! ```
//!
//! # Modules
//!
//! - [`wave_equation`]: Core wave equation trait definitions
//! - [`coupling`]: Multi-physics coupling interfaces and traits

pub mod coupling;
pub mod wave_equation;

// Re-export commonly used wave equation types for convenience
pub use wave_equation::{
    AcousticWaveEquation, AutodiffElasticWaveEquation, AutodiffWaveEquation, BoundaryCondition,
    Domain, ElasticWaveEquation, SourceTerm, SpatialDimension, TimeIntegration, WaveEquation,
};

// Re-export coupling traits and types
pub use coupling::{
    AcousticElasticCoupling, AcousticThermalCoupling, CouplingStrength,
    ElectromagneticAcousticCoupling, ElectromagneticThermalCoupling, InterfaceCondition,
    MultiPhysicsCoupling, SchwarzMethod, TransmissionCondition,
};

/// Prelude module for convenient imports
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::physics::foundations::prelude::*;
///
/// // Now have access to all core physics traits and types
/// let domain = Domain::new_1d(0.0, 1.0, 101, BoundaryCondition::Periodic);
/// ```
pub mod prelude {
    pub use super::coupling::{
        AcousticElasticCoupling, AcousticThermalCoupling, CouplingStrength,
        ElectromagneticAcousticCoupling, ElectromagneticThermalCoupling, InterfaceCondition,
        MultiPhysicsCoupling,
    };
    pub use super::wave_equation::{
        AcousticWaveEquation, AutodiffElasticWaveEquation, AutodiffWaveEquation, BoundaryCondition,
        Domain, ElasticWaveEquation, SourceTerm, SpatialDimension, TimeIntegration, WaveEquation,
    };
}
