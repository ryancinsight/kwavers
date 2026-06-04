//! Multi-Physics Simulation Orchestration
//!
//! This module provides the orchestration framework for coupled multi-physics simulations,
//! enabling conservative field transfer between different physics domains (acoustic, thermal,
//! optical, chemical) with proper coupling strategies.
//!
//! ## Mathematical Foundation
//!
//! Multi-physics coupling requires:
//! - **Conservative interpolation**: Preserves physical invariants across domain interfaces
//! - **Stability analysis**: Ensures coupled system remains numerically stable
//! - **Convergence acceleration**: Improves coupling iteration convergence
//! - **Load balancing**: Distributes computational work efficiently
//!
//! ## Architecture
//!
//! The multi-physics framework consists of:
//! - **MultiPhysicsFieldCoupler**: Manages conservative field transfer between physics domains
//! - **SimulationCouplingStrategy**: Defines coupling algorithms (explicit, implicit, partitioned)
//! - **SimulationMultiPhysicsSolver**: Orchestrates coupled physics simulations
//! - **DomainManager**: Handles domain decomposition and load balancing
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers_simulation::multi_physics::{SimulationMultiPhysicsSolver, SimulationCouplingStrategy};
//!
//! // Create coupled acoustic-thermal simulation
//! let config = MultiPhysicsConfig {
//!     coupling_strategy: SimulationCouplingStrategy::Implicit,
//!     max_iterations: 50,
//!     tolerance: 1e-6,
//!     relaxation_factor: 0.8,
//! };
//!
//! let solver = SimulationMultiPhysicsSolver::new(config, acoustic_solver, thermal_solver)?;
//! let result = solver.solve_coupled(&initial_conditions, time_span)?;
//! ```

pub mod conservation;
pub mod coupler;
pub mod interface;
mod residual;
pub mod residual_gas;
pub mod schwarz;
pub mod solver;

pub use conservation::MultiPhysicsConservationEnforcer;
pub use coupler::MultiPhysicsFieldCoupler;
pub use residual_gas::ResidualGasField;
pub use interface::SimulationMultiPhysicsInterface;
pub use schwarz::SchwarzCoupling;
pub use solver::SimulationMultiPhysicsSolver;

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use ndarray::{Array3, ArrayView3};

/// Physics domain types for coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimulationPhysicsDomain {
    /// Acoustic wave propagation
    Acoustic,
    /// Thermal diffusion/convection
    Thermal,
    /// Optical light transport
    Optical,
    /// Chemical reaction-diffusion
    Chemical,
    /// Elastic wave propagation
    Elastic,
    /// Electromagnetic fields
    Electromagnetic,
}

/// Coupling strategy for multi-physics simulations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationCouplingStrategy {
    /// Explicit coupling (no iteration, use previous timestep values)
    Explicit,
    /// Implicit coupling (iterative solution until convergence)
    Implicit,
    /// Partitioned coupling (solve each physics sequentially with updates)
    Partitioned,
    /// Monolithic coupling (solve all physics simultaneously)
    Monolithic,
}

/// Configuration for multi-physics coupling
#[derive(Debug, Clone)]
pub struct MultiPhysicsConfig {
    /// Coupling strategy to use
    pub coupling_strategy: SimulationCouplingStrategy,
    /// Maximum coupling iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Relaxation factor for iterative coupling (0 < omega <= 1)
    pub relaxation_factor: f64,
    /// Enable adaptive time stepping
    pub adaptive_timestep: bool,
    /// Minimum time step size
    pub min_dt: f64,
    /// Maximum time step size
    pub max_dt: f64,
}

impl Default for MultiPhysicsConfig {
    fn default() -> Self {
        Self {
            coupling_strategy: SimulationCouplingStrategy::Implicit,
            max_iterations: 20,
            tolerance: 1e-6,
            relaxation_factor: 0.8,
            adaptive_timestep: true,
            min_dt: 1e-9,
            max_dt: 1e-3,
        }
    }
}

/// Interface for physics solvers that can participate in coupling
pub trait CoupledPhysicsSolver: Send + Sync {
    /// Get the physics domain type
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn domain_type(&self) -> SimulationPhysicsDomain;

    /// Get the computational grid
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn grid(&self) -> &Grid;

    /// Get current field values
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn get_field(&self, field_name: &str) -> KwaversResult<ArrayView3<'_, f64>>;

    /// Set field values (for coupling updates)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn set_field(&mut self, field_name: &str, field: ArrayView3<f64>) -> KwaversResult<()>;

    /// Perform a single time step
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn step(&mut self, dt: f64) -> KwaversResult<()>;

    /// Get coupling source terms from this physics domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn get_coupling_source(
        &self,
        target_domain: SimulationPhysicsDomain,
    ) -> KwaversResult<Option<Array3<f64>>>;

    /// Apply coupling source terms to this physics domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_coupling_source(
        &mut self,
        source_domain: SimulationPhysicsDomain,
        source: ArrayView3<f64>,
    ) -> KwaversResult<()>;
}
