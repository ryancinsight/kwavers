//! Simulation-layer entry point for building a populated plugin manager.
//!
//! This module is a thin façade over [`crate::physics::factory::PhysicsCatalog`].
//! Capability types ([`PhysicsConfig`], [`PhysicsModelConfig`],
//! [`PhysicsModelType`]) live in `crate::physics::factory` (SSOT) and are
//! re-exported here so callers in the simulation layer can keep a single
//! `crate::simulation::factory` import surface.

pub use crate::physics::factory::{
    AcousticSolver, BoundaryType, BubbleModel, NonlinearEquation, PhysicsCatalog, PhysicsConfig,
    PhysicsModelConfig, PhysicsModelType,
};

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::solver::plugin::PluginManager;

/// Simulation-layer factory entry point.
///
/// Wraps [`PhysicsCatalog`] to provide the canonical
/// `Configuration → PluginManager` construction path used by
/// [`crate::simulation::manager::PhysicsManager`].
#[derive(Debug)]
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Build a plugin manager from validated configuration and runtime context.
    ///
    /// `dt` is the global integrator timestep. `medium` and `grid` are
    /// borrowed only during plugin construction.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create_physics(
        config: &PhysicsConfig,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<PluginManager> {
        PhysicsCatalog::build(config, grid, medium, dt)
    }
}
