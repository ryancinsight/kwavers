//! Physics manager — façade for the capability catalog.
//!
//! Delegates to [`kwavers_solver::plugin::PhysicsCatalog`] which performs
//! the concrete `PhysicsConfig → PluginManager` translation. This shell
//! preserves the GRASP "Manager" entry point used elsewhere in the
//! simulation layer.

use super::factory::PhysicsConfig;
use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_solver::plugin::{PhysicsCatalog, PluginManager};

/// Specialized physics manager following the Manager pattern from GRASP.
#[derive(Debug)]
pub struct PhysicsManager;

impl PhysicsManager {
    /// Build plugin manager from validated configuration and runtime context.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn build(
        config: &PhysicsConfig,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<PluginManager> {
        PhysicsCatalog::build(config, grid, medium, dt)
    }
}
