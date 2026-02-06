//! Simulation setup orchestration
//!
//! Replaces legacy factory system. Orchestrates the creation of simulation components
//! from configuration (SSOT).

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::{Medium, MediumBuilder};
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::{Source, SourceFactory};
use crate::simulation::configuration::Configuration;
use std::sync::Arc;

/// Container for simulation components
///
/// Owns the components created during setup.
pub struct SimulationComponents {
    pub grid: Grid,
    pub medium: Box<dyn Medium>,
    pub sources: Vec<Arc<dyn Source>>,
    pub sensors: Vec<GridSensorSet>,
}

impl std::fmt::Debug for SimulationComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulationComponents")
            .field("grid", &self.grid)
            .field("medium_type", &std::any::type_name_of_val(&*self.medium))
            .field("sources_len", &self.sources.len())
            .field("sensors_len", &self.sensors.len())
            .finish()
    }
}

/// Simulation setup orchestrator
#[derive(Debug)]
pub struct SimulationSetup;

impl SimulationSetup {
    /// Create simulation components from configuration
    pub fn setup(config: &Configuration) -> KwaversResult<SimulationComponents> {
        // 1. Create Grid
        // GridParameters has spacing, dimensions.
        // Grid::new(nx, ny, nz, dx, dy, dz)
        let grid = Grid::new(
            config.grid.dimensions[0],
            config.grid.dimensions[1],
            config.grid.dimensions[2],
            config.grid.spacing[0],
            config.grid.spacing[1],
            config.grid.spacing[2],
        )?;

        // 2. Create Medium
        let medium = MediumBuilder::build(&config.medium, &grid)?;

        // 3. Create Sources
        // Configuration likely has ONE source config? "pub source: SourceParameters".
        // What if multiple sources? Support multiple sources in config later.
        // For now, crate creates one source from params.
        // We wrap it in Arc.
        let source_impl = SourceFactory::create_source(&config.source, &grid)?;
        let sources = vec![Arc::from(source_impl)];

        // 4. Create Sensors
        // Not currently in Configuration top-level?
        // Wait, Configuration struct (Step 79) does NOT have `sensors`.
        // It has `output`, `pml` (boundary), etc.
        // If sensors are not in config, we initialize empty or default.
        let sensors = Vec::new();

        Ok(SimulationComponents {
            grid,
            medium,
            sources,
            sensors,
        })
    }

    // Helper to create CoreSimulation from components if needed?
    // CoreSimulation takes reference to Medium.
    // User code flow:
    // let config = ...
    // let components = SimulationSetup::setup(&config)?;
    // let simulation = CoreSimulation::new(components.grid.clone(), &*components.medium, components.sources, ...)?;
}
