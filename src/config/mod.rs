// config/mod.rs

use crate::boundary::pml::PMLBoundary;
use crate::grid::Grid;
use crate::source::Source;
use crate::time::Time;
use log::debug;
use std::fs;

pub mod output;
pub mod simulation;
pub mod source;

pub use output::OutputConfig;
pub use simulation::SimulationConfig;
pub use source::SourceConfig;

#[derive(Debug)]
pub struct Config {
    pub simulation: SimulationConfig,
    pub source: SourceConfig,
    pub output: OutputConfig,
    grid: Option<Grid>,
    time: Option<Time>,
    source_instance: Option<Box<dyn Source>>,
    pml: Option<PMLBoundary>,
}

impl Config {
    pub fn from_file(filename: &str) -> Result<Self, String> {
        debug!("Loading config from {}", filename);
        let contents = fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read {}: {}", filename, e))?;

        let simulation: SimulationConfig = toml::from_str(&contents)
            .map_err(|e| format!("Simulation config parse error: {}", e))?;
        let source: SourceConfig =
            toml::from_str(&contents).map_err(|e| format!("Source config parse error: {}", e))?;
        let output: OutputConfig =
            toml::from_str(&contents).map_err(|e| format!("Output config parse error: {}", e))?;

        let mut config = Self {
            simulation,
            source,
            output,
            grid: None,
            time: None,
            source_instance: None,
            pml: None,
        };
        config.initialize_components()?;
        Ok(config)
    }

    fn initialize_components(&mut self) -> Result<(), String> {
        self.grid = Some(self.simulation.initialize_grid()?);
        let grid = self.grid.as_ref().unwrap();
        self.time = Some(self.simulation.initialize_time(grid)?);

        let medium = self.simulation.initialize_medium(grid);
        self.source_instance = Some(self.source.initialize_source(medium.as_ref(), grid)?);
        
        // Use the PML parameters from the simulation config
        self.pml = Some(PMLBoundary::new(
            self.simulation.pml_thickness,
            self.simulation.pml_sigma_acoustic,
            self.simulation.pml_sigma_light,
            medium.as_ref(),
            grid,
            self.simulation.frequency,
            Some(self.simulation.pml_polynomial_order),
            Some(self.simulation.pml_reflection),
        ));

        Ok(())
    }

    pub fn grid(&self) -> &Grid {
        self.grid.as_ref().unwrap()
    }
    pub fn time(&self) -> &Time {
        self.time.as_ref().unwrap()
    }
    pub fn source(&self) -> &Box<dyn Source> {
        self.source_instance.as_ref().unwrap()
    }
    pub fn pml(&self) -> &PMLBoundary {
        self.pml.as_ref().unwrap()
    }
}
