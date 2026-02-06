//! Simulation control module
//!
//! This module provides the core simulation control functionality that
//! orchestrates solvers, sources, sensor probe sets, and medium to run complete simulations.
//! This module supports building a simulation orchestration layer.
//!
//! # Sensor model
//! Kwavers supports multi-physics (acoustics + optics). The canonical high-level
//! sensor representation is therefore a grid probe set (`GridSensorSet`) rather than
//! a domain-specific array sensor.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::Source;
use crate::solver::config::{SolverConfiguration, SolverType};
use crate::solver::factory::SolverFactory;
use crate::solver::feature::{FeatureManager, SolverFeature};
use crate::solver::interface::Solver;
use crate::solver::progress::{FieldsSummary, ProgressReporter, ProgressUpdate};
use std::sync::Arc;

/// Core simulation controller
pub struct CoreSimulation<'a, M: Medium> {
    grid: Grid,
    medium: &'a M,
    sources: Vec<Arc<dyn Source>>,
    /// High-level sensor probe sets (grid-indexed), supporting multi-physics (acoustics + optics).
    sensors: Vec<GridSensorSet>,
    feature_manager: FeatureManager,
    progress_reporter: Box<dyn ProgressReporter>,
    /// Solver configuration (determines which solver to use and how)
    solver_config: SolverConfiguration,
    /// The instantiated solver (created on first run or explicit initialize)
    solver: Option<Box<dyn Solver>>,
}

// Debug implementation for CoreSimulation
impl<'a, M: Medium + std::fmt::Debug> std::fmt::Debug for CoreSimulation<'a, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreSimulation")
            .field("grid", &self.grid)
            .field("medium", &self.medium)
            .field("sources", &self.sources.len())
            .field("sensors", &self.sensors.len())
            .field("feature_manager", &self.feature_manager)
            .field("solver_config", &self.solver_config)
            .field("solver_initialized", &self.solver.is_some())
            .finish()
    }
}

impl<'a, M: Medium> CoreSimulation<'a, M> {
    /// Create a new core simulation
    pub fn new(
        grid: Grid,
        medium: &'a M,
        sources: Vec<Arc<dyn Source>>,
        sensors: Vec<GridSensorSet>,
        progress_reporter: Box<dyn ProgressReporter>,
    ) -> KwaversResult<Self> {
        let feature_manager = FeatureManager::new();

        Ok(Self {
            grid,
            medium,
            sources,
            sensors,
            feature_manager,
            progress_reporter,
            solver_config: SolverConfiguration::default(),
            solver: None,
        })
    }

    /// Set the solver configuration (call before run)
    pub fn set_solver_config(&mut self, config: SolverConfiguration) {
        self.solver_config = config;
    }

    /// Initialize the simulation — creates the solver and prepares for execution
    pub fn initialize(&mut self) -> KwaversResult<()> {
        self.progress_reporter.on_start(0, 0.0);

        // Create the solver via factory
        let mut solver = SolverFactory::create_solver(
            self.solver_config.solver_type,
            self.solver_config.clone(),
            &self.grid,
            self.medium,
        )?;

        // Initialize the solver with grid and medium
        solver.initialize(&self.grid, self.medium)?;

        // Add sensors to the solver
        for sensor in &self.sensors {
            solver.add_sensor(sensor)?;
        }

        // Enable configured features
        if self.feature_manager.is_enabled(SolverFeature::Reconstruction) {
            let _ = solver.enable_feature(SolverFeature::Reconstruction, true);
        }
        if self.feature_manager.is_enabled(SolverFeature::GpuAcceleration) {
            let _ = solver.enable_feature(SolverFeature::GpuAcceleration, true);
        }

        self.solver = Some(solver);
        Ok(())
    }

    /// Run the simulation using the configured solver
    pub fn run(&mut self, num_steps: usize, dt: f64) -> KwaversResult<SimulationResult> {
        // Auto-initialize if not already done
        if self.solver.is_none() {
            // Update dt in config before initialization
            self.solver_config.dt = dt;
            self.solver_config.max_steps = num_steps;
            self.initialize()?;
        }

        let solver = self.solver.as_mut().ok_or_else(|| {
            crate::core::error::KwaversError::InternalError(
                "Solver not initialized".to_string(),
            )
        })?;

        self.progress_reporter.on_start(num_steps, dt);
        let start_time = std::time::Instant::now();

        // Run the solver for the requested number of steps
        solver.run(num_steps)?;

        let elapsed = start_time.elapsed();
        let stats = solver.statistics();

        // Report final progress
        let progress = ProgressUpdate {
            current_step: num_steps,
            total_steps: num_steps,
            current_time: num_steps as f64 * dt,
            total_time: num_steps as f64 * dt,
            step_duration: elapsed / num_steps.max(1) as u32,
            estimated_remaining: std::time::Duration::ZERO,
            fields_summary: FieldsSummary::new(),
        };
        let progress_json = serde_json::to_string(&progress).unwrap_or_default();
        self.progress_reporter.report(&progress_json);
        self.progress_reporter.on_complete();

        Ok(SimulationResult {
            success: true,
            final_step: stats.current_step,
            total_time: num_steps as f64 * dt,
        })
    }

    /// Get a reference to the underlying solver (if initialized)
    pub fn solver(&self) -> Option<&dyn Solver> {
        self.solver.as_deref()
    }

    /// Add a source to the simulation
    pub fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        self.sources.push(source);
        Ok(())
    }

    /// Add a sensor to the simulation
    pub fn add_sensor(&mut self, sensor: GridSensorSet) -> KwaversResult<()> {
        self.sensors.push(sensor);
        Ok(())
    }

    /// Enable a feature for this simulation
    pub fn enable_feature(&mut self, feature: SolverFeature) -> KwaversResult<()> {
        self.feature_manager.enable_feature(feature)?;
        Ok(())
    }

    /// Check if a feature is enabled
    pub fn is_feature_enabled(&self, feature: SolverFeature) -> bool {
        self.feature_manager.is_enabled(feature)
    }

    /// Get the feature manager
    pub fn feature_manager(&self) -> &FeatureManager {
        &self.feature_manager
    }

    /// Get simulation statistics
    pub fn statistics(&self) -> SimulationStatistics {
        SimulationStatistics {
            num_sources: self.sources.len(),
            num_sensors: self.sensors.len(),
            grid_size: self.grid.nx * self.grid.ny * self.grid.nz,
            enabled_features: self.feature_manager.feature_set(),
        }
    }
}

/// Simulation result
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub success: bool,
    pub final_step: usize,
    pub total_time: f64,
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct SimulationStatistics {
    pub num_sources: usize,
    pub num_sensors: usize,
    pub grid_size: usize,
    pub enabled_features: crate::solver::feature::SolverFeatureSet,
}

/// Simulation builder for fluent configuration
#[derive(Debug)]
pub struct SimulationBuilder<'a, M: Medium> {
    grid: Option<Grid>,
    medium: Option<&'a M>,
    sources: Vec<Arc<dyn Source>>,
    sensors: Vec<GridSensorSet>,
    feature_config: crate::solver::feature::FeatureBasedConfig,
    solver_config: SolverConfiguration,
}

impl<'a, M: Medium> SimulationBuilder<'a, M> {
    /// Create a new simulation builder
    pub fn new() -> Self {
        Self {
            grid: None,
            medium: None,
            sources: Vec::new(),
            sensors: Vec::new(),
            feature_config: crate::solver::feature::FeatureBasedConfig::default(),
            solver_config: SolverConfiguration::default(),
        }
    }

    /// Set the computational grid
    pub fn with_grid(mut self, grid: Grid) -> Self {
        self.grid = Some(grid);
        self
    }

    /// Set the medium
    pub fn with_medium(mut self, medium: &'a M) -> Self {
        self.medium = Some(medium);
        self
    }

    /// Add a source
    pub fn with_source(mut self, source: Arc<dyn Source>) -> Self {
        self.sources.push(source);
        self
    }

    /// Add a sensor
    pub fn with_sensor(mut self, sensor: GridSensorSet) -> Self {
        self.sensors.push(sensor);
        self
    }

    /// Enable a feature
    pub fn with_feature(mut self, feature: SolverFeature) -> Self {
        self.feature_config.enabled_features.enable(feature);
        self
    }

    /// Set fallback behavior
    pub fn with_fallback_behavior(
        mut self,
        behavior: crate::solver::feature::FallbackBehavior,
    ) -> Self {
        self.feature_config.fallback_behavior = behavior;
        self
    }

    /// Set solver configuration
    pub fn with_solver_config(mut self, config: SolverConfiguration) -> Self {
        self.solver_config = config;
        self
    }

    /// Set solver type
    pub fn with_solver_type(mut self, solver_type: SolverType) -> Self {
        self.solver_config.solver_type = solver_type;
        self
    }

    /// Build the simulation
    pub fn build(self) -> KwaversResult<CoreSimulation<'a, M>> {
        let grid = self
            .grid
            .clone()
            .ok_or_else(|| "Grid must be specified".to_string())?;
        let medium = self
            .medium
            .ok_or_else(|| "Medium must be specified".to_string())?;

        // Use console reporter by default
        let progress_reporter: Box<dyn ProgressReporter> =
            Box::new(crate::solver::progress::ConsoleProgressReporter::default());

        let mut simulation =
            CoreSimulation::new(grid, medium, self.sources, self.sensors, progress_reporter)?;

        simulation.set_solver_config(self.solver_config);

        for feature in self.feature_config.enabled_features.enabled_features() {
            let feature_enum = Self::parse_feature_name_static(feature)?;
            simulation.enable_feature(feature_enum)?;
        }

        Ok(simulation)
    }

    /// Helper to parse feature names back to enum (static version)
    fn parse_feature_name_static(name: &str) -> Result<SolverFeature, String> {
        match name {
            "Reconstruction" => Ok(SolverFeature::Reconstruction),
            "Time Reversal" => Ok(SolverFeature::TimeReversal),
            "Adaptive Mesh Refinement" => Ok(SolverFeature::AdaptiveMeshRefinement),
            "GPU Acceleration" => Ok(SolverFeature::GpuAcceleration),
            "Detailed Logging" => Ok(SolverFeature::DetailedLogging),
            "Validation Mode" => Ok(SolverFeature::ValidationMode),
            "High Precision" => Ok(SolverFeature::HighPrecision),
            "Multi-Threaded" => Ok(SolverFeature::MultiThreaded),
            "Memory Optimization" => Ok(SolverFeature::MemoryOptimization),
            "Experimental Features" => Ok(SolverFeature::ExperimentalFeatures),
            _ => Err(format!("Unknown feature: {}", name)),
        }
    }

    /// Helper to parse feature names back to enum (simplified)
    #[allow(dead_code)]
    fn parse_feature_name(&self, name: &str) -> Result<SolverFeature, String> {
        Self::parse_feature_name_static(name)
    }
}

impl<'a, M: Medium> Default for SimulationBuilder<'a, M> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use crate::domain::signal::SineWave;
    use crate::domain::source::PointSource;
    use std::sync::Arc;

    #[test]
    fn test_simulation_creation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let source = PointSource::new((0.032, 0.032, 0.032), signal);
        let sources: Vec<Arc<dyn Source>> = vec![Arc::new(source)];

        let simulation = CoreSimulation::new(
            grid,
            &medium,
            sources,
            vec![],
            Box::new(crate::solver::progress::ConsoleProgressReporter::default()),
        )
        .unwrap();

        assert_eq!(simulation.statistics().num_sources, 1);
        assert_eq!(simulation.statistics().num_sensors, 0);
    }

    #[test]
    fn test_feature_management() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let mut simulation = CoreSimulation::new(
            grid,
            &medium,
            vec![],
            vec![],
            Box::new(crate::solver::progress::ConsoleProgressReporter::default()),
        )
        .unwrap();

        // Test enabling features
        assert!(simulation
            .enable_feature(SolverFeature::Reconstruction)
            .is_ok());
        assert!(simulation.is_feature_enabled(SolverFeature::Reconstruction));

        assert!(simulation
            .enable_feature(SolverFeature::GpuAcceleration)
            .is_ok());
        assert!(simulation.is_feature_enabled(SolverFeature::GpuAcceleration));
    }

    #[test]
    fn test_simulation_builder() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let source = PointSource::new((0.016, 0.016, 0.016), signal);
        let source: Arc<dyn Source> = Arc::new(source);

        let simulation = SimulationBuilder::new()
            .with_grid(grid)
            .with_medium(&medium)
            .with_source(source)
            .with_feature(SolverFeature::Reconstruction)
            .build();

        assert!(simulation.is_ok());
        let simulation = simulation.unwrap();
        assert!(simulation.is_feature_enabled(SolverFeature::Reconstruction));
    }
}
