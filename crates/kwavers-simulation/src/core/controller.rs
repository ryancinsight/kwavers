//! `CoreSimulation` controller — orchestrates solver, sources, sensors, and medium.

use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_domain::sensor::GridSensorSet;
use kwavers_domain::source::Source;
use crate::solver_factory::SimulationSolverFactory;
use kwavers_solver::config::SolverConfiguration;
use kwavers_solver::feature::{FeatureManager, SolverFeature};
use kwavers_solver::interface::{FieldsSummary, ProgressReporter, ProgressUpdate, Solver};
use std::sync::Arc;

use super::types::{CoreSimulationStatistics, SimulationResult};

/// Core simulation controller
pub struct CoreSimulation<'a, M: Medium> {
    pub(super) grid: Grid,
    pub(super) medium: &'a M,
    pub(super) sources: Vec<Arc<dyn Source>>,
    /// High-level sensor probe sets (grid-indexed), supporting multi-physics (acoustics + optics).
    pub(super) sensors: Vec<GridSensorSet>,
    pub(super) feature_manager: FeatureManager,
    pub(super) progress_reporter: Box<dyn ProgressReporter>,
    /// Solver configuration (determines which solver to use and how)
    pub(super) solver_config: SolverConfiguration,
    /// The instantiated solver (created on first run or explicit initialize)
    pub(super) solver: Option<Box<dyn Solver>>,
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn initialize(&mut self) -> KwaversResult<()> {
        self.progress_reporter.on_start(0, 0.0);

        // Create the solver via factory
        let mut solver = SimulationSolverFactory::create_solver(
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
        if self
            .feature_manager
            .is_enabled(SolverFeature::Reconstruction)
        {
            let _ = solver.enable_feature(SolverFeature::Reconstruction, true);
        }
        if self
            .feature_manager
            .is_enabled(SolverFeature::GpuAcceleration)
        {
            let _ = solver.enable_feature(SolverFeature::GpuAcceleration, true);
        }

        self.solver = Some(solver);
        Ok(())
    }

    /// Run the simulation using the configured solver
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn run(&mut self, num_steps: usize, dt: f64) -> KwaversResult<SimulationResult> {
        // Auto-initialize if not already done
        if self.solver.is_none() {
            // Update dt in config before initialization
            self.solver_config.dt = dt;
            self.solver_config.max_steps = num_steps;
            self.initialize()?;
        }

        let solver = self.solver.as_mut().ok_or_else(|| {
            kwavers_core::error::KwaversError::InternalError("Solver not initialized".to_owned())
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        self.sources.push(source);
        Ok(())
    }

    /// Add a sensor to the simulation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_sensor(&mut self, sensor: GridSensorSet) -> KwaversResult<()> {
        self.sensors.push(sensor);
        Ok(())
    }

    /// Enable a feature for this simulation
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    pub fn statistics(&self) -> CoreSimulationStatistics {
        CoreSimulationStatistics {
            num_sources: self.sources.len(),
            num_sensors: self.sensors.len(),
            grid_size: self.grid.nx * self.grid.ny * self.grid.nz,
            enabled_features: self.feature_manager.feature_set(),
        }
    }
}
