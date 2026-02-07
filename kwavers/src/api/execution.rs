//! API Execution Layer
//!
//! Connects API configurations to actual solver execution.
//! This module bridges the gap between user-friendly APIs and low-level solvers.

use super::{SimulationOutput, SimulationStatistics};
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::PMLConfig;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::domain::medium::Medium;
use crate::domain::source::grid_source::GridSource;
use crate::simulation::configuration::Configuration;
use crate::solver::backend::BackendContext;
use crate::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use crate::solver::forward::hybrid::{HybridConfig, HybridSolver};
use crate::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use log::info;
use ndarray::Array3;
use std::time::Instant;

/// Solver execution engine
pub struct ExecutionEngine {
    /// Configuration
    config: Configuration,
    /// Backend (CPU/GPU)
    backend: Option<BackendContext>,
}

impl ExecutionEngine {
    /// Create new execution engine
    pub fn new(config: Configuration) -> Self {
        Self {
            config,
            backend: None,
        }
    }

    /// Set backend
    pub fn with_backend(mut self, backend: BackendContext) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Execute simulation
    pub fn execute(&self) -> KwaversResult<SimulationOutput> {
        // Determine which solver to use based on configuration
        let solver_type = &self.config.solver.solver_type;

        match solver_type.as_str() {
            "fdtd" => self.execute_fdtd(),
            "pstd" => self.execute_pstd(),
            "hybrid" => self.execute_hybrid(),
            _ => self.execute_fdtd(), // Default fallback
        }
    }

    /// Execute FDTD solver
    fn execute_fdtd(&self) -> KwaversResult<SimulationOutput> {
        let start_time = Instant::now();

        // Create grid
        let grid = Grid::new(
            self.config.grid.nx,
            self.config.grid.ny,
            self.config.grid.nz,
            self.config.grid.dx,
            self.config.grid.dy,
            self.config.grid.dz,
        );

        // Create medium (use water as default)
        let medium = HomogeneousMedium::water();

        // Create source (default: point source at center)
        let source = self.create_default_source(&grid)?;

        // Create FDTD configuration
        let fdtd_config = FdtdConfig {
            dt: self.config.simulation.dt,
            nt: self.config.simulation.num_time_steps,
            spatial_order: 2, // Default 2nd order
            pml: Some(PMLConfig::default()),
            sensor_mask: None,
            source_freq: 1e6, // Default 1 MHz
            source_mask: None,
        };

        // Create solver
        let mut solver = FdtdSolver::new(fdtd_config, &grid, &medium, source)?;

        // Execute time stepping
        for step in 0..self.config.simulation.num_time_steps {
            solver.step()?;

            // Progress reporting (every 10%)
            if step % (self.config.simulation.num_time_steps / 10).max(1) == 0 {
                let progress = (step as f64 / self.config.simulation.num_time_steps as f64) * 100.0;
                info!("Progress: {:.1}%", progress);
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64();

        // Extract results
        let pressure = solver.fields.pressure.clone();

        // Create statistics
        let stats = SimulationStatistics {
            time_steps: self.config.simulation.num_time_steps,
            final_time: self.config.simulation.dt * self.config.simulation.num_time_steps as f64,
            avg_time_per_step: execution_time / self.config.simulation.num_time_steps as f64,
            backend: self
                .backend
                .as_ref()
                .map(|b| format!("{:?}", b.backend_type()))
                .unwrap_or_else(|| "CPU".to_string()),
            memory_usage: self.estimate_memory_usage(),
            flops: self.estimate_flops(execution_time),
        };

        Ok(SimulationOutput {
            pressure,
            sensor_data: None, // TODO: Extract from solver.sensor_recorder
            statistics: stats,
            execution_time,
        })
    }

    /// Execute PSTD solver
    fn execute_pstd(&self) -> KwaversResult<SimulationOutput> {
        let start_time = Instant::now();

        // Create grid
        let grid = Grid::new(
            self.config.grid.nx,
            self.config.grid.ny,
            self.config.grid.nz,
            self.config.grid.dx,
            self.config.grid.dy,
            self.config.grid.dz,
        );

        // Create medium (use water as default)
        let medium = HomogeneousMedium::water();

        // Create source (default: point source at center)
        let source = self.create_default_source(&grid)?;

        // Create PSTD configuration
        let pstd_config = PSTDConfig {
            nt: self.config.simulation.num_time_steps,
            dt: self.config.simulation.dt,
            compatibility_mode: crate::solver::forward::pstd::config::CompatibilityMode::Optimal,
            spectral_correction: Default::default(),
            absorption_mode:
                crate::physics::acoustics::mechanics::absorption::AbsorptionMode::Lossless,
            nonlinearity: false,
            boundary: crate::solver::forward::pstd::config::BoundaryConfig::PML(
                crate::domain::boundary::PMLConfig::default(),
            ),
            sensor_mask: None,
            pml_inside: true,
            smooth_sources: true,
            anti_aliasing: Default::default(),
            kspace_method: crate::solver::forward::pstd::config::KSpaceMethod::StandardPSTD,
        };

        // Create PSTD solver
        let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, source)?;

        // Execute time stepping
        for step in 0..self.config.simulation.num_time_steps {
            solver.step_forward()?;

            // Progress reporting (every 10%)
            if step % (self.config.simulation.num_time_steps / 10).max(1) == 0 {
                let progress = (step as f64 / self.config.simulation.num_time_steps as f64) * 100.0;
                info!("PSTD Progress: {:.1}%", progress);
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64();

        // Extract results
        let pressure = solver.fields.p.clone();

        // Create statistics
        let stats = SimulationStatistics {
            time_steps: self.config.simulation.num_time_steps,
            final_time: self.config.simulation.dt * self.config.simulation.num_time_steps as f64,
            avg_time_per_step: execution_time / self.config.simulation.num_time_steps as f64,
            backend: self
                .backend
                .as_ref()
                .map(|b| format!("{:?}", b.backend_type()))
                .unwrap_or_else(|| "PSTD-CPU".to_string()),
            memory_usage: self.estimate_memory_usage(),
            flops: self.estimate_flops(execution_time),
        };

        Ok(SimulationOutput {
            pressure,
            sensor_data: solver.extract_pressure_data(),
            statistics: stats,
            execution_time,
        })
    }

    /// Execute hybrid solver
    fn execute_hybrid(&self) -> KwaversResult<SimulationOutput> {
        let start_time = Instant::now();

        // Create grid
        let grid = Grid::new(
            self.config.grid.nx,
            self.config.grid.ny,
            self.config.grid.nz,
            self.config.grid.dx,
            self.config.grid.dy,
            self.config.grid.dz,
        );

        // Create medium (use water as default)
        let medium = HomogeneousMedium::water();

        // Create Hybrid configuration (combines PSTD and FDTD)
        let hybrid_config = HybridConfig {
            pstd_config: PSTDConfig {
                nt: self.config.simulation.num_time_steps,
                dt: self.config.simulation.dt,
                ..Default::default()
            },
            fdtd_config: FdtdConfig {
                dt: self.config.simulation.dt,
                nt: self.config.simulation.num_time_steps,
                spatial_order: 2,
                pml: Some(PMLConfig::default()),
                sensor_mask: None,
                source_freq: 1e6,
                source_mask: None,
            },
            decomposition_strategy:
                crate::solver::forward::hybrid::config::DecompositionStrategy::Dynamic,
            selection_criteria: Default::default(),
            coupling_interface: Default::default(),
            optimization: Default::default(),
            validation: Default::default(),
        };

        // Create Hybrid solver
        let mut solver = HybridSolver::new(hybrid_config, &grid, &medium)?;

        // Execute time stepping
        for step in 0..self.config.simulation.num_time_steps {
            solver.step_forward()?;

            // Progress reporting (every 10%)
            if step % (self.config.simulation.num_time_steps / 10).max(1) == 0 {
                let progress = (step as f64 / self.config.simulation.num_time_steps as f64) * 100.0;
                info!("Hybrid Progress: {:.1}%", progress);
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64();

        // Extract results from hybrid solver
        // Access fields directly (they're public in the struct)
        let pressure = solver.fields.p.clone();

        // Create statistics
        let stats = SimulationStatistics {
            time_steps: self.config.simulation.num_time_steps,
            final_time: self.config.simulation.dt * self.config.simulation.num_time_steps as f64,
            avg_time_per_step: execution_time / self.config.simulation.num_time_steps as f64,
            backend: self
                .backend
                .as_ref()
                .map(|b| format!("{:?}", b.backend_type()))
                .unwrap_or_else(|| "Hybrid-CPU".to_string()),
            memory_usage: self.estimate_memory_usage() * 1.5, // Hybrid uses more memory
            flops: self.estimate_flops(execution_time),
        };

        Ok(SimulationOutput {
            pressure,
            sensor_data: None, // Hybrid solver sensor data extraction TBD
            statistics: stats,
            execution_time,
        })
    }

    /// Create default source at grid center
    fn create_default_source(&self, grid: &Grid) -> KwaversResult<GridSource> {
        let source_x = grid.nx / 2;
        let source_y = grid.ny / 2;
        let source_z = grid.nz / 2;

        let mut source_mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        source_mask[[source_x, source_y, source_z]] = 1.0;

        // Create simple sinusoidal source
        let frequency = 1e6; // 1 MHz default
        let dt = self.config.simulation.dt;
        let nt = self.config.simulation.num_time_steps;

        let mut source_signal = vec![0.0; nt];
        for (i, val) in source_signal.iter_mut().enumerate() {
            let t = i as f64 * dt;
            *val = (2.0 * std::f64::consts::PI * frequency * t).sin();
        }

        Ok(GridSource::new(source_mask, source_signal)?)
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> f64 {
        let nx = self.config.grid.nx;
        let ny = self.config.grid.ny;
        let nz = self.config.grid.nz;

        // 4 fields (p, vx, vy, vz) × 8 bytes per f64
        let total_bytes = nx * ny * nz * 4 * 8;
        total_bytes as f64 / 1e9 // Convert to GB
    }

    /// Estimate FLOPS
    fn estimate_flops(&self, execution_time: f64) -> f64 {
        let nx = self.config.grid.nx;
        let ny = self.config.grid.ny;
        let nz = self.config.grid.nz;
        let nt = self.config.simulation.num_time_steps;

        // Rough estimate: ~30 FLOPS per grid point per time step
        let total_ops = (nx * ny * nz * nt * 30) as f64;
        total_ops / execution_time // FLOPS
    }
}

/// Execute configuration directly
pub fn execute_simulation(config: &Configuration) -> KwaversResult<SimulationOutput> {
    let engine = ExecutionEngine::new(config.clone());
    engine.execute()
}

/// Execute with backend
pub fn execute_with_backend(
    config: &Configuration,
    backend: BackendContext,
) -> KwaversResult<SimulationOutput> {
    let engine = ExecutionEngine::new(config.clone()).with_backend(backend);
    engine.execute()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::factory::SimulationFactory;

    #[test]
    fn test_execution_engine_creation() {
        let config = Configuration::default();
        let engine = ExecutionEngine::new(config);

        assert!(engine.backend.is_none());
    }

    #[test]
    fn test_execute_simulation() {
        // Create minimal configuration
        let config = SimulationFactory::new()
            .frequency(1e6)
            .domain_size(0.05, 0.05, 0.03)
            .auto_configure()
            .build()
            .unwrap();

        let result = execute_simulation(&config);
        assert!(result.is_ok());

        if let Ok(output) = result {
            assert!(output.execution_time > 0.0);
            assert_eq!(
                output.pressure.dim(),
                (config.grid.nx, config.grid.ny, config.grid.nz)
            );
        }
    }

    #[test]
    fn test_memory_estimation() {
        let config = SimulationFactory::new()
            .frequency(1e6)
            .domain_size(0.1, 0.1, 0.05)
            .auto_configure()
            .build()
            .unwrap();

        let engine = ExecutionEngine::new(config);
        let memory = engine.estimate_memory_usage();

        assert!(memory > 0.0);
        assert!(memory < 100.0); // Should be reasonable for this grid size
    }
}
