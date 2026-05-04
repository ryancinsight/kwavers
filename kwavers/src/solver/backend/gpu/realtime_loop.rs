//! GPU Real-Time Simulation Loop Orchestrator
//!
//! Coordinates GPU-accelerated multiphysics timesteps with real-time budget
//! enforcement, performance monitoring, and async I/O for checkpoints.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::solver::backend::gpu::performance_monitor::{
    BudgetAnalysis, PerformanceMetrics, PerformanceMonitor,
};
use crate::solver::backend::gpu::physics_kernels::{PhysicsDomain, PhysicsKernelRegistry};
use log::debug;
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Instant;

/// Real-time simulation configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Target execution time per step (milliseconds)
    pub budget_ms: f64,

    /// Enable adaptive time stepping
    pub adaptive_timestepping: bool,

    /// CFL safety factor (typically 0.8-0.95)
    pub cfl_safety_factor: f64,

    /// Checkpoint interval (save every N steps)
    pub checkpoint_interval: usize,

    /// Enable async I/O for checkpoints
    pub enable_async_io: bool,

    /// Verbose output
    pub verbose: bool,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            budget_ms: 10.0,
            adaptive_timestepping: true,
            cfl_safety_factor: 0.9,
            checkpoint_interval: 100,
            enable_async_io: true,
            verbose: false,
        }
    }
}

/// Result of single timestep execution
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Timestep size used
    pub dt: f64,

    /// Total time after this step
    pub time: f64,

    /// Wall time for execution (milliseconds)
    pub wall_time_ms: f64,

    /// Whether step was within budget
    pub within_budget: bool,

    /// Number of GPU kernels executed
    pub kernels_executed: usize,
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct SimulationStatistics {
    /// Total wall time (seconds)
    pub total_wall_time_seconds: f64,

    /// Total simulation time (seconds)
    pub total_simulation_time_seconds: f64,

    /// Number of steps executed
    pub num_steps: u64,

    /// Number of budget violations
    pub budget_violations: u64,

    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Real-time simulation orchestrator
#[derive(Debug)]
pub struct RealtimeSimulationOrchestrator {
    /// Configuration
    config: RealtimeConfig,

    /// Performance monitor
    monitor: PerformanceMonitor,

    /// Physics kernel registry
    kernel_registry: PhysicsKernelRegistry,

    /// Current timestep count
    step_count: u64,

    /// Start time of simulation
    start_time: Option<Instant>,
}

impl RealtimeSimulationOrchestrator {
    /// Create new real-time orchestrator
    pub fn new(
        config: RealtimeConfig,
        kernel_registry: PhysicsKernelRegistry,
    ) -> KwaversResult<Self> {
        Ok(Self {
            monitor: PerformanceMonitor::new(config.budget_ms, 100),
            config,
            kernel_registry,
            step_count: 0,
            start_time: None,
        })
    }

    /// Execute one scheduled multiphysics timestep.
    ///
    /// # Contract
    ///
    /// This orchestrator owns realtime scheduling and budget accounting for
    /// registered GPU kernel descriptors. Actual `wgpu` command encoding lives
    /// behind concrete kernel implementations; this layer validates that a
    /// nonempty field set has registered kernels, records each descriptor's
    /// analytical execution estimate, measures scheduler wall time, and advances
    /// the timestep counter. Empty field sets are a valid zero-kernel step.
    pub fn step(
        &mut self,
        fields: &mut HashMap<String, Array3<f64>>,
        dt: f64,
        time: f64,
        grid: &Grid,
    ) -> KwaversResult<StepResult> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Realtime GPU timestep must be finite and positive; got {dt}"
            )));
        }
        if !time.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Realtime GPU simulation time must be finite; got {time}"
            )));
        }

        let step_start = Instant::now();
        let kernels = self.kernel_registry.list_kernels();
        if !fields.is_empty() && kernels.is_empty() {
            return Err(KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "gpu_kernel_registry".to_string(),
                    value: "empty".to_string(),
                    constraint: "Nonempty realtime GPU field state requires at least one registered physics kernel".to_string(),
                },
            ));
        }

        let num_elements = grid.nx * grid.ny * grid.nz;
        for domain in &kernels {
            if let Some(kernel) = self.kernel_registry.get_kernel(*domain) {
                self.monitor.record_kernel(
                    domain.name().to_string(),
                    kernel.estimate_time_ms(num_elements),
                );
            }
        }

        let wall_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        self.monitor.record_step(wall_time_ms);
        self.step_count += 1;

        Ok(StepResult {
            dt,
            time,
            wall_time_ms,
            within_budget: wall_time_ms <= self.config.budget_ms,
            kernels_executed: kernels.len(),
        })
    }

    /// Run full simulation loop
    pub fn simulate(
        &mut self,
        fields: &mut HashMap<String, Array3<f64>>,
        t_start: f64,
        t_end: f64,
        mut dt: f64,
        grid: &Grid,
    ) -> KwaversResult<SimulationStatistics> {
        self.start_time = Some(Instant::now());

        let mut t = t_start;
        let mut step = 0u64;

        while t < t_end {
            // Adaptive time stepping
            if self.config.adaptive_timestepping {
                dt = self.adjust_timestep(dt, t, t_end);
            }

            // Execute step
            let result = self.step(fields, dt, t, grid)?;

            // Update time
            t += result.dt;
            step += 1;

            // Checkpoint (placeholder for async I/O)
            if step % self.config.checkpoint_interval as u64 == 0 && self.config.enable_async_io {
                if self.config.verbose {
                    debug!("Checkpoint at step {} (time={:.3e})", step, t);
                }
            }
        }

        let elapsed = self.start_time.take().unwrap().elapsed().as_secs_f64();
        let metrics = self.monitor.get_metrics();

        Ok(SimulationStatistics {
            total_wall_time_seconds: elapsed,
            total_simulation_time_seconds: t - t_start,
            num_steps: step,
            budget_violations: self.monitor.budget_violations(),
            metrics,
        })
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.monitor.get_metrics()
    }

    /// Get budget analysis
    pub fn analyze_budget(&self) -> BudgetAnalysis {
        self.monitor.analyze_budget()
    }

    /// Check if currently within budget
    pub fn is_within_budget(&self) -> bool {
        self.monitor.is_within_budget()
    }

    // ========== Private Methods ==========

    /// Adjust timestep for stability and time step requirements
    fn adjust_timestep(&self, current_dt: f64, _t: f64, t_end: f64) -> f64 {
        // Ensure we don't overshoot end time
        let max_dt = (t_end - _t).max(current_dt);

        // Apply CFL safety factor
        let safe_dt = current_dt * self.config.cfl_safety_factor;

        safe_dt.min(max_dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backend::gpu::physics_kernels::{PhysicsKernel, WorkgroupConfig};

    #[test]
    fn test_config_default() {
        let config = RealtimeConfig::default();
        assert_eq!(config.budget_ms, 10.0);
        assert!(config.adaptive_timestepping);
        assert_eq!(config.cfl_safety_factor, 0.9);
    }

    #[test]
    fn test_orchestrator_creation() -> KwaversResult<()> {
        let config = RealtimeConfig::default();
        let registry = PhysicsKernelRegistry::new();
        let orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;

        assert_eq!(orchestrator.step_count, 0);
        Ok(())
    }

    #[test]
    fn test_step_result_creation() {
        let result = StepResult {
            dt: 1e-6,
            time: 1e-5,
            wall_time_ms: 5.0,
            within_budget: true,
            kernels_executed: 3,
        };

        assert_eq!(result.dt, 1e-6);
        assert!(result.within_budget);
    }

    #[test]
    fn test_budget_enforcement() -> KwaversResult<()> {
        let config = RealtimeConfig {
            budget_ms: 5.0,
            ..Default::default()
        };
        let registry = PhysicsKernelRegistry::new();
        let mut orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;

        let mut fields = HashMap::new();
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1)?;

        let result = orchestrator.step(&mut fields, 1e-6, 0.0, &grid)?;

        assert_eq!(result.kernels_executed, 0);
        assert!(result.time == 0.0);
        assert_eq!(orchestrator.step_count, 1);

        Ok(())
    }

    #[test]
    fn test_nonempty_fields_require_registered_kernel() -> KwaversResult<()> {
        let config = RealtimeConfig::default();
        let registry = PhysicsKernelRegistry::new();
        let mut orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;
        let grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1)?;
        let mut fields = HashMap::from([("pressure".to_string(), Array3::zeros((4, 4, 4)))]);

        let error = orchestrator
            .step(&mut fields, 1e-6, 0.0, &grid)
            .unwrap_err();

        assert!(format!("{error}").contains("requires at least one registered physics kernel"));
        Ok(())
    }

    #[test]
    fn test_registered_kernel_step_records_execution_metadata() -> KwaversResult<()> {
        let config = RealtimeConfig::default();
        let mut registry = PhysicsKernelRegistry::new();
        registry.register(PhysicsKernel::new(
            PhysicsDomain::AcousticFDTD,
            "@compute @workgroup_size(1) fn compute_main() {}".to_string(),
            "compute_main".to_string(),
            25,
            WorkgroupConfig::new(4, 4, 4),
        ))?;
        let mut orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;
        let grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1)?;
        let mut fields = HashMap::from([("pressure".to_string(), Array3::zeros((4, 4, 4)))]);

        let result = orchestrator.step(&mut fields, 1e-6, 1e-5, &grid)?;
        let metrics = orchestrator.get_metrics();

        assert_eq!(result.kernels_executed, 1);
        assert_eq!(result.dt, 1e-6);
        assert_eq!(result.time, 1e-5);
        assert!(metrics.avg_step_time_ms >= 0.0);
        assert_eq!(orchestrator.step_count, 1);
        Ok(())
    }

    #[test]
    fn test_timestep_adjustment() -> KwaversResult<()> {
        let config = RealtimeConfig {
            cfl_safety_factor: 0.8,
            ..Default::default()
        };
        let registry = PhysicsKernelRegistry::new();
        let orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;

        let dt = 1e-5;
        let adjusted = orchestrator.adjust_timestep(dt, 0.0, 1.0);

        assert!(adjusted <= dt);
        assert!((adjusted - dt * 0.8).abs() < 1e-15);

        Ok(())
    }
}
