//! GPU-Accelerated Time Integration
//!
//! Provides GPU-specific time integration strategies that leverage real-time
//! orchestration, performance monitoring, and budget-aware timestep adjustment.
//!
//! ## References
//!
//! - k-Wave: GPU time stepping with adaptive CFL
//! - j-Wave: JAX-based adaptive time integration
//! - SimSonic: Real-time HIFU simulation

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::plugin::Plugin;
use crate::solver::backend::gpu::realtime_loop::{RealtimeConfig, RealtimeSimulationOrchestrator};
use ndarray::Array3;
use std::collections::HashMap;

/// GPU-accelerated time integration strategy
///
/// Manages GPU-based multiphysics timesteps with real-time budget enforcement,
/// performance monitoring, and adaptive timestepping.
#[derive(Debug)]
pub struct GPUTimeIntegrator {
    /// Real-time orchestrator for GPU execution
    orchestrator: RealtimeSimulationOrchestrator,

    /// Configuration for real-time constraints
    config: RealtimeConfig,

    /// Total wall time accumulated (seconds)
    total_wall_time: f64,

    /// Total simulation time (seconds)
    total_sim_time: f64,

    /// Number of steps executed
    step_count: u64,
}

impl GPUTimeIntegrator {
    /// Create a new GPU time integrator with real-time constraints
    ///
    /// # Arguments
    ///
    /// * `config` - Real-time configuration with budget_ms and CFL safety factor
    /// * `grid` - Computational grid (used for GPU kernel setup)
    ///
    /// # Returns
    ///
    /// A new GPU time integrator ready for multiphysics simulation
    pub fn new(config: RealtimeConfig, _grid: &Grid) -> KwaversResult<Self> {
        use crate::solver::backend::gpu::physics_kernels::PhysicsKernelRegistry;

        let kernel_registry = PhysicsKernelRegistry::new();
        let orchestrator = RealtimeSimulationOrchestrator::new(config.clone(), kernel_registry)?;

        Ok(Self {
            orchestrator,
            config,
            total_wall_time: 0.0,
            total_sim_time: 0.0,
            step_count: 0,
        })
    }

    /// Advance solution using GPU time integration
    ///
    /// Executes multiphysics timesteps on GPU with real-time budget enforcement
    /// and adaptive timestepping based on CFL conditions.
    ///
    /// # Arguments
    ///
    /// * `fields` - Mutable map of field arrays (acoustic, optical, thermal)
    /// * `_physics_components` - Physics plugins (for future coupling)
    /// * `global_time` - Current simulation time
    /// * `target_time` - Target end time for integration
    /// * `dt` - Initial timestep size
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Final time reached after integration
    pub fn advance(
        &mut self,
        fields: &mut HashMap<String, Array3<f64>>,
        _physics_components: &HashMap<String, Box<dyn Plugin>>,
        global_time: f64,
        target_time: f64,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let mut current_time = global_time;
        let mut current_dt = dt;

        while current_time < target_time {
            // Adaptive timestepping: ensure we don't overshoot target
            let step_dt = current_dt.min(target_time - current_time);

            // Execute GPU multiphysics step with performance tracking
            let step_result = self
                .orchestrator
                .step(fields, step_dt, current_time, grid)?;

            // Update time tracking
            current_time += step_result.dt;
            self.total_sim_time += step_result.dt;
            self.total_wall_time += step_result.wall_time_ms / 1000.0;
            self.step_count += 1;

            // Adjust timestep based on budget status
            if !step_result.within_budget && self.config.adaptive_timestepping {
                // If we exceeded budget, reduce timestep for next iteration
                current_dt = current_dt * 0.95;
                if self.config.verbose {
                    eprintln!(
                        "Budget exceeded ({}ms), reducing dt to {:.3e}",
                        step_result.wall_time_ms, current_dt
                    );
                }
            } else if step_result.within_budget && self.config.adaptive_timestepping {
                // If we have headroom, slightly increase timestep
                current_dt = (current_dt * 1.02).min(dt * 1.5);
            }

            // Apply CFL safety factor
            current_dt = current_dt * self.config.cfl_safety_factor;
        }

        Ok(current_time)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> crate::solver::backend::gpu::PerformanceMetrics {
        self.orchestrator.get_metrics()
    }

    /// Get budget analysis
    pub fn analyze_budget(&self) -> crate::solver::backend::gpu::BudgetAnalysis {
        self.orchestrator.analyze_budget()
    }

    /// Get integration statistics
    pub fn get_statistics(&self) -> GPUIntegrationStatistics {
        let metrics = self.orchestrator.get_metrics();

        GPUIntegrationStatistics {
            steps: self.step_count,
            total_wall_time_seconds: self.total_wall_time,
            total_sim_time_seconds: self.total_sim_time,
            average_step_time_ms: if self.step_count > 0 {
                (self.total_wall_time / self.step_count as f64) * 1000.0
            } else {
                0.0
            },
            speedup_estimate: if self.total_sim_time > 0.0 && self.total_wall_time > 0.0 {
                self.total_sim_time / self.total_wall_time
            } else {
                0.0
            },
            budget_satisfaction: metrics.budget_satisfaction,
            p95_step_time_ms: metrics.p95_step_time_ms,
            p99_step_time_ms: metrics.p99_step_time_ms,
        }
    }

    /// Check if currently within real-time budget
    pub fn is_within_budget(&self) -> bool {
        self.orchestrator.is_within_budget()
    }

    /// Enable/disable verbose output
    pub fn set_verbose(&mut self, verbose: bool) {
        self.config.verbose = verbose;
    }
}

/// Statistics for GPU time integration
#[derive(Debug, Clone)]
pub struct GPUIntegrationStatistics {
    /// Total number of timesteps executed
    pub steps: u64,

    /// Total wall time in seconds
    pub total_wall_time_seconds: f64,

    /// Total simulation time in seconds
    pub total_sim_time_seconds: f64,

    /// Average step execution time in milliseconds
    pub average_step_time_ms: f64,

    /// Speedup relative to real-time (sim_time / wall_time)
    pub speedup_estimate: f64,

    /// Satisfaction ratio (fraction of steps within budget)
    pub budget_satisfaction: f64,

    /// 95th percentile step time in milliseconds
    pub p95_step_time_ms: f64,

    /// 99th percentile step time in milliseconds
    pub p99_step_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_integrator_creation() -> KwaversResult<()> {
        let config = RealtimeConfig::default();
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1)?;

        let integrator = GPUTimeIntegrator::new(config, &grid)?;
        assert_eq!(integrator.step_count, 0);
        assert_eq!(integrator.total_wall_time, 0.0);

        Ok(())
    }

    #[test]
    fn test_statistics_initialization() -> KwaversResult<()> {
        let config = RealtimeConfig::default();
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1)?;
        let integrator = GPUTimeIntegrator::new(config, &grid)?;

        let stats = integrator.get_statistics();
        assert_eq!(stats.steps, 0);
        assert_eq!(stats.speedup_estimate, 0.0);

        Ok(())
    }

    #[test]
    fn test_budget_checking() -> KwaversResult<()> {
        let config = RealtimeConfig {
            budget_ms: 5.0,
            ..Default::default()
        };
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1)?;
        let integrator = GPUTimeIntegrator::new(config, &grid)?;

        // Initially should be within budget (no steps executed)
        assert!(integrator.is_within_budget());

        Ok(())
    }

    #[test]
    fn test_verbose_mode() -> KwaversResult<()> {
        let config = RealtimeConfig {
            verbose: false,
            ..Default::default()
        };
        let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1)?;
        let mut integrator = GPUTimeIntegrator::new(config, &grid)?;

        integrator.set_verbose(true);
        assert!(integrator.config.verbose);

        Ok(())
    }
}
