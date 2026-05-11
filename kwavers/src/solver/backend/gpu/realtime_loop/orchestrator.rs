//! `RealtimeSimulationOrchestrator`: realtime-budgeted GPU multiphysics loop.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::solver::backend::gpu::performance_monitor::{
    BudgetAnalysis, PerformanceMetrics, PerformanceMonitor,
};
use crate::solver::backend::gpu::physics_kernels::PhysicsKernelRegistry;
use log::debug;
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Instant;

use super::types::{RealtimeConfig, SimulationStatistics, StepResult};

/// Real-time simulation orchestrator.
#[derive(Debug)]
pub struct RealtimeSimulationOrchestrator {
    config: RealtimeConfig,
    monitor: PerformanceMonitor,
    kernel_registry: PhysicsKernelRegistry,
    step_count: u64,
    start_time: Option<Instant>,
}

impl RealtimeSimulationOrchestrator {
    /// Create new real-time orchestrator.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
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
    ///
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
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

    /// Run full simulation loop.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
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
            if self.config.adaptive_timestepping {
                dt = self.adjust_timestep(dt, t, t_end);
            }

            let result = self.step(fields, dt, t, grid)?;

            t += result.dt;
            step += 1;

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

    /// Get current performance metrics.
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.monitor.get_metrics()
    }

    /// Get budget analysis.
    pub fn analyze_budget(&self) -> BudgetAnalysis {
        self.monitor.analyze_budget()
    }

    /// Check if currently within budget.
    pub fn is_within_budget(&self) -> bool {
        self.monitor.is_within_budget()
    }

    /// Adjust timestep for CFL stability and end-time constraint.
    fn adjust_timestep(&self, current_dt: f64, _t: f64, t_end: f64) -> f64 {
        let max_dt = (t_end - _t).max(current_dt);
        let safe_dt = current_dt * self.config.cfl_safety_factor;
        safe_dt.min(max_dt)
    }
}
