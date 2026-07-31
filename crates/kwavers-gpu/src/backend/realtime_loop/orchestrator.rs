//! `RealtimeSimulationOrchestrator`: realtime-budgeted GPU multiphysics loop.

use crate::backend::performance_monitor::{BudgetAnalysis, GpuPerformanceMonitor, GpuStepMetrics};
use crate::backend::physics_kernels::PhysicsKernelRegistry;
use aequitas::systems::si::quantities::Time;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3 as LetoArray3;
use log::debug;
use std::collections::HashMap;
use std::time::Instant;

use super::types::{GpuRealtimeSimulationStatistics, RealtimeConfig, StepResult};

/// Real-time simulation orchestrator.
#[derive(Debug)]
pub struct RealtimeSimulationOrchestrator {
    config: RealtimeConfig,
    monitor: GpuPerformanceMonitor,
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
            monitor: GpuPerformanceMonitor::new(config.budget, 100),
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
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    pub fn step(
        &mut self,
        fields: &mut HashMap<String, LetoArray3<f64>>,
        dt: Time<f64>,
        time: Time<f64>,
        grid: &Grid,
    ) -> KwaversResult<StepResult> {
        if !dt.into_base().is_finite() || dt.into_base() <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Realtime GPU timestep must be finite and positive; got {} s",
                dt.into_base()
            )));
        }
        if !time.into_base().is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Realtime GPU simulation time must be finite; got {} s",
                time.into_base()
            )));
        }

        let step_start = Instant::now();
        let kernels = self.kernel_registry.list_kernels();
        if !fields.is_empty() && kernels.is_empty() {
            return Err(KwaversError::Config(
                kwavers_core::error::ConfigError::InvalidValue {
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
                    kernel.estimate_time(num_elements),
                );
            }
        }

        let wall_time = Time::from_base(step_start.elapsed().as_secs_f64());
        self.monitor.record_step(wall_time);
        self.step_count += 1;

        Ok(StepResult {
            dt,
            time,
            wall_time,
            within_budget: wall_time.into_base() <= self.config.budget.into_base(),
            kernels_executed: kernels.len(),
        })
    }

    /// Number of time steps advanced so far.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Run full simulation loop.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    pub fn simulate(
        &mut self,
        fields: &mut HashMap<String, LetoArray3<f64>>,
        t_start: Time<f64>,
        t_end: Time<f64>,
        mut dt: Time<f64>,
        grid: &Grid,
    ) -> KwaversResult<GpuRealtimeSimulationStatistics> {
        self.start_time = Some(Instant::now());

        let mut t = t_start;
        let mut step = 0u64;

        while t.into_base() < t_end.into_base() {
            if self.config.adaptive_timestepping {
                dt = self.adjust_timestep(dt, t, t_end);
            }

            let result = self.step(fields, dt, t, grid)?;

            t = Time::from_base(t.into_base() + result.dt.into_base());
            step += 1;

            if step.is_multiple_of(self.config.checkpoint_interval as u64)
                && self.config.enable_async_io
                && self.config.verbose
            {
                debug!("Checkpoint at step {} (time={:.3e} s)", step, t.into_base());
            }
        }

        let elapsed = self
            .start_time
            .take()
            .expect("invariant: simulate initializes start_time before measuring")
            .elapsed()
            .as_secs_f64();
        let metrics = self.monitor.get_metrics();

        Ok(GpuRealtimeSimulationStatistics {
            total_wall_time: Time::from_base(elapsed),
            total_simulation_time: Time::from_base(t.into_base() - t_start.into_base()),
            num_steps: step,
            budget_violations: self.monitor.budget_violations(),
            metrics,
        })
    }

    /// Get current performance metrics.
    pub fn get_metrics(&self) -> GpuStepMetrics {
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
    pub(crate) fn adjust_timestep(
        &self,
        current_dt: Time<f64>,
        time: Time<f64>,
        t_end: Time<f64>,
    ) -> Time<f64> {
        let current_dt_seconds = current_dt.into_base();
        let max_dt = (t_end.into_base() - time.into_base()).max(current_dt_seconds);
        let safe_dt = current_dt_seconds * self.config.cfl_safety_factor.into_base();
        Time::from_base(safe_dt.min(max_dt))
    }
}
