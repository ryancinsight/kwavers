//! Plain data types for the realtime simulation loop.

use crate::backend::performance_monitor::GpuStepMetrics;
use aequitas::systems::si::quantities::{Dimensionless, Time};

/// Real-time simulation configuration.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Target execution time per step.
    pub budget: Time<f64>,

    /// Enable adaptive time stepping.
    pub adaptive_timestepping: bool,

    /// CFL safety factor (typically 0.8–0.95).
    pub cfl_safety_factor: Dimensionless<f64>,

    /// Checkpoint interval (save every N steps).
    pub checkpoint_interval: usize,

    /// Enable async I/O for checkpoints.
    pub enable_async_io: bool,

    /// Verbose output.
    pub verbose: bool,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            budget: Time::from_base(0.010),
            adaptive_timestepping: true,
            cfl_safety_factor: Dimensionless::from_base(0.9),
            checkpoint_interval: 100,
            enable_async_io: true,
            verbose: false,
        }
    }
}

/// Result of single timestep execution.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Timestep size used.
    pub dt: Time<f64>,

    /// Total time after this step.
    pub time: Time<f64>,

    /// Wall time for execution.
    pub wall_time: Time<f64>,

    /// Whether step was within budget.
    pub within_budget: bool,

    /// Number of GPU kernels executed.
    pub kernels_executed: usize,
}

/// Simulation statistics.
#[derive(Debug, Clone)]
pub struct GpuRealtimeSimulationStatistics {
    /// Total wall time.
    pub total_wall_time: Time<f64>,

    /// Total simulation time.
    pub total_simulation_time: Time<f64>,

    /// Number of steps executed.
    pub num_steps: u64,

    /// Number of budget violations.
    pub budget_violations: u64,

    /// Performance metrics.
    pub metrics: GpuStepMetrics,
}
