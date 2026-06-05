//! Plain data types for the realtime simulation loop.

use crate::backend::performance_monitor::GpuStepMetrics;

/// Real-time simulation configuration.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Target execution time per step (milliseconds).
    pub budget_ms: f64,

    /// Enable adaptive time stepping.
    pub adaptive_timestepping: bool,

    /// CFL safety factor (typically 0.8–0.95).
    pub cfl_safety_factor: f64,

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
            budget_ms: 10.0,
            adaptive_timestepping: true,
            cfl_safety_factor: 0.9,
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
    pub dt: f64,

    /// Total time after this step.
    pub time: f64,

    /// Wall time for execution (milliseconds).
    pub wall_time_ms: f64,

    /// Whether step was within budget.
    pub within_budget: bool,

    /// Number of GPU kernels executed.
    pub kernels_executed: usize,
}

/// Simulation statistics.
#[derive(Debug, Clone)]
pub struct GpuRealtimeSimulationStatistics {
    /// Total wall time (seconds).
    pub total_wall_time_seconds: f64,

    /// Total simulation time (seconds).
    pub total_simulation_time_seconds: f64,

    /// Number of steps executed.
    pub num_steps: u64,

    /// Number of budget violations.
    pub budget_violations: u64,

    /// Performance metrics.
    pub metrics: GpuStepMetrics,
}
