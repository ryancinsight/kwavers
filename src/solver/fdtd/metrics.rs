//! Performance metrics for FDTD solver

use std::time::Duration;

/// Dedicated struct for FDTD performance metrics
#[derive(Debug, Default, Clone))]
pub struct FdtdMetrics {
    /// Time spent updating pressure field
    pub update_pressure_time: Duration,
    /// Time spent updating velocity fields
    pub update_velocity_time: Duration,
    /// Time spent computing divergence
    pub divergence_time: Duration,
    /// Time spent computing gradients
    pub gradient_time: Duration,
    /// Time spent on boundary conditions
    pub boundary_time: Duration,
    /// Number of CFL stability checks performed
    pub cfl_checks: u64,
    /// Maximum CFL number encountered
    pub max_cfl_number: f64,
    /// Total number of time steps
    pub time_steps: u64,
}

impl FdtdMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge metrics from another instance
    pub fn merge(&mut self, other: &FdtdMetrics) {
        self.update_pressure_time += other.update_pressure_time;
        self.update_velocity_time += other.update_velocity_time;
        self.divergence_time += other.divergence_time;
        self.gradient_time += other.gradient_time;
        self.boundary_time += other.boundary_time;
        self.cfl_checks += other.cfl_checks;
        self.max_cfl_number = self.max_cfl_number.max(other.max_cfl_number);
        self.time_steps += other.time_steps;
    }

    /// Get average time per time step
    pub fn avg_time_per_step(&self) -> Duration {
        if self.time_steps == 0 {
            Duration::ZERO
        } else {
            let total = self.update_pressure_time + self.update_velocity_time;
            total / self.time_steps as u32
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "FDTD Metrics: {} steps, {:.2} ms/step avg, max CFL: {:.3}",
            self.time_steps,
            self.avg_time_per_step().as_secs_f64() * 1000.0,
            self.max_cfl_number
        )
    }
}
