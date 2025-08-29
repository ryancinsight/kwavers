// src/solver/mod.rs
// Clean module structure focusing only on the plugin-based architecture

// Core solver modules
pub mod amr;
pub mod cpml_integration;
pub mod fdtd;
pub mod fdtd_proper;      // Optimized FDTD with ghost cells
pub mod heterogeneous;
pub mod hybrid;
pub mod imex;
pub mod kspace_correction;
pub mod pstd;
pub mod pstd_solver;      // New proper PSTD solver
pub mod pstd_proper;      // Optimized PSTD with cached FFTs
pub mod spectral_dg;
pub mod time_integration;
pub mod validation;

// Re-export field indices from the single source of truth
pub use crate::physics::field_indices::{
    PRESSURE_IDX as P_IDX, STRESS_XX_IDX as SXX_IDX, STRESS_XY_IDX as SXY_IDX,
    STRESS_XZ_IDX as SXZ_IDX, STRESS_YY_IDX as SYY_IDX, STRESS_YZ_IDX as SYZ_IDX,
    STRESS_ZZ_IDX as SZZ_IDX, TOTAL_FIELDS, VX_IDX, VY_IDX, VZ_IDX,
};
pub mod reconstruction;
pub mod thermal_diffusion;
pub mod time_reversal;
pub mod workspace;

// The new plugin-based architecture - the primary solver
pub mod plugin_based;
pub use plugin_based::PluginBasedSolver;

// Re-export commonly used types from submodules
pub use amr::{AMRSolver, MemoryStats};
pub use fdtd::FdtdConfig;
pub use fdtd_proper::{ProperFdtdPlugin, FdtdConfig as ProperFdtdConfig, MetricType};
pub use imex::{IMEXIntegrator, IMEXSchemeType};
pub use pstd::PstdConfig;
pub use pstd_solver::PstdSolver;     // Export new PSTD solver
pub use pstd_proper::{ProperPstdPlugin, PstdConfig as ProperPstdConfig};

use serde::Serialize;
use std::collections::HashMap;

// Flexible progress reporting system

/// A trait for types that can be reported as progress updates
/// This allows different simulation types to define custom progress data
pub trait ProgressData: Send + Sync {}

/// Generic progress reporter trait - decoupled from specific data structures
/// This follows the principle of designing to interfaces, not implementations
pub trait ProgressReporter: Send + Sync {
    /// Report progress with any type implementing ProgressData
    fn report(&mut self, progress_json: &str);

    /// Called when simulation starts
    fn on_start(&mut self, total_steps: usize, dt: f64) {}

    /// Called when simulation completes
    fn on_complete(&mut self) {}
}

/// Standard progress update information for acoustic simulations
#[derive(Debug, Clone, Serialize)]
pub struct ProgressUpdate {
    pub current_step: usize,
    pub total_steps: usize,
    pub current_time: f64,
    pub total_time: f64,
    pub step_duration: std::time::Duration,
    pub estimated_remaining: std::time::Duration,
    pub fields_summary: FieldsSummary,
}

// Implement ProgressData for the standard ProgressUpdate
impl ProgressData for ProgressUpdate {}

/// Flexible field summary using HashMap for extensibility
/// This allows any simulation type to report arbitrary metrics
#[derive(Debug, Clone, Serialize)]
pub struct FieldsSummary(HashMap<String, f64>);

impl Default for FieldsSummary {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldsSummary {
    /// Create a new empty field summary
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Insert a field value
    pub fn insert(&mut self, key: &str, value: f64) {
        self.0.insert(key.to_string(), value);
    }

    /// Get a field value
    pub fn get(&self, key: &str) -> Option<f64> {
        self.0.get(key).copied()
    }

    /// Create a standard acoustic simulation summary
    pub fn acoustic(
        max_pressure: f64,
        max_velocity: f64,
        max_temperature: f64,
        total_energy: f64,
    ) -> Self {
        let mut summary = Self::new();
        summary.insert("max_pressure", max_pressure);
        summary.insert("max_velocity", max_velocity);
        summary.insert("max_temperature", max_temperature);
        summary.insert("total_energy", total_energy);
        summary
    }
}

/// Console progress reporter implementation
pub struct ConsoleProgressReporter {
    last_report_time: std::time::Instant,
    report_interval: std::time::Duration,
    start_time: std::time::Instant,
}

impl Default for ConsoleProgressReporter {
    fn default() -> Self {
        Self {
            last_report_time: std::time::Instant::now(),
            report_interval: std::time::Duration::from_secs(10),
            start_time: std::time::Instant::now(),
        }
    }
}

impl ProgressReporter for ConsoleProgressReporter {
    fn on_start(&mut self, total_steps: usize, dt: f64) {
        self.start_time = std::time::Instant::now();
        log::info!(
            "Starting simulation: {} steps, dt = {:.6e}s, total time = {:.6e}s",
            total_steps,
            dt,
            total_steps as f64 * dt
        );
    }

    fn report(&mut self, progress_json: &str) {
        let now = std::time::Instant::now();

        // Parse the progress data from JSON for flexible handling
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(progress_json) {
            // Try to extract standard fields if they exist
            let current_step = json
                .get("current_step")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total_steps = json
                .get("total_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(1);

            // Report at start, end, or at intervals
            if current_step == 0
                || current_step == total_steps - 1
                || now.duration_since(self.last_report_time) >= self.report_interval
            {
                let percent = (current_step as f64 / total_steps as f64) * 100.0;

                // Extract other fields if available
                let current_time = json
                    .get("current_time")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let max_pressure = json
                    .get("fields_summary")
                    .and_then(|fs| fs.get("max_pressure"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                log::info!(
                    "Step {}/{} ({:.1}%) | t={:.6e}s | Progress: {}",
                    current_step,
                    total_steps,
                    percent,
                    current_time,
                    serde_json::to_string(&json).unwrap_or_default()
                );

                self.last_report_time = now;
            }
        }
    }

    fn on_complete(&mut self) {
        let elapsed = std::time::Instant::now().duration_since(self.start_time);
        log::info!(
            "Simulation completed in {}",
            crate::utils::format::format_duration(elapsed)
        );
    }
}

// Note: format_duration has been moved to utils::format module for reusability

/// Null progress reporter for when progress reporting is not needed
#[derive(Debug)]
pub struct NullProgressReporter;

impl ProgressReporter for NullProgressReporter {
    fn report(&mut self, _progress_json: &str) {}
}

/// Asynchronous console reporter for non-blocking progress reporting
/// This prevents I/O operations from affecting simulation performance
pub struct AsyncConsoleReporter {
    sender: std::sync::mpsc::Sender<String>,
    last_report_time: std::time::Instant,
    report_interval: std::time::Duration,
    start_time: std::time::Instant,
}

impl AsyncConsoleReporter {
    /// Create a new async console reporter with a dedicated reporting thread
    pub fn new() -> Self {
        use std::sync::mpsc;
        use std::thread;

        let (sender, receiver) = mpsc::channel();

        // Spawn a dedicated thread for console I/O
        thread::spawn(move || {
            for message in receiver {
                // Perform the actual printing in a separate thread
                // This ensures the simulation loop is never blocked by I/O
                println!("{}", message);
            }
        });

        Self {
            sender,
            last_report_time: std::time::Instant::now(),
            report_interval: std::time::Duration::from_secs(10),
            start_time: std::time::Instant::now(),
        }
    }

    /// Set the reporting interval
    pub fn with_interval(mut self, interval: std::time::Duration) -> Self {
        self.report_interval = interval;
        self
    }
}

impl ProgressReporter for AsyncConsoleReporter {
    fn on_start(&mut self, total_steps: usize, dt: f64) {
        self.start_time = std::time::Instant::now();
        let message = format!(
            "Starting simulation: {} steps, dt = {:.6e}s, total time = {:.6e}s",
            total_steps,
            dt,
            total_steps as f64 * dt
        );
        // Use try_send to avoid blocking if channel is full
        let _ = self.sender.send(message);
    }

    fn report(&mut self, progress_json: &str) {
        let now = std::time::Instant::now();

        // Only report at intervals to avoid overwhelming the channel
        if now.duration_since(self.last_report_time) >= self.report_interval {
            // Use try_send to avoid blocking the simulation
            // If the channel is full, we skip this update rather than block
            let _ = self.sender.send(format!("Progress: {}", progress_json));
            self.last_report_time = now;
        }
    }

    fn on_complete(&mut self) {
        let elapsed = std::time::Instant::now().duration_since(self.start_time);
        let message = format!("Simulation completed in {:?}", elapsed);
        let _ = self.sender.send(message);
    }
}

impl Default for AsyncConsoleReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flexible_fields_summary() {
        let mut summary = FieldsSummary::new();
        summary.insert("custom_metric", 42.0);
        summary.insert("another_metric", 3.14);

        assert_eq!(summary.get("custom_metric"), Some(42.0));
        assert_eq!(summary.get("another_metric"), Some(3.14));
        assert_eq!(summary.get("nonexistent"), None);
    }

    #[test]
    fn test_acoustic_fields_summary() {
        let summary = FieldsSummary::acoustic(100.0, 50.0, 300.0, 1000.0);

        assert_eq!(summary.get("max_pressure"), Some(100.0));
        assert_eq!(summary.get("max_velocity"), Some(50.0));
        assert_eq!(summary.get("max_temperature"), Some(300.0));
        assert_eq!(summary.get("total_energy"), Some(1000.0));
    }

    #[test]
    fn test_progress_data_trait() {
        // Custom progress type for testing
        #[derive(Debug, Clone, Serialize)]
        struct CustomProgress {
            iteration: usize,
            residual: f64,
        }

        impl ProgressData for CustomProgress {}

        let progress = CustomProgress {
            iteration: 10,
            residual: 0.001,
        };

        // Should be able to serialize
        let json = serde_json::to_string(&progress).unwrap();
        assert!(json.contains("iteration"));
        assert!(json.contains("residual"));
    }
}
