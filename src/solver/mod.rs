// src/solver/mod.rs
// Clean module structure focusing only on the plugin-based architecture

// Core solver modules
pub mod pstd;
pub mod fdtd;
pub mod hybrid;
pub mod time_integration;
pub mod spectral_dg;
pub mod imex;
pub mod amr;
pub mod kspace_correction;
pub mod heterogeneous_handler;
pub mod cpml_integration;
pub mod validation;
pub mod workspace;
pub mod time_reversal;
pub mod thermal_diffusion;
pub mod reconstruction;

// The new plugin-based architecture - the primary solver
pub mod plugin_based_solver;

// Re-export the main solver type for convenience
pub use plugin_based_solver::PluginBasedSolver;

// Re-export commonly used types from submodules
pub use pstd::PstdConfig;
pub use fdtd::FdtdConfig;
pub use imex::{IMEXIntegrator, IMEXSchemeType};
pub use amr::{AMRConfig, AMRManager};

// Progress reporting trait for the plugin-based solver
pub trait ProgressReporter: Send + Sync {
    /// Report progress update
    fn report(&mut self, progress: &ProgressUpdate);
    
    /// Called when simulation starts
    fn on_start(&mut self, total_steps: usize, dt: f64) {}
    
    /// Called when simulation completes
    fn on_complete(&mut self) {}
}

/// Progress update information
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub current_step: usize,
    pub total_steps: usize,
    pub current_time: f64,
    pub total_time: f64,
    pub step_duration: std::time::Duration,
    pub estimated_remaining: std::time::Duration,
    pub fields_summary: FieldsSummary,
}

/// Summary of field values for progress reporting
#[derive(Debug, Clone)]
pub struct FieldsSummary {
    pub max_pressure: f64,
    pub max_velocity: f64,
    pub max_temperature: f64,
    pub total_energy: f64,
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
    
    fn report(&mut self, progress: &ProgressUpdate) {
        let now = std::time::Instant::now();
        
        // Report at start, end, or at intervals
        if progress.current_step == 0 
            || progress.current_step == progress.total_steps - 1
            || now.duration_since(self.last_report_time) >= self.report_interval {
            
            let percent = (progress.current_step as f64 / progress.total_steps as f64) * 100.0;
            let elapsed = now.duration_since(self.start_time);
            
            log::info!(
                "Step {}/{} ({:.1}%) | t={:.6e}s | Max P={:.2e} | ETA: {}",
                progress.current_step,
                progress.total_steps,
                percent,
                progress.current_time,
                progress.fields_summary.max_pressure,
                format_duration(progress.estimated_remaining)
            );
            
            self.last_report_time = now;
        }
    }
    
    fn on_complete(&mut self) {
        let elapsed = std::time::Instant::now().duration_since(self.start_time);
        log::info!("Simulation completed in {}", format_duration(elapsed));
    }
}

/// Format duration for display
fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Null progress reporter for when progress reporting is not needed
pub struct NullProgressReporter;

impl ProgressReporter for NullProgressReporter {
    fn report(&mut self, _progress: &ProgressUpdate) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_secs(45)), "45s");
        assert_eq!(format_duration(std::time::Duration::from_secs(125)), "2m 5s");
        assert_eq!(format_duration(std::time::Duration::from_secs(3725)), "1h 2m 5s");
    }
}