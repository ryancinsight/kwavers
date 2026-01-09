//! Progress reporting system
//!
//! This module provides a flexible progress reporting system that allows
//! different simulation types to report progress in a standardized way.

use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A trait for types that can be reported as progress updates
/// This allows different simulation types to define custom progress data
pub trait ProgressData: Send + Sync {}

/// Generic progress reporter trait - decoupled from specific data structures
/// This follows the principle of designing to interfaces, not implementations
pub trait ProgressReporter: Send + Sync {
    /// Report progress with any type implementing `ProgressData`
    fn report(&mut self, progress_json: &str);

    /// Called when simulation starts
    fn on_start(&mut self, _total_steps: usize, _dt: f64) {}

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
    pub step_duration: Duration,
    pub estimated_remaining: Duration,
    pub fields_summary: FieldsSummary,
}

// Implement ProgressData for the standard ProgressUpdate
impl ProgressData for ProgressUpdate {}

/// Flexible field summary using `HashMap` for extensibility
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
    #[must_use]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Insert a field value
    pub fn insert(&mut self, key: &str, value: f64) {
        self.0.insert(key.to_string(), value);
    }

    /// Get a field value
    #[must_use]
    pub fn get(&self, key: &str) -> Option<f64> {
        self.0.get(key).copied()
    }

    /// Create a standard acoustic simulation summary
    #[must_use]
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
#[derive(Debug)]
pub struct ConsoleProgressReporter {
    last_report_time: Instant,
    report_interval: Duration,
    start_time: Instant,
}

impl Default for ConsoleProgressReporter {
    fn default() -> Self {
        Self {
            last_report_time: Instant::now(),
            report_interval: Duration::from_secs(10),
            start_time: Instant::now(),
        }
    }
}

impl ProgressReporter for ConsoleProgressReporter {
    fn on_start(&mut self, total_steps: usize, dt: f64) {
        self.start_time = Instant::now();
        log::info!(
            "Starting simulation: {} steps, dt = {:.6e}s, total time = {:.6e}s",
            total_steps,
            dt,
            total_steps as f64 * dt
        );
    }

    fn report(&mut self, progress_json: &str) {
        let now = Instant::now();
        if now.duration_since(self.last_report_time) >= self.report_interval {
            log::info!("Progress: {}", progress_json);
            self.last_report_time = now;
        }
    }

    fn on_complete(&mut self) {
        let total_time = self.start_time.elapsed();
        log::info!("Simulation completed in {:?}", total_time);
    }
}
