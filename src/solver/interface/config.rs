//! Common solver configuration
//!
//! This module provides common configuration structures used
//! by multiple solver implementations.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Common solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum number of time steps
    pub max_steps: usize,

    /// Time step size
    pub dt: f64,

    /// Courant-Friedrichs-Lewy number for stability
    pub cfl: f64,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// Enable adaptive mesh refinement
    pub enable_amr: bool,

    /// Progress reporting interval
    pub progress_interval: Duration,

    /// Validation mode (for testing)
    pub validation_mode: bool,

    /// Detailed logging
    pub detailed_logging: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_steps: 1000,
            dt: 1e-7,
            cfl: 0.3,
            enable_gpu: false,
            enable_amr: false,
            progress_interval: Duration::from_secs(10),
            validation_mode: false,
            detailed_logging: false,
        }
    }
}

impl SolverConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_steps == 0 {
            return Err("max_steps must be greater than 0".to_string());
        }

        if self.dt <= 0.0 {
            return Err("dt must be positive".to_string());
        }

        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err("cfl must be between 0 and 1".to_string());
        }

        Ok(())
    }

    /// Create a configuration optimized for accuracy
    pub fn accuracy_optimized() -> Self {
        Self {
            cfl: 0.1,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            cfl: 0.5,
            enable_gpu: true,
            progress_interval: Duration::from_secs(30),
            ..Default::default()
        }
    }
}
