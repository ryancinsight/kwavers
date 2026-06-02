//! Types for trilateration source localization.

use kwavers_core::constants::SOUND_SPEED_TISSUE;
use serde::{Deserialize, Serialize};

/// Configuration for trilateration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrilaterationConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,
    /// Maximum number of iterations for iterative solver
    pub max_iterations: usize,
    /// Convergence tolerance (m)
    pub convergence_tolerance: f64,
    /// Initial guess for source position (m), None = centroid of sensors
    pub initial_guess: Option<[f64; 3]>,
}

impl Default for TrilaterationConfig {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            max_iterations: 100,
            convergence_tolerance: 1e-6,
            initial_guess: None,
        }
    }
}

/// Localization result with uncertainty
#[derive(Debug, Clone)]
pub struct LocalizationResult {
    /// Estimated source position (m)
    pub position: [f64; 3],
    /// Position uncertainty (standard deviation, m)
    pub uncertainty: f64,
    /// Residual error (m)
    pub residual: f64,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Whether solution converged
    pub converged: bool,
}
