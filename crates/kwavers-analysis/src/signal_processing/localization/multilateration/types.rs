//! Configuration type for multilateration.

use kwavers_core::constants::SOUND_SPEED_TISSUE;
use serde::{Deserialize, Serialize};

/// Configuration for multilateration algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilaterationConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,

    /// Maximum iterations for Levenberg-Marquardt refinement
    pub max_iterations: usize,

    /// Convergence tolerance for position update (m)
    pub convergence_tolerance: f64,

    /// Initial damping parameter for Levenberg-Marquardt
    pub initial_damping: f64,

    /// Damping adjustment factor (increase on failure, decrease on success)
    pub damping_factor: f64,

    /// Use weighted least squares (requires sensor_uncertainties)
    pub use_weighted_ls: bool,

    /// Initial guess for source position (m), None = centroid
    pub initial_guess: Option<[f64; 3]>,
}

impl Default for MultilaterationConfig {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            max_iterations: 50,
            convergence_tolerance: 1e-6,
            initial_damping: 1e-3,
            damping_factor: 10.0,
            use_weighted_ls: false,
            initial_guess: None,
        }
    }
}
