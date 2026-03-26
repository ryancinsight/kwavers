use crate::core::constants::numerical::{
    DEFAULT_ABSOLUTE_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE, MAX_SUBSTEPS, MAX_TIME_STEP,
    MAX_TIME_STEP_DECREASE, MAX_TIME_STEP_INCREASE, MIN_NUMERICAL_TIME_STEP, SAFETY_FACTOR,
};

/// Configuration for adaptive bubble integration
#[derive(Debug, Clone)]
pub struct AdaptiveBubbleConfig {
    /// Maximum time step (limited by acoustic period)
    pub dt_max: f64,
    /// Minimum time step (for extreme collapse)
    pub dt_min: f64,
    /// Relative tolerance for error control
    pub rtol: f64,
    /// Absolute tolerance for error control
    pub atol: f64,
    /// Safety factor for time step adjustment (0.8-0.95 typical)
    pub safety_factor: f64,
    /// Maximum factor to increase time step
    pub dt_increase_max: f64,
    /// Maximum factor to decrease time step
    pub dt_decrease_max: f64,
    /// Maximum number of sub-steps per main time step
    pub max_substeps: usize,
    /// Enable stability monitoring
    pub monitor_stability: bool,
}

impl Default for AdaptiveBubbleConfig {
    fn default() -> Self {
        Self {
            dt_max: MAX_TIME_STEP,
            dt_min: MIN_NUMERICAL_TIME_STEP,
            rtol: DEFAULT_RELATIVE_TOLERANCE,
            atol: DEFAULT_ABSOLUTE_TOLERANCE,
            safety_factor: SAFETY_FACTOR,
            dt_increase_max: MAX_TIME_STEP_INCREASE,
            dt_decrease_max: MAX_TIME_STEP_DECREASE,
            max_substeps: MAX_SUBSTEPS,
            monitor_stability: true,
        }
    }
}
