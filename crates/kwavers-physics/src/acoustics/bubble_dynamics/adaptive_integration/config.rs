use kwavers_core::constants::numerical::{
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Default config satisfies the basic ordering invariant: dt_min < dt_max.
    #[test]
    fn default_dt_min_less_than_dt_max() {
        let cfg = AdaptiveBubbleConfig::default();
        assert!(
            cfg.dt_min < cfg.dt_max,
            "dt_min ({}) must be < dt_max ({})",
            cfg.dt_min,
            cfg.dt_max
        );
    }

    /// Default safety_factor is in the physically meaningful range (0.5, 1.0).
    #[test]
    fn default_safety_factor_in_valid_range() {
        let cfg = AdaptiveBubbleConfig::default();
        assert!(
            cfg.safety_factor > 0.5 && cfg.safety_factor < 1.0,
            "safety_factor {} must be in (0.5, 1.0)",
            cfg.safety_factor
        );
    }

    /// Clone produces an equal copy.
    #[test]
    fn clone_preserves_all_fields() {
        let cfg = AdaptiveBubbleConfig::default();
        let cloned = cfg.clone();
        assert!((cfg.dt_max - cloned.dt_max).abs() < 1e-30);
        assert!((cfg.rtol - cloned.rtol).abs() < 1e-30);
        assert_eq!(cfg.max_substeps, cloned.max_substeps);
    }
}
