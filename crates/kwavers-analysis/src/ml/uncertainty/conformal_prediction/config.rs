//! Conformal confidence and calibration parameters.

/// Configuration for conformal prediction.
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Desired confidence level in `(0, 1)`.
    pub confidence_level: f64,
    /// Required calibration-set size.
    pub calibration_size: usize,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            calibration_size: 1000,
        }
    }
}
