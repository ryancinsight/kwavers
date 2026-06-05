//! ConformalConfig — confidence and calibration parameters

/// Configuration for conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Desired confidence level (0-1)
    pub confidence_level: f64,
    /// Size of calibration set
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
