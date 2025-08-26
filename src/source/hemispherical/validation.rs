//! Array validation and performance metrics

use super::constants::*;
use crate::error::KwaversResult;

/// Array validator for safety and performance checks
#[derive(Debug, Clone)]
pub struct ArrayValidator {
    /// Maximum allowed pressure (Pa)
    max_pressure: f64,
    /// Maximum temperature rise (K)
    max_temp_rise: f64,
}

impl ArrayValidator {
    /// Create new validator
    pub fn new() -> Self {
        Self {
            max_pressure: 10e6,  // 10 MPa safety limit
            max_temp_rise: 10.0, // 10K safety limit
        }
    }

    /// Validate array configuration
    pub fn validate(&self, metrics: &PerformanceMetrics) -> KwaversResult<()> {
        // Check safety limits
        if metrics.peak_pressure > self.max_pressure {
            log::warn!(
                "Peak pressure exceeds safety limit: {} Pa",
                metrics.peak_pressure
            );
        }

        if metrics.grating_lobe_level > GRATING_LOBE_THRESHOLD_RATIO {
            log::warn!(
                "Grating lobes exceed threshold: {}",
                metrics.grating_lobe_level
            );
        }

        Ok(())
    }
}

/// Performance metrics for array evaluation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Peak pressure at focus (Pa)
    pub peak_pressure: f64,
    /// Focal volume (-6dB) (mm³)
    pub focal_volume: f64,
    /// Grating lobe level (ratio)
    pub grating_lobe_level: f64,
    /// Power efficiency
    pub efficiency: f64,
    /// Steering range (radians)
    pub steering_range: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            peak_pressure: 0.0,
            focal_volume: 0.0,
            grating_lobe_level: 0.0,
            efficiency: 1.0,
            steering_range: MAX_STEERING_ANGLE_RAD,
        }
    }
}
