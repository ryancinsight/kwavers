//! Array validation and performance metrics

use super::constants::{GRATING_LOBE_THRESHOLD_RATIO, MAX_STEERING_ANGLE};
use aequitas::systems::si::quantities::{Angle, Pressure, Volume};
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::KwaversResult;

/// Array validator for safety and performance checks
#[derive(Debug, Clone)]
pub struct ArrayValidator {
    /// Maximum allowed pressure in pascals.
    max_pressure: Pressure<f64>,
}

impl Default for ArrayValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayValidator {
    /// Create new validator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_pressure: Pressure::from_base(10.0 * MPA_TO_PA), // 10 MPa safety limit
        }
    }

    /// Validate array configuration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self, metrics: &HemisphericalArrayMetrics) -> KwaversResult<()> {
        // Check safety limits
        if metrics.peak_pressure.into_base() > self.max_pressure.into_base() {
            log::warn!(
                "Peak pressure exceeds safety limit: {} Pa",
                metrics.peak_pressure.into_base()
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
pub struct HemisphericalArrayMetrics {
    /// Peak pressure at focus in pascals.
    pub peak_pressure: Pressure<f64>,
    /// Focal volume (-6 dB) in cubic metres.
    pub focal_volume: Volume<f64>,
    /// Grating lobe level (ratio)
    pub grating_lobe_level: f64,
    /// Power efficiency
    pub efficiency: f64,
    /// Steering range in radians.
    pub steering_range: Angle<f64>,
}

impl Default for HemisphericalArrayMetrics {
    fn default() -> Self {
        Self {
            peak_pressure: Pressure::from_base(0.0),
            focal_volume: Volume::from_base(0.0),
            grating_lobe_level: 0.0,
            efficiency: 1.0,
            steering_range: MAX_STEERING_ANGLE,
        }
    }
}
