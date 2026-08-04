//! Array validation and performance metrics

use super::constants::{GRATING_LOBE_THRESHOLD_RATIO, MAX_STEERING_ANGLE_RAD};
use aequitas::systems::si::quantities::{Angle, Dimensionless, Pressure, Volume};
use aequitas::systems::si::units::{Pascal, Radian};
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::KwaversResult;

/// Array validator for safety and performance checks
#[derive(Debug, Clone)]
pub struct ArrayValidator {
    /// Maximum allowed pressure (Pa)
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
            max_pressure: Pressure::from_unit::<Pascal>(10.0 * MPA_TO_PA),
        }
    }

    /// Validate array configuration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self, metrics: &HemisphericalArrayMetrics) -> KwaversResult<()> {
        // Check safety limits
        if metrics.peak_pressure.in_unit::<Pascal>() > self.max_pressure.in_unit::<Pascal>() {
            log::warn!(
                "Peak pressure exceeds safety limit: {} Pa",
                metrics.peak_pressure.in_unit::<Pascal>()
            );
        }

        if metrics.grating_lobe_level.into_base() > GRATING_LOBE_THRESHOLD_RATIO {
            log::warn!(
                "Grating lobes exceed threshold: {}",
                metrics.grating_lobe_level.into_base()
            );
        }

        Ok(())
    }
}

/// Performance metrics for array evaluation
#[derive(Debug, Clone)]
pub struct HemisphericalArrayMetrics {
    /// Peak pressure at focus (Pa)
    pub peak_pressure: Pressure<f64>,
    /// Focal volume (-6dB) in cubic metres.
    pub focal_volume: Volume<f64>,
    /// Grating lobe level (ratio)
    pub grating_lobe_level: Dimensionless<f64>,
    /// Power efficiency
    pub efficiency: Dimensionless<f64>,
    /// Steering range (radians)
    pub steering_range: Angle<f64>,
}

impl Default for HemisphericalArrayMetrics {
    fn default() -> Self {
        Self {
            peak_pressure: Pressure::from_unit::<Pascal>(0.0),
            focal_volume: Volume::from_unit::<aequitas::systems::si::units::CubicMeter>(0.0),
            grating_lobe_level: Dimensionless::from_base(0.0),
            efficiency: Dimensionless::from_base(1.0),
            steering_range: Angle::from_unit::<Radian>(MAX_STEERING_ANGLE_RAD),
        }
    }
}
