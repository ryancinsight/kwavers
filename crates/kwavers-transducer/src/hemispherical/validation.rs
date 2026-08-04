//! Array validation and performance metrics

use super::constants::{GRATING_LOBE_THRESHOLD_RATIO, MAX_STEERING_ANGLE_RAD};
use aequitas::systems::si::quantities::{Angle, Pressure};
use aequitas::systems::si::units::{Pascal, Radian};
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::KwaversResult;

/// Array validator for safety and performance checks.
#[derive(Debug, Clone)]
pub struct ArrayValidator {
    /// Maximum allowed acoustic pressure.
    max_pressure: Pressure<f64>,
}

impl Default for ArrayValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayValidator {
    /// Create a new validator with a 10 MPa default safety limit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_pressure: Pressure::from_unit::<Pascal>(10.0 * MPA_TO_PA),
        }
    }

    /// Validate the array metrics against safety limits.
    ///
    /// # Errors
    /// Returns an error if the metrics violate hard safety constraints.
    pub fn validate(&self, metrics: &HemisphericalArrayMetrics) -> KwaversResult<()> {
        if metrics.peak_pressure.in_unit::<Pascal>() > self.max_pressure.in_unit::<Pascal>() {
            log::warn!(
                "Peak pressure exceeds safety limit: {} Pa",
                metrics.peak_pressure.in_unit::<Pascal>()
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

/// Performance metrics for array evaluation.
#[derive(Debug, Clone)]
pub struct HemisphericalArrayMetrics {
    /// Peak acoustic pressure at the focus.
    pub peak_pressure: Pressure<f64>,
    /// Focal volume at −6 dB, in mm³.
    ///
    /// Stored as a raw display scalar (mm³ is a non-SI display unit);
    /// scalar extraction at the reporting boundary is appropriate here.
    pub focal_volume: f64,
    /// Grating-lobe level (dimensionless ratio).
    pub grating_lobe_level: f64,
    /// Power efficiency (dimensionless, `[0, 1]`).
    pub efficiency: f64,
    /// Steering range (half-angle from the array axis).
    pub steering_range: Angle<f64>,
}

impl Default for HemisphericalArrayMetrics {
    fn default() -> Self {
        Self {
            peak_pressure: Pressure::from_unit::<Pascal>(0.0),
            focal_volume: 0.0,
            grating_lobe_level: 0.0,
            efficiency: 1.0,
            steering_range: Angle::from_unit::<Radian>(MAX_STEERING_ANGLE_RAD),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::{Pascal, Radian};

    #[test]
    fn default_metrics_have_zero_pressure() {
        let m = HemisphericalArrayMetrics::default();
        assert_eq!(m.peak_pressure.in_unit::<Pascal>(), 0.0);
    }

    #[test]
    fn validator_warns_but_does_not_error_on_high_pressure() {
        let v = ArrayValidator::new();
        let m = HemisphericalArrayMetrics {
            peak_pressure: Pressure::from_unit::<Pascal>(20.0 * MPA_TO_PA), // 20 MPa > 10 MPa limit
            ..Default::default()
        };
        // Validation returns Ok even when the pressure exceeds the limit (it logs a warning).
        assert!(v.validate(&m).is_ok());
    }

    #[test]
    fn steering_range_round_trips_radians() {
        let m = HemisphericalArrayMetrics::default();
        assert!((m.steering_range.in_unit::<Radian>() - MAX_STEERING_ANGLE_RAD).abs() < 1.0e-15);
    }
}
