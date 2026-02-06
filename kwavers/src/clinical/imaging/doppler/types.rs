//! Common types for Doppler velocity estimation

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Flow direction classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowDirection {
    /// Flow toward the transducer (positive velocity)
    Toward,
    /// Flow away from the transducer (negative velocity)
    Away,
    /// No significant flow detected
    None,
}

impl FlowDirection {
    /// Classify flow direction from velocity
    pub fn from_velocity(velocity: f64, threshold: f64) -> Self {
        if velocity > threshold {
            FlowDirection::Toward
        } else if velocity < -threshold {
            FlowDirection::Away
        } else {
            FlowDirection::None
        }
    }
}

/// Velocity estimate with quality metrics
#[derive(Debug, Clone)]
pub struct VelocityEstimate {
    /// Estimated velocity (m/s)
    pub velocity: f64,

    /// Variance of the estimate (0-1, lower is better)
    pub variance: f64,

    /// Flow direction
    pub direction: FlowDirection,

    /// Signal-to-noise ratio (dB)
    pub snr: Option<f64>,
}

impl VelocityEstimate {
    /// Create a new velocity estimate
    pub fn new(velocity: f64, variance: f64) -> Self {
        let direction = FlowDirection::from_velocity(velocity, 0.01);
        Self {
            velocity,
            variance,
            direction,
            snr: None,
        }
    }

    /// Check if estimate is reliable based on variance threshold
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.variance < threshold
    }
}

/// Result of Doppler velocity estimation
#[derive(Debug, Clone)]
pub struct DopplerResult {
    /// 2D velocity map (m/s)
    pub velocity: Array2<f64>,

    /// Variance map (quality metric)
    pub variance: Array2<f64>,

    /// Power map (signal intensity)
    pub power: Option<Array2<f64>>,

    /// Configuration parameters used
    pub center_frequency: f64,
    pub prf: f64,
}

impl DopplerResult {
    /// Get velocity at a specific location
    pub fn velocity_at(&self, depth: usize, beam: usize) -> Option<VelocityEstimate> {
        if depth < self.velocity.nrows() && beam < self.velocity.ncols() {
            Some(VelocityEstimate::new(
                self.velocity[[depth, beam]],
                self.variance[[depth, beam]],
            ))
        } else {
            None
        }
    }

    /// Calculate maximum velocity in the map
    pub fn max_velocity(&self) -> f64 {
        self.velocity
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Calculate minimum velocity in the map
    pub fn min_velocity(&self) -> f64 {
        self.velocity.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Get velocity range (min, max)
    pub fn velocity_range(&self) -> (f64, f64) {
        (self.min_velocity(), self.max_velocity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_direction_classification() {
        assert_eq!(
            FlowDirection::from_velocity(0.5, 0.1),
            FlowDirection::Toward
        );
        assert_eq!(FlowDirection::from_velocity(-0.5, 0.1), FlowDirection::Away);
        assert_eq!(FlowDirection::from_velocity(0.05, 0.1), FlowDirection::None);
    }

    #[test]
    fn test_velocity_estimate_reliability() {
        let estimate = VelocityEstimate::new(0.5, 0.2);
        assert!(estimate.is_reliable(0.3));
        assert!(!estimate.is_reliable(0.1));
    }
}
