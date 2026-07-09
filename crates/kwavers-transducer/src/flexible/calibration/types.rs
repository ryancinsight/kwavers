//! Data types for flexible array calibration.

use leto::{Array1, Array2};
use leto::{
    Array1 as NdArray1,
    Array2 as NdArray2,
};

/// Calibration data storage
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Time-dependent geometry snapshots
    pub geometry_history: Vec<GeometrySnapshot>,
    /// Calibration quality metrics
    pub quality_metrics: CalibrationQualityMetrics,
    /// Reference configuration
    pub reference_geometry: Option<Array2<f64>>,
}

/// Geometry snapshot at a specific time
#[derive(Debug, Clone)]
pub struct GeometrySnapshot {
    /// Timestamp
    pub timestamp: f64,
    /// Element positions [`n_elements` x 3]
    pub positions: NdArray2<f64>,
    /// Confidence scores per element
    pub confidence: NdArray1<f64>,
}

/// Calibration quality metrics
#[derive(Debug, Clone)]
pub struct CalibrationQualityMetrics {
    /// Position uncertainty (meters)
    pub position_uncertainty: f64,
    /// Orientation uncertainty (radians)
    pub orientation_uncertainty: f64,
    /// Overall calibration confidence [0, 1]
    pub confidence: f64,
}

/// Kalman filter state for position tracking
#[derive(Debug, Clone)]
pub(super) struct KalmanState {
    /// State estimate (positions and velocities)
    pub(super) state: Array1<f64>,
    /// Error covariance matrix
    pub(super) covariance: Array2<f64>,
    /// Process noise covariance
    pub(super) process_noise: Array2<f64>,
    /// Measurement noise covariance
    pub(super) measurement_noise: Array2<f64>,
}
