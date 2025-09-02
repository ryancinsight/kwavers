//! Sensor recording functionality matching k-Wave
//!
//! Implements various sensor types and recording options

use ndarray::Array3;

/// Sensor configuration
#[derive(Debug, Clone)]
pub struct SensorMask {
    /// Binary mask indicating sensor positions
    pub mask: Array3<bool>,
    /// Record pressure time series
    pub record_pressure: bool,
    /// Record velocity components
    pub record_velocity: bool,
    /// Record maximum pressure
    pub record_p_max: bool,
    /// Record RMS pressure
    pub record_p_rms: bool,
}
