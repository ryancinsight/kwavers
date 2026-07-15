//! Sampled velocity statistics at discrete sensor positions.

use leto::Array1;

/// Velocity statistics sampled at specific sensor positions.
///
/// Each field is a 1-D array of length `n_sensors`.  Units are m/s throughout.
#[derive(Debug, Clone)]
pub struct SampledVelocityStats {
    /// Maximum ux at each sensor  [m/s]
    pub ux_max: Array1<f64>,
    /// Minimum ux at each sensor  [m/s]
    pub ux_min: Array1<f64>,
    /// RMS ux at each sensor  [m/s]
    pub ux_rms: Array1<f64>,
    /// Maximum uy at each sensor  [m/s]
    pub uy_max: Array1<f64>,
    /// Minimum uy at each sensor  [m/s]
    pub uy_min: Array1<f64>,
    /// RMS uy at each sensor  [m/s]
    pub uy_rms: Array1<f64>,
    /// Maximum uz at each sensor  [m/s]
    pub uz_max: Array1<f64>,
    /// Minimum uz at each sensor  [m/s]
    pub uz_min: Array1<f64>,
    /// RMS uz at each sensor  [m/s]
    pub uz_rms: Array1<f64>,
}

impl SampledVelocityStats {
    /// Number of sensors.
    #[must_use]
    pub fn num_sensors(&self) -> usize {
        self.ux_max.len()
    }
}
