//! Sensor-array initialization seam — keeps geometry concerns out of the
//! algorithm traits.

use kwavers_core::error::KwaversResult;

/// Configuration trait for beamformers that can be constructed from domain geometry.
///
/// This trait separates the concern of "how to initialize a beamformer from sensor
/// array geometry" from the core beamforming algorithms. It enables dependency
/// injection and testability.
///
/// # Architectural Rationale
///
/// - **Separation**: Domain (geometry) vs. Analysis (algorithms)
/// - **Testability**: Allows mock sensor arrays for unit tests
/// - **Flexibility**: Multiple initialization strategies without changing algorithm traits
pub trait BeamformerConfig<SensorArray>: Sized {
    /// Create beamformer from sensor array geometry.
    ///
    /// # Parameters
    ///
    /// - `array`: Sensor array with geometry information
    /// - `sound_speed`: Medium sound speed in m/s
    /// - `sampling_rate`: Data sampling rate in Hz
    ///
    /// # Returns
    ///
    /// Configured beamformer instance.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Array geometry is invalid or empty
    /// - Sound speed or sampling rate non-positive or non-finite
    /// - Insufficient sensor elements for algorithm requirements
    fn from_sensor_array(
        array: &SensorArray,
        sound_speed: f64,
        sampling_rate: f64,
    ) -> KwaversResult<Self>;
}
