//! Time-domain beamforming trait — RF time-series with delay computation
//! and apodization weighting.

use kwavers_core::error::KwaversResult;

use super::Beamformer;

/// Time-domain beamforming algorithms.
///
/// Time-domain beamformers operate directly on RF time series data. They apply
/// geometric delays to align signals from different sensors and then combine
/// them (typically via weighted sum).
///
/// # Mathematical Formulation
///
/// ```text
/// y(r, t) = ∑ᵢ₌₁ᴺ wᵢ · xᵢ(t - τᵢ(r))
/// ```
///
/// where:
/// - `wᵢ` = apodization weight for sensor i
/// - `xᵢ(t)` = RF signal at sensor i
/// - `τᵢ(r)` = geometric time delay from focal point r to sensor i
///
/// # Delay Calculation
///
/// For a focal point **r** and sensor position **sᵢ**, the delay is:
///
/// ```text
/// τᵢ(r) = ||r - sᵢ|| / c
/// ```
///
/// where c is the medium sound speed (m/s).
///
/// # Characteristics
///
/// - **Domain**: Operates on time series (N_sensors × N_samples)
/// - **Complexity**: O(N·M·K) for K focal points
/// - **Parallelism**: Highly parallelizable (independent focal points)
/// - **GPU**: Excellent GPU acceleration potential
///
/// # Implementations
///
/// - `DelayAndSum`: Standard DAS with apodization
/// - `SyntheticAperture`: Virtual source reconstruction
pub trait TimeDomainBeamformer: Beamformer<Input = f64, Output = f64> {
    /// Get the sampling rate in Hz.
    ///
    /// This is required to convert geometric delays (seconds) to sample indices.
    ///
    /// # Invariants
    ///
    /// - Must be positive and finite
    /// - Should match the sampling rate of input RF data
    fn sampling_rate(&self) -> f64;

    /// Get the medium sound speed in m/s.
    ///
    /// Used for time-of-flight delay calculations.
    ///
    /// # Invariants
    ///
    /// - Must be positive and finite
    /// - Typically in range [1000, 2000] m/s for ultrasound
    fn sound_speed(&self) -> f64;

    /// Compute geometric delay from focal point to sensor.
    ///
    /// # Parameters
    ///
    /// - `focal_point`: 3D Cartesian coordinates [x, y, z] in meters
    /// - `sensor_position`: 3D sensor position [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Time delay in seconds (τ = distance / sound_speed).
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if inputs contain non-finite values.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// τ = ||r_focal - r_sensor|| / c
    /// ```
    fn compute_delay(&self, focal_point: [f64; 3], sensor_position: [f64; 3])
        -> KwaversResult<f64>;

    /// Apply apodization (windowing) weights to sensor array.
    ///
    /// Apodization reduces side lobes at the cost of main lobe width.
    /// Common windows: Hamming, Hanning, Blackman, rectangular (uniform).
    ///
    /// # Returns
    ///
    /// Apodization weight in [0, 1].
    ///
    /// # Default Implementation
    ///
    /// Returns 1.0 (uniform weighting, no apodization).
    ///
    /// # Mathematical Definition
    ///
    /// For Hamming window:
    /// ```text
    /// w`N` = 0.54 - 0.46·cos(2πn/(N-1))
    /// ```
    fn apodization_weight(&self, _sensor_index: usize) -> f64 {
        1.0 // Default: uniform (rectangular) window
    }
}
