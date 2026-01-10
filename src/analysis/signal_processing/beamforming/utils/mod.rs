//! # Beamforming Utilities
//!
//! This module provides utility functions and helper types for beamforming algorithms.
//! It consolidates shared operations that are used across multiple beamforming implementations.
//!
//! # Architectural Intent (SSOT + Accessor Layer)
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth**: Shared operations defined once, used everywhere
//! 2. **Accessor Pattern**: Provides well-defined interfaces for common computations
//! 3. **Layer Separation**: Pure mathematical operations, no domain coupling
//! 4. **Explicit Failure**: No silent fallbacks or error masking
//!
//! ## SSOT Enforcement (Strict)
//!
//! This module is the **only** place for shared beamforming utilities:
//!
//! - ❌ **NO duplicate steering vector computation** in algorithms
//! - ❌ **NO local implementations** of distance calculations
//! - ❌ **NO ad-hoc interpolation** scattered across codebase
//! - ❌ **NO silent fallbacks** or dummy outputs
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::utils (Layer 7)
//!   ↓ imports from
//! math::geometry (Layer 1) - distance, vector operations
//! core::error (Layer 0) - error types
//! ```
//!
//! # Module Organization
//!
//! ## Delay Calculations (`delays` module)
//!
//! - **Phase delays** for transmit beamforming (hardware control)
//! - **Time delays** and **steering vectors** for receive beamforming
//! - Focus delays, plane wave steering, spherical coordinates
//! - Beam width and focal zone estimation
//!
//! ## Steering Vectors (this module)
//!
//! - Complex steering vectors for receive signal processing
//! - Plane wave and focused variants
//! - Narrowband formulations
//!
//! ## Interpolation
//!
//! - Temporal interpolation for fractional delays
//! - Spatial interpolation for arbitrary focal points
//!
//! ## Sparse Operations
//!
//! - Sparse matrix utilities for large-scale beamforming
//! - Memory-efficient delay table storage
//!
//! ## Window Functions
//!
//! - Apodization windows (Hamming, Hanning, Blackman, etc.)
//! - Spatial tapering for side lobe reduction
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::utils;
//! use num_complex::Complex64;
//!
//! // Compute steering vector for plane wave
//! let sensor_positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], ...];
//! let steering = utils::plane_wave_steering_vector(
//!     &sensor_positions,
//!     [1.0, 0.0, 0.0],  // look direction (unit vector)
//!     1e6,               // 1 MHz
//!     1540.0,            // sound speed (m/s)
//! )?;
//!
//! // Apply Hamming window
//! let window = utils::hamming_window(8);
//! let apodized_steering = steering
//!     .iter()
//!     .zip(window.iter())
//!     .map(|(s, w)| s * w)
//!     .collect();
//! ```
//!
//! # Mathematical Foundation
//!
//! ## Plane Wave Steering Vector
//!
//! For a plane wave arriving from direction **d** (unit vector), the steering
//! vector for sensor i at position **sᵢ** is:
//!
//! ```text
//! aᵢ = exp(j·k·(sᵢ · d))
//! ```
//!
//! where:
//! - `k = 2πf/c` (wavenumber, rad/m)
//! - `f` = frequency (Hz)
//! - `c` = sound speed (m/s)
//! - `j` = √(-1)
//!
//! ## Focused Steering Vector
//!
//! For a focused array (transmit or receive) with focal point **r**, the
//! steering vector accounts for distance from each sensor:
//!
//! ```text
//! aᵢ = exp(j·k·||r - sᵢ||)
//! ```
//!
//! ## Temporal Interpolation
//!
//! For fractional delay τ between samples n and n+1, use linear interpolation:
//!
//! ```text
//! x_interp(τ) = (1-α)·x[n] + α·x[n+1]
//! ```
//!
//! where α = τ - floor(τ) ∈ [0, 1).
//!
//! # Performance Considerations
//!
//! | Operation | Complexity | Parallelizable | GPU-Friendly |
//! |-----------|------------|----------------|--------------|
//! | Steering Vector | O(N) | ✅ Yes | ✅ Yes |
//! | Window Function | O(N) | ✅ Yes | ✅ Yes |
//! | Linear Interpolation | O(N·M) | ✅ Yes | ✅ Yes |
//! | Sparse Operations | O(K) K<<N² | ⚠️ Partial | ⚠️ Moderate |
//!
//! # Literature References
//!
//! ## Array Signal Processing
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience.
//!   ISBN: 978-0-471-09390-9
//!
//! ## Interpolation and Windowing
//!
//! - Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*
//!   (3rd ed.). Prentice Hall.
//!   ISBN: 978-0-13-198842-2
//!
//! - Harris, F. J. (1978). "On the use of windows for harmonic analysis with the
//!   discrete Fourier transform." *Proceedings of the IEEE*, 66(1), 51-83.
//!   DOI: 10.1109/PROC.1978.10837

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

// Submodules (SSOT for beamforming utilities)
pub mod delays; // Delay/phase calculations (transmit + receive SSOT)
pub mod sparse; // Sparse matrix utilities for large-scale beamforming (SSOT)

// Future submodules (migration targets)
// pub mod interpolation; // Temporal/spatial interpolation
// pub mod windows;      // Apodization window functions

/// Compute plane wave steering vector for sensor array.
///
/// The steering vector represents the expected phase response of the array
/// for a plane wave arriving from a specific direction.
///
/// # Parameters
///
/// - `sensor_positions`: Array element positions as [[x, y, z], ...] in meters
/// - `look_direction`: Unit vector [dx, dy, dz] pointing toward source
/// - `frequency`: Signal frequency in Hz
/// - `sound_speed`: Medium sound speed in m/s
///
/// # Returns
///
/// Steering vector **a** (N×1 complex) where N = number of sensors.
///
/// # Mathematical Definition
///
/// For sensor i at position **sᵢ**:
/// ```text
/// aᵢ = exp(j·k·(sᵢ · d))
/// ```
/// where k = 2πf/c is the wavenumber and d is the look direction.
///
/// # Errors
///
/// Returns `Err(...)` if:
/// - `sensor_positions` is empty
/// - `look_direction` is not unit length (within tolerance)
/// - `frequency` ≤ 0 or non-finite
/// - `sound_speed` ≤ 0 or non-finite
/// - Any position coordinate is non-finite
///
/// # Performance
///
/// - **Time Complexity**: O(N) where N = number of sensors
/// - **Space Complexity**: O(N)
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::utils;
///
/// let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];
/// let direction = [1.0, 0.0, 0.0]; // Broadside
/// let steering = utils::plane_wave_steering_vector(
///     &positions,
///     direction,
///     1e6,   // 1 MHz
///     1540.0 // soft tissue
/// )?;
/// assert_eq!(steering.len(), 3);
/// ```
pub fn plane_wave_steering_vector(
    sensor_positions: &[[f64; 3]],
    look_direction: [f64; 3],
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<Complex64>> {
    // Validate inputs
    if sensor_positions.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Sensor positions array is empty".into(),
        ));
    }

    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Frequency must be positive and finite, got {}",
            frequency
        )));
    }

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Sound speed must be positive and finite, got {}",
            sound_speed
        )));
    }

    // Validate look direction is unit vector
    let dir_norm_sq =
        look_direction[0].powi(2) + look_direction[1].powi(2) + look_direction[2].powi(2);

    if !dir_norm_sq.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Look direction contains non-finite values".into(),
        ));
    }

    if (dir_norm_sq - 1.0).abs() > 1e-6 {
        return Err(KwaversError::InvalidInput(format!(
            "Look direction must be unit vector, got norm² = {}",
            dir_norm_sq
        )));
    }

    // Compute wavenumber: k = 2πf/c
    let wavenumber = 2.0 * PI * frequency / sound_speed;

    // Compute steering vector
    let n = sensor_positions.len();
    let mut steering = Array1::<Complex64>::zeros(n);

    for (i, pos) in sensor_positions.iter().enumerate() {
        // Validate position
        if !pos.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "Sensor position {} contains non-finite values",
                i
            )));
        }

        // Compute dot product: sᵢ · d
        let dot_product =
            pos[0] * look_direction[0] + pos[1] * look_direction[1] + pos[2] * look_direction[2];

        // Phase: φᵢ = k·(sᵢ · d)
        let phase = wavenumber * dot_product;

        // Steering vector element: aᵢ = exp(j·φᵢ)
        steering[i] = Complex64::from_polar(1.0, phase);
    }

    Ok(steering)
}

/// Compute focused steering vector for sensor array.
///
/// The focused steering vector accounts for the distance from each sensor to
/// a specific focal point in space (spherical wave model).
///
/// # Parameters
///
/// - `sensor_positions`: Array element positions as [[x, y, z], ...] in meters
/// - `focal_point`: Focal point [x, y, z] in meters
/// - `frequency`: Signal frequency in Hz
/// - `sound_speed`: Medium sound speed in m/s
///
/// # Returns
///
/// Focused steering vector **a** (N×1 complex).
///
/// # Mathematical Definition
///
/// For sensor i at position **sᵢ**:
/// ```text
/// aᵢ = exp(j·k·||r - sᵢ||)
/// ```
/// where k = 2πf/c and r is the focal point.
///
/// # Errors
///
/// Returns `Err(...)` if:
/// - `sensor_positions` is empty
/// - `focal_point` contains non-finite values
/// - `frequency` ≤ 0 or non-finite
/// - `sound_speed` ≤ 0 or non-finite
/// - Any position coordinate is non-finite
///
/// # Performance
///
/// - **Time Complexity**: O(N)
/// - **Space Complexity**: O(N)
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::utils;
///
/// let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];
/// let focal = [0.0, 0.0, 0.02]; // 2 cm depth
/// let steering = utils::focused_steering_vector(
///     &positions,
///     focal,
///     1e6,
///     1540.0
/// )?;
/// ```
pub fn focused_steering_vector(
    sensor_positions: &[[f64; 3]],
    focal_point: [f64; 3],
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<Complex64>> {
    // Validate inputs
    if sensor_positions.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Sensor positions array is empty".into(),
        ));
    }

    if !focal_point.iter().all(|&x| x.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "Focal point contains non-finite values".into(),
        ));
    }

    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Frequency must be positive and finite, got {}",
            frequency
        )));
    }

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Sound speed must be positive and finite, got {}",
            sound_speed
        )));
    }

    // Compute wavenumber
    let wavenumber = 2.0 * PI * frequency / sound_speed;

    // Compute steering vector
    let n = sensor_positions.len();
    let mut steering = Array1::<Complex64>::zeros(n);

    for (i, pos) in sensor_positions.iter().enumerate() {
        // Validate position
        if !pos.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "Sensor position {} contains non-finite values",
                i
            )));
        }

        // Compute distance: ||r - sᵢ||
        let dx = focal_point[0] - pos[0];
        let dy = focal_point[1] - pos[1];
        let dz = focal_point[2] - pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Phase: φᵢ = k·distance
        let phase = wavenumber * distance;

        // Steering vector element: aᵢ = exp(j·φᵢ)
        steering[i] = Complex64::from_polar(1.0, phase);
    }

    Ok(steering)
}

/// Generate Hamming window for apodization.
///
/// The Hamming window is a raised cosine window used to reduce side lobes in
/// beamforming and spectral analysis.
///
/// # Parameters
///
/// - `length`: Window length (number of samples)
///
/// # Returns
///
/// Window coefficients w[n] for n ∈ [0, length-1].
///
/// # Mathematical Definition
///
/// ```text
/// w[n] = 0.54 - 0.46·cos(2πn/(N-1))
/// ```
/// for n ∈ [0, N-1].
///
/// # Errors
///
/// Returns `Err(...)` if `length` is 0.
///
/// # Performance
///
/// - **Time Complexity**: O(N)
/// - **Space Complexity**: O(N)
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::utils;
///
/// let window = utils::hamming_window(8)?;
/// assert_eq!(window.len(), 8);
/// assert!(window[0] < 1.0); // Tapered edges
/// assert!(window[4] > 0.95); // Peak near center
/// ```
pub fn hamming_window(length: usize) -> KwaversResult<Array1<f64>> {
    if length == 0 {
        return Err(KwaversError::InvalidInput(
            "Window length must be positive".into(),
        ));
    }

    if length == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }

    let mut window = Array1::<f64>::zeros(length);
    let n_minus_1 = (length - 1) as f64;

    for n in 0..length {
        window[n] = 0.54 - 0.46 * (2.0 * PI * (n as f64) / n_minus_1).cos();
    }

    Ok(window)
}

/// Generate Hanning (Hann) window for apodization.
///
/// The Hanning window is a smooth window function used for side lobe reduction.
///
/// # Parameters
///
/// - `length`: Window length (number of samples)
///
/// # Returns
///
/// Window coefficients w[n] for n ∈ [0, length-1].
///
/// # Mathematical Definition
///
/// ```text
/// w[n] = 0.5·(1 - cos(2πn/(N-1)))
/// ```
/// for n ∈ [0, N-1].
///
/// # Errors
///
/// Returns `Err(...)` if `length` is 0.
///
/// # Performance
///
/// - **Time Complexity**: O(N)
/// - **Space Complexity**: O(N)
pub fn hanning_window(length: usize) -> KwaversResult<Array1<f64>> {
    if length == 0 {
        return Err(KwaversError::InvalidInput(
            "Window length must be positive".into(),
        ));
    }

    if length == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }

    let mut window = Array1::<f64>::zeros(length);
    let n_minus_1 = (length - 1) as f64;

    for n in 0..length {
        window[n] = 0.5 * (1.0 - (2.0 * PI * (n as f64) / n_minus_1).cos());
    }

    Ok(window)
}

/// Generate Blackman window for apodization.
///
/// The Blackman window provides excellent side lobe suppression at the cost
/// of main lobe width.
///
/// # Parameters
///
/// - `length`: Window length (number of samples)
///
/// # Returns
///
/// Window coefficients w[n] for n ∈ [0, length-1].
///
/// # Mathematical Definition
///
/// ```text
/// w[n] = 0.42 - 0.5·cos(2πn/(N-1)) + 0.08·cos(4πn/(N-1))
/// ```
/// for n ∈ [0, N-1].
///
/// # Errors
///
/// Returns `Err(...)` if `length` is 0.
///
/// # Performance
///
/// - **Time Complexity**: O(N)
/// - **Space Complexity**: O(N)
pub fn blackman_window(length: usize) -> KwaversResult<Array1<f64>> {
    if length == 0 {
        return Err(KwaversError::InvalidInput(
            "Window length must be positive".into(),
        ));
    }

    if length == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }

    let mut window = Array1::<f64>::zeros(length);
    let n_minus_1 = (length - 1) as f64;

    for n in 0..length {
        let x = 2.0 * PI * (n as f64) / n_minus_1;
        window[n] = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
    }

    Ok(window)
}

/// Linear interpolation between two complex samples.
///
/// Computes weighted average: `(1-α)·x0 + α·x1` where α ∈ [0, 1].
///
/// # Parameters
///
/// - `x0`: Sample at integer index n
/// - `x1`: Sample at integer index n+1
/// - `alpha`: Fractional position α ∈ [0, 1]
///
/// # Returns
///
/// Interpolated value.
///
/// # Errors
///
/// Returns `Err(...)` if:
/// - `alpha` is not in [0, 1]
/// - `alpha` is non-finite
/// - Either sample is non-finite
///
/// # Performance
///
/// - **Time Complexity**: O(1)
/// - **Space Complexity**: O(1)
///
/// # Example
///
/// ```rust,ignore
/// use num_complex::Complex64;
/// use kwavers::analysis::signal_processing::beamforming::utils;
///
/// let x0 = Complex64::new(1.0, 0.0);
/// let x1 = Complex64::new(2.0, 1.0);
/// let interp = utils::linear_interpolate(x0, x1, 0.3)?;
/// // interp ≈ (0.7)·x0 + (0.3)·x1 = (1.3, 0.3)
/// ```
pub fn linear_interpolate(x0: Complex64, x1: Complex64, alpha: f64) -> KwaversResult<Complex64> {
    if !alpha.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Interpolation alpha is non-finite".into(),
        ));
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(KwaversError::InvalidInput(format!(
            "Interpolation alpha must be in [0, 1], got {}",
            alpha
        )));
    }

    if !x0.is_finite() || !x1.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Interpolation samples contain non-finite values".into(),
        ));
    }

    Ok(x0 * (1.0 - alpha) + x1 * alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_plane_wave_steering_vector_broadside() {
        // Linear array along x-axis
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];
        let direction = [0.0, 0.0, 1.0]; // Broadside (perpendicular)
        let frequency = 1e6; // 1 MHz
        let sound_speed = 1540.0;

        let steering = plane_wave_steering_vector(&positions, direction, frequency, sound_speed)
            .expect("should compute");

        assert_eq!(steering.len(), 3);

        // For broadside, all phases should be identical (dot product = 0)
        for i in 0..3 {
            assert_relative_eq!(steering[i].re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(steering[i].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_plane_wave_steering_vector_endfire() {
        // Linear array along x-axis
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];
        let direction = [1.0, 0.0, 0.0]; // Endfire (along array)
        let frequency = 1e6;
        let sound_speed = 1540.0;

        let steering = plane_wave_steering_vector(&positions, direction, frequency, sound_speed)
            .expect("should compute");

        assert_eq!(steering.len(), 2);

        // Elements should have different phases
        assert!(steering[0] != steering[1]);
    }

    #[test]
    fn test_focused_steering_vector() {
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];
        let focal = [0.0005, 0.0, 0.02]; // 2 cm depth, centered
        let frequency = 1e6;
        let sound_speed = 1540.0;

        let steering = focused_steering_vector(&positions, focal, frequency, sound_speed)
            .expect("should compute");

        assert_eq!(steering.len(), 2);

        // All elements should be non-zero
        for s in steering.iter() {
            assert!(s.norm() > 0.99 && s.norm() < 1.01);
        }
    }

    #[test]
    fn test_hamming_window() {
        let window = hamming_window(8).expect("should compute");

        assert_eq!(window.len(), 8);

        // Check symmetry
        for i in 0..4 {
            assert_relative_eq!(window[i], window[7 - i], epsilon = 1e-10);
        }

        // Check edge tapering (edges < 1.0)
        assert!(window[0] < 1.0);
        assert!(window[7] < 1.0);

        // Check peak near center
        assert!(window[3] > 0.9);
        assert!(window[4] > 0.9);
    }

    #[test]
    fn test_hanning_window() {
        let window = hanning_window(8).expect("should compute");

        assert_eq!(window.len(), 8);

        // Hanning window starts and ends at 0
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[7], 0.0, epsilon = 1e-10);

        // Peak at center
        assert!(window[3] > 0.9);
    }

    #[test]
    fn test_blackman_window() {
        let window = blackman_window(16).expect("should compute");

        assert_eq!(window.len(), 16);

        // Check symmetry
        for i in 0..8 {
            assert_relative_eq!(window[i], window[15 - i], epsilon = 1e-10);
        }

        // Blackman has very small values at edges
        assert!(window[0] < 0.1);
        assert!(window[15] < 0.1);
    }

    #[test]
    fn test_linear_interpolate() {
        let x0 = Complex64::new(1.0, 0.0);
        let x1 = Complex64::new(2.0, 1.0);

        // At alpha = 0, should get x0
        let interp = linear_interpolate(x0, x1, 0.0).expect("should compute");
        assert_relative_eq!(interp.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(interp.im, 0.0, epsilon = 1e-10);

        // At alpha = 1, should get x1
        let interp = linear_interpolate(x0, x1, 1.0).expect("should compute");
        assert_relative_eq!(interp.re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(interp.im, 1.0, epsilon = 1e-10);

        // At alpha = 0.5, should get midpoint
        let interp = linear_interpolate(x0, x1, 0.5).expect("should compute");
        assert_relative_eq!(interp.re, 1.5, epsilon = 1e-10);
        assert_relative_eq!(interp.im, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_inputs() {
        // Empty positions
        let result = plane_wave_steering_vector(&[], [1.0, 0.0, 0.0], 1e6, 1540.0);
        assert!(result.is_err());

        // Non-unit direction
        let positions = vec![[0.0, 0.0, 0.0]];
        let result = plane_wave_steering_vector(&positions, [2.0, 0.0, 0.0], 1e6, 1540.0);
        assert!(result.is_err());

        // Invalid frequency
        let result = plane_wave_steering_vector(&positions, [1.0, 0.0, 0.0], 0.0, 1540.0);
        assert!(result.is_err());

        // Invalid window length
        let result = hamming_window(0);
        assert!(result.is_err());

        // Invalid interpolation alpha
        let x0 = Complex64::new(1.0, 0.0);
        let x1 = Complex64::new(2.0, 0.0);
        assert!(linear_interpolate(x0, x1, -0.1).is_err());
        assert!(linear_interpolate(x0, x1, 1.5).is_err());
    }
}
