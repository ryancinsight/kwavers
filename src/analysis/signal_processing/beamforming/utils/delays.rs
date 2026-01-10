//! # Delay Calculation Utilities (SSOT)
//!
//! Single Source of Truth for geometric delay and phase calculations used across
//! transmit and receive beamforming.
//!
//! # Architectural Intent
//!
//! This module enforces **strict SSOT** for delay/phase calculations:
//!
//! - **Transmit Beamforming**: Hardware phase delays for transducer control
//! - **Receive Beamforming**: Complex steering vectors for signal processing
//! - **Both domains**: Share identical geometric foundations
//!
//! ## Layer Separation
//!
//! ```text
//! Domain Layer (hardware control)
//!   ↓ uses
//! Analysis Layer (this module) ← SSOT for delay geometry
//!   ↓ uses
//! Math Layer (vector operations, distance)
//! ```
//!
//! ## Design Principles
//!
//! 1. **Pure Geometric Operations**: No domain coupling, no hardware state
//! 2. **Explicit Validation**: All inputs checked, no silent failures
//! 3. **Mathematical Correctness**: Delays derived from first principles
//! 4. **Zero Tolerance**: No placeholders, dummy outputs, or approximations
//!
//! # Mathematical Foundation
//!
//! ## Focus Delays (Transmit/Receive)
//!
//! For a focal point **r** = (x, y, z) and sensor at **sᵢ**, the geometric
//! time delay is:
//!
//! ```text
//! τᵢ = ||r - sᵢ|| / c
//! ```
//!
//! For phase-based control (transmit), we use relative delays:
//!
//! ```text
//! Δτᵢ = τ_max - τᵢ
//! φᵢ = k · Δτᵢ · c = k · (d_max - dᵢ)
//! ```
//!
//! where `k = 2πf/c` is the wavenumber.
//!
//! ## Steering Delays (Plane Wave)
//!
//! For a plane wave in direction **d** (unit vector), the delay from reference
//! (typically center of array) is:
//!
//! ```text
//! τᵢ = (sᵢ · d) / c
//! φᵢ = k · (sᵢ · d)
//! ```
//!
//! ## Invariants
//!
//! - All positions and directions must be finite (no NaN/Inf)
//! - Sound speed must be positive and finite
//! - Frequency must be positive and finite (for phase calculations)
//! - Direction vectors must be normalized (unit vectors)
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::utils::delays;
//!
//! // Calculate focus delays for transmit beamforming
//! let element_positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];
//! let focal_point = [0.0, 0.0, 0.05]; // 5 cm depth
//! let phase_delays = delays::focus_phase_delays(
//!     &element_positions,
//!     focal_point,
//!     1e6,    // 1 MHz
//!     1540.0, // sound speed (m/s)
//! )?;
//!
//! // Calculate plane wave steering delays
//! let direction = [0.0, 0.0, 1.0]; // axial propagation
//! let steering_delays = delays::plane_wave_phase_delays(
//!     &element_positions,
//!     direction,
//!     1e6,
//!     1540.0,
//! )?;
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array1;
use std::f64::consts::PI;

/// Calculate phase delays for focusing at a target point.
///
/// Returns phase delays in radians for each element position to focus at the
/// specified target point. Delays are relative to the maximum distance (farthest
/// element), ensuring all phases are non-negative.
///
/// # Mathematical Definition
///
/// For element i at position **sᵢ** and focal point **r**:
///
/// ```text
/// dᵢ = ||r - sᵢ||           (distance, meters)
/// k = 2πf/c                  (wavenumber, rad/m)
/// φᵢ = k·(d_max - dᵢ)        (phase delay, radians)
/// ```
///
/// # Arguments
///
/// * `element_positions` - Array of 3D positions [x, y, z] in meters
/// * `focal_point` - Target focus position [x, y, z] in meters
/// * `frequency` - Operating frequency in Hz (must be positive)
/// * `sound_speed` - Speed of sound in m/s (must be positive)
///
/// # Returns
///
/// Array of phase delays in radians, one per element. All values are non-negative
/// due to reference normalization (d_max).
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` if:
/// - Element positions array is empty
/// - Any position or focal point contains non-finite values (NaN/Inf)
/// - Frequency is non-positive or non-finite
/// - Sound speed is non-positive or non-finite
///
/// # Examples
///
/// ```rust,ignore
/// // Linear array with 8 elements, 0.3mm spacing
/// let positions: Vec<[f64; 3]> = (0..8)
///     .map(|i| [i as f64 * 0.3e-3, 0.0, 0.0])
///     .collect();
///
/// let focal_point = [0.0, 0.0, 0.05]; // 5 cm axial depth
/// let delays = focus_phase_delays(&positions, focal_point, 1e6, 1540.0)?;
/// ```
pub fn focus_phase_delays(
    element_positions: &[[f64; 3]],
    focal_point: [f64; 3],
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<f64>> {
    // Validate inputs
    if element_positions.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Element positions array is empty".into(),
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

    // Compute wavenumber: k = 2πf/c
    let wavenumber = 2.0 * PI * frequency / sound_speed;

    // Calculate distances from each element to focal point
    let mut distances = Vec::with_capacity(element_positions.len());
    for (i, pos) in element_positions.iter().enumerate() {
        // Validate position
        if !pos.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "Element position {} contains non-finite values",
                i
            )));
        }

        // Euclidean distance: ||r - sᵢ||
        let dx = focal_point[0] - pos[0];
        let dy = focal_point[1] - pos[1];
        let dz = focal_point[2] - pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if !distance.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Computed distance for element {} is non-finite",
                i
            )));
        }

        distances.push(distance);
    }

    // Find maximum distance for reference normalization
    let max_distance = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !max_distance.is_finite() || max_distance <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "Maximum distance is non-positive or non-finite".into(),
        ));
    }

    // Calculate phase delays: φᵢ = k·(d_max - dᵢ)
    let mut phase_delays = Array1::<f64>::zeros(element_positions.len());
    for (i, &distance) in distances.iter().enumerate() {
        phase_delays[i] = wavenumber * (max_distance - distance);

        // Invariant: all phase delays must be non-negative due to normalization
        debug_assert!(
            phase_delays[i] >= 0.0,
            "Phase delay must be non-negative (got {})",
            phase_delays[i]
        );
    }

    Ok(phase_delays)
}

/// Calculate phase delays for plane wave steering.
///
/// Returns phase delays in radians for steering the beam in the specified
/// direction. Unlike focused delays, these can be positive or negative depending
/// on element position relative to the propagation direction.
///
/// # Mathematical Definition
///
/// For element i at position **sᵢ** and propagation direction **d** (unit vector):
///
/// ```text
/// k = 2πf/c                  (wavenumber, rad/m)
/// φᵢ = -k·(sᵢ · d)           (phase delay, radians)
/// ```
///
/// The negative sign ensures phase alignment for a wave arriving from direction **d**.
///
/// # Arguments
///
/// * `element_positions` - Array of 3D positions [x, y, z] in meters
/// * `direction` - Propagation direction [x, y, z] (must be unit vector)
/// * `frequency` - Operating frequency in Hz (must be positive)
/// * `sound_speed` - Speed of sound in m/s (must be positive)
///
/// # Returns
///
/// Array of phase delays in radians, one per element. Values can be positive or
/// negative depending on geometry.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` if:
/// - Element positions array is empty
/// - Any position contains non-finite values
/// - Direction vector is not normalized (||d|| ≠ 1 within 1e-6 tolerance)
/// - Direction contains non-finite values
/// - Frequency is non-positive or non-finite
/// - Sound speed is non-positive or non-finite
///
/// # Examples
///
/// ```rust,ignore
/// // Steer beam 30 degrees from broadside (elevation plane)
/// let theta = 30.0_f64.to_radians();
/// let direction = [theta.sin(), 0.0, theta.cos()];
///
/// let delays = plane_wave_phase_delays(&positions, direction, 1e6, 1540.0)?;
/// ```
pub fn plane_wave_phase_delays(
    element_positions: &[[f64; 3]],
    direction: [f64; 3],
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<f64>> {
    // Validate inputs
    if element_positions.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Element positions array is empty".into(),
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

    // Validate direction is unit vector
    let dir_norm_sq = direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2);

    if !dir_norm_sq.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Direction vector contains non-finite values".into(),
        ));
    }

    if (dir_norm_sq - 1.0).abs() > 1e-6 {
        return Err(KwaversError::InvalidInput(format!(
            "Direction must be unit vector, got norm² = {} (norm = {})",
            dir_norm_sq,
            dir_norm_sq.sqrt()
        )));
    }

    // Compute wavenumber: k = 2πf/c
    let wavenumber = 2.0 * PI * frequency / sound_speed;

    // Calculate phase delays: φᵢ = -k·(sᵢ · d)
    let mut phase_delays = Array1::<f64>::zeros(element_positions.len());

    for (i, pos) in element_positions.iter().enumerate() {
        // Validate position
        if !pos.iter().all(|&x| x.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "Element position {} contains non-finite values",
                i
            )));
        }

        // Dot product: sᵢ · d
        let dot_product = pos[0] * direction[0] + pos[1] * direction[1] + pos[2] * direction[2];

        if !dot_product.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Dot product for element {} is non-finite",
                i
            )));
        }

        // Phase delay with negative sign for arrival alignment
        phase_delays[i] = -wavenumber * dot_product;
    }

    Ok(phase_delays)
}

/// Calculate phase delays for steering to spherical angles (theta, phi).
///
/// Convenience function that converts spherical steering angles to Cartesian
/// direction and calls `plane_wave_phase_delays`.
///
/// # Arguments
///
/// * `element_positions` - Array of 3D positions [x, y, z] in meters
/// * `theta` - Polar angle from z-axis in radians [0, π]
/// * `phi` - Azimuthal angle in xy-plane in radians [0, 2π]
/// * `frequency` - Operating frequency in Hz (must be positive)
/// * `sound_speed` - Speed of sound in m/s (must be positive)
///
/// # Coordinate Convention
///
/// ```text
/// x = sin(θ)·cos(φ)
/// y = sin(θ)·sin(φ)
/// z = cos(θ)
/// ```
///
/// - θ = 0: positive z-axis (axial)
/// - θ = π/2: xy-plane (lateral)
/// - φ = 0: positive x-axis
/// - φ = π/2: positive y-axis
///
/// # Returns
///
/// Array of phase delays in radians.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` if:
/// - Angles contain non-finite values
/// - Other validation errors from `plane_wave_phase_delays`
pub fn spherical_steering_phase_delays(
    element_positions: &[[f64; 3]],
    theta: f64,
    phi: f64,
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<Array1<f64>> {
    // Validate angles
    if !theta.is_finite() || !phi.is_finite() {
        return Err(KwaversError::InvalidInput(
            "Steering angles must be finite".into(),
        ));
    }

    // Convert spherical to Cartesian (unit vector)
    let direction = [
        theta.sin() * phi.cos(), // x
        theta.sin() * phi.sin(), // y
        theta.cos(),             // z
    ];

    plane_wave_phase_delays(element_positions, direction, frequency, sound_speed)
}

/// Calculate beam width for a given aperture size (Rayleigh criterion).
///
/// Estimates the angular resolution of the array based on diffraction-limited
/// beam width.
///
/// # Mathematical Definition
///
/// ```text
/// Δθ ≈ 1.22 · λ / D
/// ```
///
/// where:
/// - `λ = c/f` (wavelength, meters)
/// - `D` = aperture size (meters)
///
/// # Arguments
///
/// * `aperture_size` - Physical size of array aperture in meters
/// * `frequency` - Operating frequency in Hz
/// * `sound_speed` - Speed of sound in m/s
///
/// # Returns
///
/// Beam width in radians.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` if any parameter is non-positive or non-finite.
pub fn calculate_beam_width(
    aperture_size: f64,
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<f64> {
    if !aperture_size.is_finite() || aperture_size <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Aperture size must be positive and finite, got {}",
            aperture_size
        )));
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

    let wavelength = sound_speed / frequency;
    let beam_width = 1.22 * wavelength / aperture_size;

    Ok(beam_width)
}

/// Calculate focal zone depth (depth of field).
///
/// Estimates the axial extent over which the beam remains focused, based on the
/// Rayleigh range for a focused aperture.
///
/// # Mathematical Definition
///
/// ```text
/// DOF ≈ 7 · λ · F²
/// ```
///
/// where:
/// - `λ = c/f` (wavelength, meters)
/// - `F = focal_distance / aperture_size` (f-number, dimensionless)
///
/// # Arguments
///
/// * `aperture_size` - Physical size of array aperture in meters
/// * `focal_distance` - Distance from array to focal point in meters
/// * `frequency` - Operating frequency in Hz
/// * `sound_speed` - Speed of sound in m/s
///
/// # Returns
///
/// Depth of field in meters (axial extent around focal point).
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` if any parameter is non-positive or non-finite.
pub fn calculate_focal_zone(
    aperture_size: f64,
    focal_distance: f64,
    frequency: f64,
    sound_speed: f64,
) -> KwaversResult<f64> {
    if !aperture_size.is_finite() || aperture_size <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Aperture size must be positive and finite, got {}",
            aperture_size
        )));
    }

    if !focal_distance.is_finite() || focal_distance <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Focal distance must be positive and finite, got {}",
            focal_distance
        )));
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

    let wavelength = sound_speed / frequency;
    let f_number = focal_distance / aperture_size;
    let focal_zone = 7.0 * wavelength * f_number.powi(2);

    Ok(focal_zone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_focus_phase_delays_linear_array() {
        // Linear array: 4 elements, 0.5mm spacing, centered at origin
        let positions = vec![
            [-0.75e-3, 0.0, 0.0],
            [-0.25e-3, 0.0, 0.0],
            [0.25e-3, 0.0, 0.0],
            [0.75e-3, 0.0, 0.0],
        ];

        let focal_point = [0.0, 0.0, 0.05]; // 5cm axial depth
        let frequency = 1e6; // 1 MHz
        let sound_speed = 1540.0;

        let delays = focus_phase_delays(&positions, focal_point, frequency, sound_speed).unwrap();

        assert_eq!(delays.len(), 4);

        // All delays should be non-negative (normalized to max distance)
        for &delay in delays.iter() {
            assert!(delay >= 0.0, "Phase delay must be non-negative");
        }

        // Center elements have larger phase delays due to normalization φᵢ = k·(d_max - dᵢ)
        // Inner elements are closer (smaller dᵢ) → larger (d_max - dᵢ) → larger phase delay
        assert!(
            delays[1] > delays[0],
            "Inner elements have larger phase delays (normalized)"
        );
        assert!(
            delays[2] > delays[3],
            "Inner elements have larger phase delays (normalized)"
        );

        // Symmetry: outer elements should have equal delays
        assert_relative_eq!(delays[0], delays[3], epsilon = 1e-10);
        assert_relative_eq!(delays[1], delays[2], epsilon = 1e-10);
    }

    #[test]
    fn test_plane_wave_phase_delays_broadside() {
        // Linear array along x-axis
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];

        // Broadside direction (z-axis)
        let direction = [0.0, 0.0, 1.0];
        let frequency = 1e6;
        let sound_speed = 1540.0;

        let delays =
            plane_wave_phase_delays(&positions, direction, frequency, sound_speed).unwrap();

        assert_eq!(delays.len(), 3);

        // All delays should be zero for broadside (no projection on x-axis)
        for &delay in delays.iter() {
            assert_relative_eq!(delay, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_plane_wave_phase_delays_endfire() {
        // Linear array along x-axis
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];

        // Endfire direction (x-axis)
        let direction = [1.0, 0.0, 0.0];
        let frequency = 1e6;
        let sound_speed = 1540.0;

        let delays =
            plane_wave_phase_delays(&positions, direction, frequency, sound_speed).unwrap();

        assert_eq!(delays.len(), 3);

        // Delays should increase linearly along array
        let wavenumber = 2.0 * PI * frequency / sound_speed;
        let expected_increment = -wavenumber * 0.001; // spacing = 1mm

        assert_relative_eq!(delays[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(delays[1], expected_increment, epsilon = 1e-10);
        assert_relative_eq!(delays[2], 2.0 * expected_increment, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_steering_broadside() {
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];

        // Broadside: theta=0 (z-axis)
        let delays = spherical_steering_phase_delays(&positions, 0.0, 0.0, 1e6, 1540.0).unwrap();

        // Should give same result as broadside plane wave
        for &delay in delays.iter() {
            assert_relative_eq!(delay, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spherical_steering_lateral() {
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];

        // Lateral: theta=π/2, phi=0 (x-axis)
        let delays =
            spherical_steering_phase_delays(&positions, PI / 2.0, 0.0, 1e6, 1540.0).unwrap();

        // Should give same result as endfire
        let wavenumber = 2.0 * PI * 1e6 / 1540.0;
        assert_relative_eq!(delays[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(delays[1], -wavenumber * 0.001, epsilon = 1e-10);
    }

    #[test]
    fn test_beam_width_calculation() {
        let aperture_size = 0.01; // 1 cm
        let frequency = 1e6; // 1 MHz
        let sound_speed = 1540.0;

        let beam_width = calculate_beam_width(aperture_size, frequency, sound_speed).unwrap();

        // Expected: λ = 1540 / 1e6 = 1.54 mm
        // Beam width ≈ 1.22 * 1.54e-3 / 0.01 ≈ 0.188 rad ≈ 10.8 degrees
        let wavelength = sound_speed / frequency;
        let expected = 1.22 * wavelength / aperture_size;

        assert_relative_eq!(beam_width, expected, epsilon = 1e-10);
        assert!(beam_width > 0.0);
    }

    #[test]
    fn test_focal_zone_calculation() {
        let aperture_size = 0.01; // 1 cm
        let focal_distance = 0.05; // 5 cm
        let frequency = 1e6;
        let sound_speed = 1540.0;

        let focal_zone =
            calculate_focal_zone(aperture_size, focal_distance, frequency, sound_speed).unwrap();

        // F-number = 5cm / 1cm = 5
        // λ = 1.54 mm
        // DOF ≈ 7 * 1.54e-3 * 25 ≈ 0.27 m
        let wavelength = sound_speed / frequency;
        let f_number = focal_distance / aperture_size;
        let expected = 7.0 * wavelength * f_number.powi(2);

        assert_relative_eq!(focal_zone, expected, epsilon = 1e-10);
        assert!(focal_zone > 0.0);
    }

    #[test]
    fn test_invalid_empty_positions() {
        let result = focus_phase_delays(&[], [0.0, 0.0, 0.05], 1e6, 1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_non_unit_direction() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let direction = [1.0, 1.0, 0.0]; // Not normalized

        let result = plane_wave_phase_delays(&positions, direction, 1e6, 1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_negative_frequency() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let result = focus_phase_delays(&positions, [0.0, 0.0, 0.05], -1e6, 1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_negative_sound_speed() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let result = focus_phase_delays(&positions, [0.0, 0.0, 0.05], 1e6, -1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_non_finite_focal_point() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let result = focus_phase_delays(&positions, [f64::NAN, 0.0, 0.05], 1e6, 1540.0);
        assert!(result.is_err());
    }
}
