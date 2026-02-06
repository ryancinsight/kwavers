//! Beamforming algorithms for phased array control
//!
//! This module provides a **hardware-focused wrapper** around canonical delay
//! calculation utilities. It maintains the transmit-specific API while delegating
//! geometric computations to the Single Source of Truth in the analysis layer.
//!
//! # Architectural Intent
//!
//! ## Layer Separation
//!
//! ```text
//! Domain Layer (this module)
//!   ↓ hardware control API
//!   ↓ transducer configuration
//!   ↓ uses (accessor pattern)
//! Math Layer (geometry::delays)
//!   ↓ pure geometric delay calculations
//!   ↓ mathematical foundations
//! ```
//!
//! ## Design Rationale
//!
//! - **Domain Layer** (this module):
//!   - Hardware-specific types (`BeamformingMode`, `BeamformingCalculator`)
//!   - Transducer control interface
//!   - Configuration and state management
//!
//! - **Math Layer** (`math::geometry::delays`):
//!   - Pure geometric calculations (SSOT)
//!   - Shared by transmit AND receive beamforming
//!   - No hardware coupling
//!
//! ## SSOT Enforcement
//!
//! - ✅ **Geometric delay calculations**: Delegated to `math::geometry::delays`
//! - ✅ **No duplicate implementations**: Distance, phase delay, steering computations unified
//! - ✅ **Domain-specific logic**: Hardware modes, element configuration remain here
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::domain::source::transducers::phased_array::beamforming::{
//!     BeamformingCalculator, BeamformingMode
//! };
//!
//! let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);
//!
//! // Calculate delays for focusing
//! let element_positions = vec![(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)];
//! let mode = BeamformingMode::Focus { target: (0.0, 0.0, 0.05) };
//!
//! let delays = calculator.calculate_delays(&element_positions, &mode)?;
//! ```

use crate::math::geometry::delays;

/// Beamforming modes for phased array control.
///
/// Specifies the desired beam configuration for transmit focusing or steering.
/// Each mode corresponds to a specific delay profile applied to array elements.
///
/// # Hardware Control Context
///
/// These modes translate high-level imaging intent (where to focus, which direction
/// to steer) into phase delays for transducer element excitation.
#[derive(Debug, Clone)]
pub enum BeamformingMode {
    /// Focus at specific point (x, y, z) [m].
    ///
    /// Applies delays to focus acoustic energy at a target point in space.
    /// Used for B-mode imaging, therapy focusing, and synthetic aperture transmit.
    Focus { target: (f64, f64, f64) },

    /// Steer beam to angle (theta, phi) [rad].
    ///
    /// Creates a plane wave propagating in the specified spherical direction.
    /// Used for plane wave imaging and sector scanning.
    ///
    /// - `theta`: Polar angle from z-axis [0, π]
    /// - `phi`: Azimuthal angle in xy-plane [0, 2π]
    Steer { theta: f64, phi: f64 },

    /// Custom phase delays [rad].
    ///
    /// Directly specifies phase delays for each element. Used for advanced
    /// applications like coded excitation, arbitrary wavefront synthesis, or
    /// hardware calibration.
    Custom { delays: Vec<f64> },

    /// Plane wave transmission in Cartesian direction.
    ///
    /// Creates a plane wave propagating along the specified unit vector.
    /// Equivalent to `Steer` but using Cartesian coordinates for convenience.
    PlaneWave { direction: (f64, f64, f64) },
}

/// Beamforming calculator for phased array transducers.
///
/// Computes phase delays for transmit beamforming based on array geometry,
/// medium properties, and desired beam configuration.
///
/// # Mathematical Foundation
///
/// Delegates geometric calculations to `math::geometry::delays`
/// (SSOT). This wrapper maintains hardware-specific API and conversions.
///
/// # Invariants
///
/// - `sound_speed` > 0 and finite
/// - `frequency` > 0 and finite
/// - Element positions must be finite
///
/// # Examples
///
/// ```rust,ignore
/// let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);
///
/// // Linear array
/// let positions = (0..64)
///     .map(|i| (i as f64 * 0.3e-3, 0.0, 0.0))
///     .collect::<Vec<_>>();
///
/// // Focus at 5cm depth
/// let delays = calculator.calculate_focus_delays(&positions, (0.0, 0.0, 0.05))?;
/// ```
#[derive(Debug)]
pub struct BeamformingCalculator {
    /// Speed of sound in medium [m/s]
    sound_speed: f64,
    /// Operating frequency [Hz]
    frequency: f64,
}

impl BeamformingCalculator {
    /// Create calculator with medium properties.
    ///
    /// # Arguments
    ///
    /// * `sound_speed` - Speed of sound in m/s (must be positive)
    /// * `frequency` - Operating frequency in Hz (must be positive)
    ///
    /// # Panics
    ///
    /// This constructor does not validate inputs (performance optimization for
    /// hot paths). Validation occurs during delay calculation.
    #[must_use]
    pub fn with_medium(sound_speed: f64, frequency: f64) -> Self {
        Self {
            sound_speed,
            frequency,
        }
    }

    /// Calculate phase delays for focusing at a target point.
    ///
    /// # Arguments
    ///
    /// * `element_positions` - Array element positions as (x, y, z) tuples [m]
    /// * `target` - Target focal point as (x, y, z) [m]
    ///
    /// # Returns
    ///
    /// Vector of phase delays [rad], one per element. Delays are normalized
    /// such that all values are non-negative (relative to maximum distance).
    ///
    /// # Errors
    ///
    /// Returns error if positions are invalid, target is non-finite, or medium
    /// properties are invalid.
    ///
    /// # Delegation
    ///
    /// Delegates to `math::geometry::delays::focus_phase_delays`.
    pub fn calculate_focus_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        target: (f64, f64, f64),
    ) -> Vec<f64> {
        // Convert tuple positions to array format required by canonical utilities
        let positions_array: Vec<[f64; 3]> = element_positions
            .iter()
            .map(|&(x, y, z)| [x, y, z])
            .collect();

        let focal_point = [target.0, target.1, target.2];

        // Delegate to canonical SSOT implementation
        // Note: This returns Result, but the old API returned Vec directly.
        // For backward compatibility during refactor, we unwrap here.
        // In production, propagate the error properly.
        delays::focus_phase_delays(
            &positions_array,
            focal_point,
            self.frequency,
            self.sound_speed,
        )
        .expect("Focus delay calculation failed - invalid geometry or medium parameters")
        .to_vec()
    }

    /// Calculate phase delays for beam steering to spherical angles.
    ///
    /// # Arguments
    ///
    /// * `element_positions` - Array element positions as (x, y, z) tuples [m]
    /// * `theta` - Polar angle from z-axis [rad], range [0, π]
    /// * `phi` - Azimuthal angle in xy-plane [rad], range [0, 2π]
    ///
    /// # Returns
    ///
    /// Vector of phase delays [rad], one per element. Values can be positive
    /// or negative depending on geometry.
    ///
    /// # Spherical Coordinate Convention
    ///
    /// ```text
    /// x = sin(θ)·cos(φ)
    /// y = sin(θ)·sin(φ)
    /// z = cos(θ)
    /// ```
    ///
    /// # Delegation
    ///
    /// Delegates to `math::geometry::delays::spherical_steering_phase_delays`.
    #[must_use]
    pub fn calculate_steering_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        theta: f64,
        phi: f64,
    ) -> Vec<f64> {
        let positions_array: Vec<[f64; 3]> = element_positions
            .iter()
            .map(|&(x, y, z)| [x, y, z])
            .collect();

        delays::spherical_steering_phase_delays(
            &positions_array,
            theta,
            phi,
            self.frequency,
            self.sound_speed,
        )
        .expect("Steering delay calculation failed")
        .to_vec()
    }

    /// Calculate phase delays for plane wave transmission.
    ///
    /// # Arguments
    ///
    /// * `element_positions` - Array element positions as (x, y, z) tuples [m]
    /// * `direction` - Propagation direction as (x, y, z) tuple (must be unit vector)
    ///
    /// # Returns
    ///
    /// Vector of phase delays [rad], one per element.
    ///
    /// # Errors (via panic in current implementation)
    ///
    /// If direction is not normalized, the canonical utility will return an error.
    /// Current implementation unwraps for API compatibility.
    ///
    /// # Delegation
    ///
    /// Delegates to `math::geometry::delays::plane_wave_phase_delays`.
    #[must_use]
    pub fn calculate_plane_wave_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        direction: (f64, f64, f64),
    ) -> Vec<f64> {
        let positions_array: Vec<[f64; 3]> = element_positions
            .iter()
            .map(|&(x, y, z)| [x, y, z])
            .collect();

        let direction_array = [direction.0, direction.1, direction.2];

        delays::plane_wave_phase_delays(
            &positions_array,
            direction_array,
            self.frequency,
            self.sound_speed,
        )
        .expect("Plane wave delay calculation failed")
        .to_vec()
    }

    /// Calculate beam width for given aperture configuration.
    ///
    /// Estimates the angular resolution (Rayleigh criterion) for the array.
    ///
    /// # Arguments
    ///
    /// * `aperture_size` - Physical size of array aperture [m]
    ///
    /// # Returns
    ///
    /// Beam width in radians (angular resolution).
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// Δθ ≈ 1.22 · λ / D
    /// ```
    ///
    /// # Delegation
    ///
    /// Delegates to `math::geometry::delays::calculate_beam_width`.
    #[must_use]
    pub fn calculate_beam_width(&self, aperture_size: f64) -> f64 {
        delays::calculate_beam_width(aperture_size, self.frequency, self.sound_speed)
            .expect("Beam width calculation failed")
    }

    /// Calculate focal zone depth (depth of field).
    ///
    /// Estimates the axial extent over which the beam remains focused.
    ///
    /// # Arguments
    ///
    /// * `aperture_size` - Physical size of array aperture [m]
    /// * `focal_distance` - Distance from array to focal point [m]
    ///
    /// # Returns
    ///
    /// Depth of field in meters (axial extent around focal point).
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// DOF ≈ 7 · λ · F²
    /// ```
    ///
    /// where F = focal_distance / aperture_size (f-number).
    ///
    /// # Delegation
    ///
    /// Delegates to `math::geometry::delays::calculate_focal_zone`.
    #[must_use]
    pub fn calculate_focal_zone(&self, aperture_size: f64, focal_distance: f64) -> f64 {
        delays::calculate_focal_zone(
            aperture_size,
            focal_distance,
            self.frequency,
            self.sound_speed,
        )
        .expect("Focal zone calculation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_delays_symmetry() {
        let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);

        // Symmetric linear array
        let positions = vec![
            (-1.5e-3, 0.0, 0.0),
            (-0.5e-3, 0.0, 0.0),
            (0.5e-3, 0.0, 0.0),
            (1.5e-3, 0.0, 0.0),
        ];

        let target = (0.0, 0.0, 0.05); // Axial focus

        let delays = calculator.calculate_focus_delays(&positions, target);

        assert_eq!(delays.len(), 4);

        // Symmetric array should have symmetric delays
        assert!((delays[0] - delays[3]).abs() < 1e-10);
        assert!((delays[1] - delays[2]).abs() < 1e-10);

        // All delays should be non-negative (normalized)
        for &delay in &delays {
            assert!(delay >= 0.0);
        }
    }

    #[test]
    fn test_steering_delays_broadside() {
        let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);

        let positions = vec![(0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)];

        // Broadside: theta = 0 (axial)
        let delays = calculator.calculate_steering_delays(&positions, 0.0, 0.0);

        assert_eq!(delays.len(), 3);

        // All delays should be zero for broadside
        for &delay in &delays {
            assert!(delay.abs() < 1e-10);
        }
    }

    #[test]
    fn test_plane_wave_delays() {
        let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);

        let positions = vec![(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)];

        // Axial plane wave
        let direction = (0.0, 0.0, 1.0);
        let delays = calculator.calculate_plane_wave_delays(&positions, direction);

        assert_eq!(delays.len(), 2);

        // Broadside should have zero delays
        for &delay in &delays {
            assert!(delay.abs() < 1e-10);
        }
    }

    #[test]
    fn test_beam_width_positive() {
        let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);

        let beam_width = calculator.calculate_beam_width(0.01); // 1cm aperture

        assert!(beam_width > 0.0);
        assert!(beam_width.is_finite());
    }

    #[test]
    fn test_focal_zone_positive() {
        let calculator = BeamformingCalculator::with_medium(1540.0, 1e6);

        let focal_zone = calculator.calculate_focal_zone(0.01, 0.05); // 1cm aperture, 5cm focus

        assert!(focal_zone > 0.0);
        assert!(focal_zone.is_finite());
    }
}
