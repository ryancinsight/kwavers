//! Beamforming algorithms for phased array control
//!
//! Implements various beamforming strategies including focusing and steering

use std::f64::consts::PI;

/// Beamforming algorithms for phased array control
#[derive(Debug, Clone)]
pub enum BeamformingMode {
    /// Focus at specific point (x, y, z) [m]
    Focus { target: (f64, f64, f64) },
    /// Steer beam to angle (theta, phi) [rad]
    Steer { theta: f64, phi: f64 },
    /// Custom phase delays [rad]
    Custom { delays: Vec<f64> },
    /// Plane wave transmission
    PlaneWave { direction: (f64, f64, f64) },
}

/// Beamforming calculator
#[derive(Debug, Debug)]
pub struct BeamformingCalculator {
    sound_speed: f64,
    frequency: f64,
}

impl BeamformingCalculator {
    /// Create calculator with medium properties
    pub fn with_medium(sound_speed: f64, frequency: f64) -> Self {
        Self {
            sound_speed,
            frequency,
        }
    }

    /// Calculate phase delays for focusing
    pub fn calculate_focus_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        target: (f64, f64, f64),
    ) -> Vec<f64> {
        let wavelength = self.sound_speed / self.frequency;
        let k = 2.0 * PI / wavelength;

        // Find maximum distance for reference
        let distances: Vec<f64> = element_positions
            .iter()
            .map(|pos| distance_3d(*pos, target))
            .collect();

        let max_distance = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate relative delays
        distances.iter().map(|&d| k * (max_distance - d)).collect()
    }

    /// Calculate phase delays for beam steering
    pub fn calculate_steering_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        theta: f64,
        phi: f64,
    ) -> Vec<f64> {
        let wavelength = self.sound_speed / self.frequency;
        let k = 2.0 * PI / wavelength;

        // Steering vector components
        let kx = k * theta.sin() * phi.cos();
        let ky = k * theta.sin() * phi.sin();
        let kz = k * theta.cos();

        element_positions
            .iter()
            .map(|(x, y, z)| -(kx * x + ky * y + kz * z))
            .collect()
    }

    /// Calculate delays for plane wave
    pub fn calculate_plane_wave_delays(
        &self,
        element_positions: &[(f64, f64, f64)],
        direction: (f64, f64, f64),
    ) -> Vec<f64> {
        let wavelength = self.sound_speed / self.frequency;
        let k = 2.0 * PI / wavelength;

        // Normalize direction
        let norm = (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt();
        let dir_norm = (direction.0 / norm, direction.1 / norm, direction.2 / norm);

        element_positions
            .iter()
            .map(|(x, y, z)| -k * (dir_norm.0 * x + dir_norm.1 * y + dir_norm.2 * z))
            .collect()
    }

    /// Calculate beam width for given configuration
    pub fn calculate_beam_width(&self, aperture_size: f64) -> f64 {
        // Rayleigh criterion for beam width
        let wavelength = self.sound_speed / self.frequency;
        1.22 * wavelength / aperture_size
    }

    /// Calculate focal zone depth
    pub fn calculate_focal_zone(&self, aperture_size: f64, focal_distance: f64) -> f64 {
        // Depth of field calculation
        let wavelength = self.sound_speed / self.frequency;
        let f_number = focal_distance / aperture_size;
        7.0 * wavelength * f_number.powi(2)
    }
}

/// Calculate 3D Euclidean distance
fn distance_3d(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
    ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2) + (p2.2 - p1.2).powi(2)).sqrt()
}
