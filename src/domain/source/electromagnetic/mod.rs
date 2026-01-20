//! Electromagnetic source definitions and implementations
//!
//! This module defines electromagnetic sources for wave generation and excitation
//! in simulation domains.

use super::types::{EMWaveType, Polarization};
use num_complex::Complex;

/// Basic electromagnetic source trait
///
/// Defines the interface for electromagnetic wave sources in simulation domains.
pub trait EMSource: Send + Sync {
    /// Get source polarization
    fn polarization(&self) -> Polarization;

    /// Get source wave type
    fn wave_type(&self) -> EMWaveType;

    /// Get source frequency spectrum (Hz)
    fn frequency_spectrum(&self) -> Vec<f64>;

    /// Get peak electric field amplitude (V/m)
    fn peak_electric_field(&self) -> f64;

    /// Compute time-domain electric field at given time and position
    fn electric_field_at_time(&self, time: f64, position: &[f64]) -> [f64; 3];

    /// Compute frequency-domain electric field at given frequency and position
    fn electric_field_at_frequency(&self, frequency: f64, position: &[f64]) -> Complex<f64>;

    /// Check if source is active at given time
    fn is_active(&self, time: f64) -> bool;

    /// Get source directivity pattern (dimensionless, 0-1)
    fn directivity(&self, direction: &[f64]) -> f64;
}

/// Point source implementation
#[derive(Debug)]
pub struct PointEMSource {
    pub position: [f64; 3],
    pub polarization: Polarization,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
}

impl PointEMSource {
    pub fn new(position: [f64; 3], frequency: f64, amplitude: f64) -> Self {
        Self {
            position,
            polarization: Polarization::LinearX,
            frequency,
            amplitude,
            phase: 0.0,
        }
    }
}

impl EMSource for PointEMSource {
    fn polarization(&self) -> Polarization {
        self.polarization
    }

    fn wave_type(&self) -> EMWaveType {
        EMWaveType::TEM
    }

    fn frequency_spectrum(&self) -> Vec<f64> {
        vec![self.frequency]
    }

    fn peak_electric_field(&self) -> f64 {
        self.amplitude
    }

    fn electric_field_at_time(&self, time: f64, position: &[f64]) -> [f64; 3] {
        let distance = ((position[0] - self.position[0]).powi(2)
            + (position[1] - self.position[1]).powi(2)
            + (position[2] - self.position[2]).powi(2))
        .sqrt();

        if distance < 1e-10 {
            return [0.0, 0.0, 0.0]; // Avoid singularity at source
        }

        // Simplified spherical wave: E ∝ (1/r) sin(kr - ωt + φ)
        let k = 2.0 * std::f64::consts::PI * self.frequency / 3e8; // Wave number (c = 3e8 m/s approximation)
        let omega = 2.0 * std::f64::consts::PI * self.frequency;
        let phase = k * distance - omega * time + self.phase;

        let field_magnitude = self.amplitude / distance * phase.sin();

        match self.polarization {
            Polarization::LinearX => [field_magnitude, 0.0, 0.0],
            Polarization::LinearY => [0.0, field_magnitude, 0.0],
            _ => [field_magnitude, 0.0, 0.0], // Default to X-polarized
        }
    }

    fn electric_field_at_frequency(&self, frequency: f64, position: &[f64]) -> Complex<f64> {
        if (frequency - self.frequency).abs() > 1e-6 {
            return Complex::new(0.0, 0.0); // Off-resonance
        }

        let distance = ((position[0] - self.position[0]).powi(2)
            + (position[1] - self.position[1]).powi(2)
            + (position[2] - self.position[2]).powi(2))
        .sqrt();

        if distance < 1e-10 {
            return Complex::new(0.0, 0.0);
        }

        let k = 2.0 * std::f64::consts::PI * frequency / 3e8;
        let phase = k * distance + self.phase;

        // E ∝ (1/r) exp(ikr + iφ)
        let amplitude = self.amplitude / distance;
        Complex::from_polar(amplitude, phase)
    }

    fn is_active(&self, _time: f64) -> bool {
        true // Always active
    }

    fn directivity(&self, _direction: &[f64]) -> f64 {
        1.0 // Isotropic (simplified)
    }
}

/// Plane wave source
#[derive(Debug)]
pub struct PlaneWaveEMSource {
    pub direction: [f64; 3], // Propagation direction (unit vector)
    pub polarization: Polarization,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
}

impl PlaneWaveEMSource {
    pub fn new(direction: [f64; 3], frequency: f64, amplitude: f64) -> Self {
        // Normalize direction
        let norm = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        let normalized_dir = if norm > 0.0 {
            [
                direction[0] / norm,
                direction[1] / norm,
                direction[2] / norm,
            ]
        } else {
            [1.0, 0.0, 0.0] // Default to x-direction
        };

        Self {
            direction: normalized_dir,
            polarization: Polarization::LinearX,
            frequency,
            amplitude,
            phase: 0.0,
        }
    }
}

impl EMSource for PlaneWaveEMSource {
    fn polarization(&self) -> Polarization {
        self.polarization
    }

    fn wave_type(&self) -> EMWaveType {
        EMWaveType::TEM
    }

    fn frequency_spectrum(&self) -> Vec<f64> {
        vec![self.frequency]
    }

    fn peak_electric_field(&self) -> f64 {
        self.amplitude
    }

    fn electric_field_at_time(&self, time: f64, position: &[f64]) -> [f64; 3] {
        // Plane wave: E ∝ sin(k·r - ωt + φ)
        let k_dot_r = self.direction[0] * position[0]
            + self.direction[1] * position[1]
            + self.direction[2] * position[2];

        let omega = 2.0 * std::f64::consts::PI * self.frequency;
        let c = 3e8; // Speed of light approximation
        let k = omega / c;

        let phase = k * k_dot_r - omega * time + self.phase;
        let field_magnitude = self.amplitude * phase.sin();

        match self.polarization {
            Polarization::LinearX => [field_magnitude, 0.0, 0.0],
            Polarization::LinearY => [0.0, field_magnitude, 0.0],
            Polarization::LinearZ => [0.0, 0.0, field_magnitude],
            _ => [field_magnitude, 0.0, 0.0], // Default to X-polarized
        }
    }

    fn electric_field_at_frequency(&self, frequency: f64, position: &[f64]) -> Complex<f64> {
        if (frequency - self.frequency).abs() > 1e-6 {
            return Complex::new(0.0, 0.0);
        }

        let k_dot_r = self.direction[0] * position[0]
            + self.direction[1] * position[1]
            + self.direction[2] * position[2];

        let omega = 2.0 * std::f64::consts::PI * frequency;
        let c = 3e8;
        let k = omega / c;

        let phase = k * k_dot_r + self.phase;
        Complex::from_polar(self.amplitude, phase)
    }

    fn is_active(&self, _time: f64) -> bool {
        true
    }

    fn directivity(&self, direction: &[f64]) -> f64 {
        // For plane waves, directivity depends on angle between propagation and observation
        let cos_theta = self.direction[0] * direction[0]
            + self.direction[1] * direction[1]
            + self.direction[2] * direction[2];

        // Simplified: higher directivity in propagation direction
        (cos_theta + 1.0) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_source_creation() {
        let source = PointEMSource::new([0.0, 0.0, 0.0], 1e9, 1e6);
        assert_eq!(source.frequency, 1e9);
        assert_eq!(source.amplitude, 1e6);
    }

    #[test]
    fn test_point_source_at_zero_distance() {
        let source = PointEMSource::new([0.0, 0.0, 0.0], 1e9, 1e6);
        let field = source.electric_field_at_time(0.0, &[0.0, 0.0, 0.0]);
        assert_eq!(field, [0.0, 0.0, 0.0]); // Should be zero at source
    }

    #[test]
    fn test_plane_wave_directivity() {
        let source = PlaneWaveEMSource::new([1.0, 0.0, 0.0], 1e9, 1e6);

        // Forward direction should have high directivity
        let forward_dir = source.directivity(&[1.0, 0.0, 0.0]);
        assert!(forward_dir > 0.5);

        // Backward direction should have low directivity
        let backward_dir = source.directivity(&[-1.0, 0.0, 0.0]);
        assert!(backward_dir < forward_dir);
    }
}
