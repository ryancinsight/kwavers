//! Common types for wave propagation module

use super::interface::InterfaceType;
use super::medium_properties::MediumProperties;

/// Wave propagation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveMode {
    /// Acoustic pressure wave
    Acoustic,
    /// Optical electromagnetic wave
    Optical,
}

/// Polarization state for electromagnetic waves
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Transverse electric (s-polarized)
    TransverseElectric,
    /// Transverse magnetic (p-polarized)
    TransverseMagnetic,
    /// Unpolarized
    Unpolarized,
    /// Circular polarization
    Circular,
    /// Elliptical polarization
    Elliptical,
}

/// Interface configuration between two media
#[derive(Debug, Clone)]
pub struct Interface {
    /// Properties of medium 1 (incident side)
    pub medium1: MediumProperties,
    /// Properties of medium 2 (transmission side)
    pub medium2: MediumProperties,
    /// Interface normal vector (unit vector)
    pub normal: [f64; 3],
    /// Interface position
    pub position: [f64; 3],
    /// Interface type
    pub interface_type: InterfaceType,
}

impl Interface {
    /// Create a planar interface
    pub fn planar(
        medium1: MediumProperties,
        medium2: MediumProperties,
        normal: [f64; 3],
        position: [f64; 3],
    ) -> Self {
        Self {
            medium1,
            medium2,
            normal,
            position,
            interface_type: InterfaceType::Planar,
        }
    }

    /// Check if critical angle exists
    pub fn has_critical_angle(&self) -> bool {
        self.medium1.wave_speed < self.medium2.wave_speed
    }

    /// Calculate impedance mismatch
    pub fn impedance_mismatch(&self) -> f64 {
        let z1 = self.medium1.acoustic_impedance();
        let z2 = self.medium2.acoustic_impedance();
        (z2 - z1) / (z2 + z1)
    }
}
