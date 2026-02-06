//! Interface properties and types for wave propagation
//!
//! This module defines interface properties and configurations for
//! wave propagation calculations at boundaries between different media.

use crate::physics::acoustics::analytical::propagation::AnalyticalMediumProperties;

/// Interface configuration between two media
#[derive(Debug, Clone)]
pub struct Interface {
    /// Properties of medium 1 (incident side)
    pub medium1: AnalyticalMediumProperties,
    /// Properties of medium 2 (transmission side)
    pub medium2: AnalyticalMediumProperties,
    /// Interface normal vector (unit vector)
    pub normal: [f64; 3],
    /// Interface position
    pub position: [f64; 3],
    /// Interface type
    pub interface_type: InterfaceType,
}

/// Interface type between media
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterfaceType {
    /// Planar interface
    Planar,
    /// Curved interface
    Curved,
    /// Rough interface with given RMS roughness
    Rough(f64),
    /// Layered interface (stratified media)
    Layered,
}

/// Interface properties
#[derive(Debug, Clone)]
pub struct InterfaceProperties {
    /// Interface type
    pub interface_type: InterfaceType,
    /// Surface roughness RMS \[m\]
    pub roughness: f64,
    /// Interface curvature radius \[m\] (for curved interfaces)
    pub curvature_radius: Option<f64>,
}

impl InterfaceProperties {
    /// Calculate normal reflection coefficient for an interface
    #[must_use]
    pub fn normal_reflection_coefficient(interface: &Interface) -> f64 {
        let z1 = interface.medium1.impedance();
        let z2 = interface.medium2.impedance();
        let r: f64 = (z2 - z1) / (z2 + z1);
        r.abs()
    }

    /// Calculate normal transmission coefficient for an interface
    #[must_use]
    pub fn normal_transmission_coefficient(interface: &Interface) -> f64 {
        let z1 = interface.medium1.impedance();
        let z2 = interface.medium2.impedance();
        let t: f64 = 2.0_f64 * z2 / (z2 + z1);
        t.abs()
    }
}
