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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::analytical::propagation::AnalyticalMediumProperties;

    /// Construct an `Interface` between two media with given acoustic impedances.
    ///
    /// Impedance Z = ρ·c; we set wave_speed=1 so density == impedance.
    fn make_interface(z1: f64, z2: f64) -> Interface {
        let medium1 = AnalyticalMediumProperties {
            wave_speed: 1.0,
            density: z1,
            refractive_index: 1.0,
            absorption: 0.0,
            anisotropy: None,
        };
        let medium2 = AnalyticalMediumProperties {
            wave_speed: 1.0,
            density: z2,
            refractive_index: 1.0,
            absorption: 0.0,
            anisotropy: None,
        };
        Interface {
            medium1,
            medium2,
            normal: [0.0, 0.0, 1.0],
            position: [0.0; 3],
            interface_type: InterfaceType::Planar,
        }
    }

    /// Impedance-matched interface (Z₁ = Z₂): R = 0, T = 1.
    ///
    /// Analytical: R = |Z₂−Z₁|/(Z₂+Z₁) = 0; T = 2Z₂/(Z₂+Z₁) = 1.
    #[test]
    fn reflection_zero_and_transmission_unity_for_matched_impedance() {
        let iface = make_interface(1500.0, 1500.0);
        let r = InterfaceProperties::normal_reflection_coefficient(&iface);
        let t = InterfaceProperties::normal_transmission_coefficient(&iface);
        assert!(r.abs() < 1e-14, "R must be 0 for matched impedance (got {r:.3e})");
        assert!((t - 1.0).abs() < 1e-14, "T must be 1 for matched impedance (got {t:.6})");
    }

    /// Concrete mismatch: Z₁ = 1000, Z₂ = 3000.
    ///
    /// Analytical: R = |3000−1000|/(3000+1000) = 2000/4000 = 0.5
    ///             T = 2·3000/(3000+1000) = 6000/4000 = 1.5
    #[test]
    fn reflection_and_transmission_match_analytical_for_3to1_impedance_ratio() {
        let iface = make_interface(1000.0, 3000.0);
        let r = InterfaceProperties::normal_reflection_coefficient(&iface);
        let t = InterfaceProperties::normal_transmission_coefficient(&iface);
        assert!((r - 0.5).abs() < 1e-14, "R must be 0.5 (got {r:.6})");
        assert!((t - 1.5).abs() < 1e-14, "T must be 1.5 (got {t:.6})");
    }

    /// Energy conservation: R² + (Z₁/Z₂)·T² = 1 for all impedance pairs.
    ///
    /// Derivation: power reflectance + power transmittance = 1:
    ///   R² + (Z₁/Z₂)·T² = 0.25 + (1/3)·2.25 = 0.25 + 0.75 = 1.0.
    #[test]
    fn energy_conservation_holds_for_normal_incidence() {
        let z1 = 1000.0_f64;
        let z2 = 3000.0_f64;
        let iface = make_interface(z1, z2);
        let r = InterfaceProperties::normal_reflection_coefficient(&iface);
        let t = InterfaceProperties::normal_transmission_coefficient(&iface);
        let energy_sum = r * r + (z1 / z2) * t * t;
        assert!(
            (energy_sum - 1.0).abs() < 1e-14,
            "R² + (Z₁/Z₂)·T² must equal 1 (got {energy_sum:.6})"
        );
    }

    /// `InterfaceType::Rough` carries the RMS roughness value.
    #[test]
    fn interface_type_rough_stores_roughness_value() {
        let roughness = 1e-4_f64;
        let t = InterfaceType::Rough(roughness);
        assert_eq!(t, InterfaceType::Rough(roughness));
        assert_ne!(t, InterfaceType::Planar);
    }
}
