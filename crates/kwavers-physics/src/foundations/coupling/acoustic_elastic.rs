//! Acoustic-Elastic Coupling for Fluid-Solid Interfaces
//!
//! ## Mathematical Foundation
//!
//! At a fluid-solid interface:
//! - Normal stress continuity: σ·n = -p (pressure = normal stress)
//! - Normal velocity continuity: v_n(fluid) = v_n(solid)
//! - Tangential velocity: free-slip (inviscid fluid) or no-slip (viscous)
//!
//! Reflection coefficient: R = (Z₂ - Z₁)/(Z₂ + Z₁)
//! Transmission coefficient: T = 2Z₂/(Z₁ + Z₂)
//!
//! ## References
//!
//! - Kinsler et al. (2000) "Fundamentals of Acoustics" Ch. 6

use super::MultiPhysicsCoupling;

/// Acoustic-elastic coupling for fluid-solid interfaces
pub trait AcousticElasticCoupling: MultiPhysicsCoupling {
    /// Normal stress continuity σ·n (Pa)
    fn normal_stress(&self, position: &[f64]) -> f64;

    /// Tangential velocity continuity v_τ (m/s)
    fn tangential_velocity(&self, position: &[f64]) -> [f64; 2];

    /// Acoustic impedance matching Z = ρc (kg/m²s)
    fn acoustic_impedance(&self, position: &[f64]) -> f64;

    /// Reflection coefficient R = (Z2 - Z1)/(Z2 + Z1)
    fn reflection_coefficient(&self, position: &[f64]) -> f64 {
        let z1 = self.acoustic_impedance(&[position[0] - 1e-6, position[1], position[2]]);
        let z2 = self.acoustic_impedance(&[position[0] + 1e-6, position[1], position[2]]);
        (z2 - z1) / (z2 + z1)
    }

    /// Transmission coefficient T = 2Z2/(Z1 + Z2)
    fn transmission_coefficient(&self, position: &[f64]) -> f64 {
        let z1 = self.acoustic_impedance(&[position[0] - 1e-6, position[1], position[2]]);
        let z2 = self.acoustic_impedance(&[position[0] + 1e-6, position[1], position[2]]);
        2.0 * z2 / (z1 + z2)
    }
}
