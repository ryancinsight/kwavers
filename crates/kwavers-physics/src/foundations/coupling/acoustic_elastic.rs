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

use super::{CouplingPosition, MultiPhysicsCoupling};
use aequitas::systems::si::quantities::{AcousticImpedance, Dimensionless, Stress, Velocity};
use eunomia::FloatElement;

/// Acoustic-elastic coupling for fluid-solid interfaces.
pub trait AcousticElasticCoupling: MultiPhysicsCoupling {
    /// Signed normal stress.
    fn normal_stress(&self, position: &CouplingPosition) -> Stress;

    /// Tangential velocity in the interface basis.
    fn tangential_velocity(&self, position: &CouplingPosition) -> [Velocity; 2];

    /// Positive, finite specific acoustic impedance.
    fn acoustic_impedance(&self, position: &CouplingPosition) -> AcousticImpedance;

    /// Pressure reflection at normal incidence. Supply samples on opposite sides.
    fn reflection_coefficient(
        &self,
        incident: &CouplingPosition,
        transmitted: &CouplingPosition,
    ) -> Dimensionless {
        pressure_coefficients(
            self.acoustic_impedance(incident),
            self.acoustic_impedance(transmitted),
        )[0]
    }

    /// Pressure transmission at normal incidence. Supply samples on opposite sides.
    fn transmission_coefficient(
        &self,
        incident: &CouplingPosition,
        transmitted: &CouplingPosition,
    ) -> Dimensionless {
        pressure_coefficients(
            self.acoustic_impedance(incident),
            self.acoustic_impedance(transmitted),
        )[1]
    }
}

/// For positive finite impedances, dividing by their maximum puts one
/// normalized impedance at 1 and the other in [0, 1]. The common factor
/// cancels analytically, and the sum and doubled numerator cannot overflow.
pub(super) fn pressure_coefficients<T: FloatElement>(
    incident: AcousticImpedance<T>,
    transmitted: AcousticImpedance<T>,
) -> [Dimensionless<T>; 2] {
    let scale = incident.max(transmitted);
    let first = incident / scale;
    let second = transmitted / scale;
    let sum = first + second;
    [(second - first) / sum, (second + second) / sum]
}
