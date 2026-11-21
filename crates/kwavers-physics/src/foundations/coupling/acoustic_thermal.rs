//! Acoustic-Thermal Coupling (Thermoacoustic Effects)
//!
//! ## Mathematical Foundation
//!
//! Pressure source from temperature changes:
//! ```text
//! ∂p/∂t = β ρ c² ∂T/∂t
//! ```
//!
//! Thermal damping penetration depth:
//! ```text
//! δ = √(2α/ω)
//! ```
//!
//! ## References
//!
//! - Swift (1988) "Thermoacoustic engines" JASA 84(4):1145-1180

use super::{CouplingPosition, MultiPhysicsCoupling};
use aequitas::systems::si::quantities::{
    Frequency, Length, MassDensity, PressureRate, ReciprocalTemperature, TemperatureDifference,
    ThermalDiffusivity, Time, Velocity, VolumetricPowerDensity,
};
use leto::Array3;

/// Acoustic-thermal coupling for thermoacoustic effects.
pub trait AcousticThermalCoupling: MultiPhysicsCoupling {
    /// Isobaric volume expansion coefficient.
    fn thermal_expansion_coefficient(&self, position: &CouplingPosition) -> ReciprocalTemperature;

    /// Mass density of the local medium.
    fn mass_density(&self, position: &CouplingPosition) -> MassDensity;

    /// Sound speed of the local medium.
    fn sound_speed(&self, position: &CouplingPosition) -> Velocity;

    /// Thermal diffusivity of the local medium.
    fn thermal_diffusivity(&self, position: &CouplingPosition) -> ThermalDiffusivity;

    /// Pressure rate induced by a temperature change over a positive timestep.
    fn pressure_source_from_temperature(
        &self,
        temperature_change: TemperatureDifference<&Array3<f64>>,
        duration: Time,
        position: &CouplingPosition,
    ) -> PressureRate<Array3<f64>> {
        let beta = self.thermal_expansion_coefficient(position);
        let rho = self.mass_density(position);
        let c = self.sound_speed(position);
        let gain = beta * rho * c * c / duration;
        PressureRate::from_base(
            temperature_change
                .into_base()
                .mapv(|change| (gain * TemperatureDifference::from_base(change)).into_base()),
        )
    }

    /// Volumetric heating from the local viscous dissipation model.
    fn viscous_heating(
        &self,
        velocity_field: Velocity<&Array3<[f64; 3]>>,
        position: &CouplingPosition,
    ) -> VolumetricPowerDensity;

    /// Thermal penetration depth for a positive frequency.
    fn thermal_penetration_depth(
        &self,
        frequency: Frequency,
        position: &CouplingPosition,
    ) -> Length {
        let diffusivity = self.thermal_diffusivity(position);
        (diffusivity * 2.0 / (frequency * std::f64::consts::TAU)).sqrt()
    }
}
