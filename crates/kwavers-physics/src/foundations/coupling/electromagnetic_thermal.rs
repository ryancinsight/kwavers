//! Electromagnetic-Thermal Coupling (Photothermal Effects)
//!
//! ## Mathematical Foundation
//!
//! Bioheat equation (Pennes 1948):
//! ```text
//! ρ c_p ∂T/∂t = k ∇²T + Q_optical - w_b ρ_b c_b (T - T_b) + Q_met
//! ```
//!
//! Optical heating rate: Q = μ_a Φ
//!
//! ## References
//!
//! - Pennes (1948) J Appl Physiol 1(2):93-122
//! - Welch & van Gemert (2011) "Optical-Thermal Response of Laser-Irradiated Tissue"

use super::{CouplingPosition, MultiPhysicsCoupling};
use aequitas::systems::si::quantities::{
    Intensity, Length, MassDensity, ReciprocalTime, SpecificHeatCapacity, ThermalDiffusivity,
    ThermodynamicTemperature, Time, VolumetricPowerDensity,
};
use hyperion::coefficient::{Absorption, InteractionCoefficient};
use leto::Array3;

/// Electromagnetic-thermal coupling for photothermal effects.
pub trait ElectromagneticThermalCoupling: MultiPhysicsCoupling {
    /// Optical absorption coefficient at the given wavelength.
    fn optical_absorption_coefficient(
        &self,
        position: &CouplingPosition,
        wavelength: Length,
    ) -> InteractionCoefficient<f64, Absorption>;

    /// Local thermal diffusivity.
    fn thermal_diffusivity(&self, position: &CouplingPosition) -> ThermalDiffusivity;

    /// Blood volume perfusion per tissue volume per unit time.
    fn perfusion_rate(&self, position: &CouplingPosition) -> ReciprocalTime;

    /// Blood mass density.
    fn blood_density(&self, position: &CouplingPosition) -> MassDensity;

    /// Blood heat capacity per unit mass.
    fn blood_specific_heat(&self, position: &CouplingPosition) -> SpecificHeatCapacity;

    /// Arterial blood reference temperature.
    fn blood_temperature(&self, position: &CouplingPosition) -> ThermodynamicTemperature;

    /// Volumetric optical heating Q = μₐ Φ̇.
    fn optical_heating_rate(
        &self,
        fluence_rate: Intensity<&Array3<f64>>,
        position: &CouplingPosition,
        wavelength: Length,
    ) -> VolumetricPowerDensity<Array3<f64>> {
        let absorption = self
            .optical_absorption_coefficient(position, wavelength)
            .into_quantity();
        VolumetricPowerDensity::from_base(
            fluence_rate
                .into_base()
                .mapv(|value| (absorption * Intensity::from_base(value)).into_base()),
        )
    }

    /// Diffusion timescale L²/α at the supplied length scale.
    fn thermal_relaxation_time(&self, length: Length, position: &CouplingPosition) -> Time {
        length * length / self.thermal_diffusivity(position)
    }

    /// Signed Pennes perfusion cooling q = w ρ_b c_b (T − T_b).
    fn perfusion_cooling(
        &self,
        temperature: ThermodynamicTemperature,
        position: &CouplingPosition,
    ) -> VolumetricPowerDensity {
        let difference = temperature - self.blood_temperature(position);
        self.perfusion_rate(position)
            * self.blood_density(position)
            * self.blood_specific_heat(position)
            * difference
    }

    /// Net optical and perfusion source in W/m³.
    fn bioheat_source(
        &self,
        fluence_rate: Intensity<&Array3<f64>>,
        temperature: ThermodynamicTemperature,
        position: &CouplingPosition,
        wavelength: Length,
    ) -> VolumetricPowerDensity<Array3<f64>> {
        let net = self.optical_heating_rate(fluence_rate, position, wavelength);
        let perfusion = self.perfusion_cooling(temperature, position);
        VolumetricPowerDensity::from_base(net.into_base().mapv(|q| q - perfusion.into_base()))
    }
}
