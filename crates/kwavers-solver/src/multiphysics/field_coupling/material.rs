use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, Intensity, MassDensity, Pressure, ReciprocalLength, ReciprocalTime,
    SpecificHeatCapacity, TemperatureDifference, Time, Velocity,
};
use hyperion::coefficient::{Absorption, InteractionCoefficient};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::ops::{Div, Mul};

/// Change in refractive index per pascal of acoustic pressure.
pub type PhotoelasticCoefficient = <Dimensionless as Div<Pressure>>::Output;

/// Local material properties evaluated at the requested acoustic frequency.
#[derive(Debug, Clone, Copy)]
pub struct FieldCouplingMaterial {
    /// Signed derivative of the refractive index with respect to pressure.
    pub photoelastic_coefficient: PhotoelasticCoefficient,
    /// Rate that converts the refractive-index perturbation to relative intensity change.
    pub photoelastic_response_rate: ReciprocalTime,
    /// Optical absorption coefficient for the simulated optical band.
    pub optical_absorption: InteractionCoefficient<f64, Absorption>,
    /// Acoustic intensity absorption coefficient in `I(x) = I(0) exp(−αx)`.
    /// An amplitude loss coefficient must be doubled before supplying it here.
    pub acoustic_intensity_absorption: ReciprocalLength,
    /// Local mass density.
    pub density: MassDensity,
    /// Local speed of sound.
    pub sound_speed: Velocity,
    /// Heat capacity per unit mass.
    pub specific_heat: SpecificHeatCapacity,
}

/// Supplies physical properties at a collocated field cell and acoustic frequency.
pub trait FieldCouplingMedium {
    /// Resolve all coefficients at the requested cell and acoustic frequency.
    ///
    /// # Errors
    /// Returns an error when the requested cell or frequency is outside the model.
    fn material_at(
        &self,
        cell: [usize; 3],
        acoustic_frequency: Frequency,
    ) -> KwaversResult<FieldCouplingMaterial>;
}

impl<F> FieldCouplingMedium for F
where
    F: Fn([usize; 3], Frequency) -> KwaversResult<FieldCouplingMaterial>,
{
    fn material_at(
        &self,
        cell: [usize; 3],
        frequency: Frequency,
    ) -> KwaversResult<FieldCouplingMaterial> {
        self(cell, frequency)
    }
}

impl FieldCouplingMedium for FieldCouplingMaterial {
    fn material_at(&self, _: [usize; 3], _: Frequency) -> KwaversResult<FieldCouplingMaterial> {
        Ok(*self)
    }
}

type PressureSquared = <Pressure as Mul<Pressure>>::Output;
type OpticalHeatGain = <TemperatureDifference as Div<Intensity>>::Output;
type AcousticHeatGain = <TemperatureDifference as Div<PressureSquared>>::Output;

/// Per-step gains retain their dimensions until the field is updated.
#[derive(Debug, Clone, Copy)]
pub(super) struct FieldCouplingGains {
    pub(super) photoelastic: PhotoelasticCoefficient,
    pub(super) optical: OpticalHeatGain,
    pub(super) acoustic: AcousticHeatGain,
}

impl FieldCouplingMaterial {
    pub(super) fn gains(self, duration: Time) -> KwaversResult<FieldCouplingGains> {
        let impedance = self.density * self.sound_speed;
        let heat_capacity = self.density * self.specific_heat;
        let acoustic_denominator = impedance * 2.0 * heat_capacity;
        let gains = FieldCouplingGains {
            photoelastic: self.photoelastic_coefficient
                * (self.photoelastic_response_rate * duration),
            optical: self.optical_absorption.into_quantity() * duration / heat_capacity,
            acoustic: self.acoustic_intensity_absorption * duration / acoustic_denominator,
        };
        let positive = [
            self.density.into_base(),
            self.sound_speed.into_base(),
            self.specific_heat.into_base(),
            impedance.into_base(),
            heat_capacity.into_base(),
            acoustic_denominator.into_base(),
        ];
        let nonnegative = [
            self.photoelastic_response_rate.into_base(),
            self.acoustic_intensity_absorption.into_base(),
        ];
        if positive
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0)
            && nonnegative
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            && [
                gains.photoelastic.into_base(),
                gains.optical.into_base(),
                gains.acoustic.into_base(),
            ]
            .into_iter()
            .all(f64::is_finite)
        {
            Ok(gains)
        } else {
            Err(KwaversError::InvalidInput(
                "invalid material properties for field coupling".into(),
            ))
        }
    }

    pub(super) fn nominal() -> KwaversResult<Self> {
        use kwavers_core::constants::fundamental::{
            DENSITY_WATER_NOMINAL, OPTICAL_ABSORPTION_TISSUE_NIR, SOUND_SPEED_TISSUE,
        };
        use kwavers_core::constants::thermodynamic::SPECIFIC_HEAT_WATER;

        // The legacy frequency-independent step applies a fixed 0.5 m⁻¹
        // intensity loss. Keep it independent of any spectral slope.
        const NOMINAL_ACOUSTIC_INTENSITY_ABSORPTION: ReciprocalLength =
            ReciprocalLength::from_base(0.5);

        let optical_absorption =
            InteractionCoefficient::new(ReciprocalLength::from_base(OPTICAL_ABSORPTION_TISSUE_NIR))
                .map_err(|error| KwaversError::InvalidInput(error.to_string()))?;
        Ok(Self {
            photoelastic_coefficient: PhotoelasticCoefficient::from_base(1e-12),
            photoelastic_response_rate: ReciprocalTime::from_base(1.0),
            optical_absorption,
            acoustic_intensity_absorption: NOMINAL_ACOUSTIC_INTENSITY_ABSORPTION,
            density: MassDensity::from_base(DENSITY_WATER_NOMINAL),
            sound_speed: Velocity::from_base(SOUND_SPEED_TISSUE),
            specific_heat: SpecificHeatCapacity::from_base(SPECIFIC_HEAT_WATER),
        })
    }
}
