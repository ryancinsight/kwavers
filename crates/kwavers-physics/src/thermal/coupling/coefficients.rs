//! Temperature dependence of acoustic properties
//!
//! Most tissue properties change with temperature:
//! - Sound speed: ∂c/∂T ≈ 2 m/s/°C
//! - Density: ∂ρ/∂T ≈ -0.5 kg/m³/°C
//! - Absorption: ∂α/∂T varies by tissue

use aequitas::systems::si::{
    quantities::{
        MassDensity, MassDensityPerTemperature, ReciprocalLength, ReciprocalLengthPerTemperature,
        TemperatureDifference, ThermodynamicTemperature, Velocity, VelocityPerTemperature,
    },
    units::{KilogramPerCubicMeterKelvin, MeterPerSecondKelvin, PerMeterKelvin},
};

/// Temperature coefficients for acoustic properties
#[derive(Debug, Clone, Copy)]
pub struct TemperatureCoefficients {
    /// Sound speed temperature coefficient [m/(s·K)].
    pub sound_speed_coefficient: VelocityPerTemperature<f64>,
    /// Density temperature coefficient [kg/(m³·K)].
    pub density_coefficient: MassDensityPerTemperature<f64>,
    /// Absorption temperature coefficient [1/(m·K)].
    pub absorption_coefficient: ReciprocalLengthPerTemperature<f64>,
}

impl TemperatureCoefficients {
    /// Create custom temperature coefficients
    #[must_use]
    pub fn new(
        sound_speed_coefficient: VelocityPerTemperature<f64>,
        density_coefficient: MassDensityPerTemperature<f64>,
        absorption_coefficient: ReciprocalLengthPerTemperature<f64>,
    ) -> Self {
        Self {
            sound_speed_coefficient,
            density_coefficient,
            absorption_coefficient,
        }
    }

    /// Generic soft tissue coefficients
    /// Reference: Duck (1990), Szabo (2004)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            sound_speed_coefficient: VelocityPerTemperature::from_unit::<MeterPerSecondKelvin>(2.0),
            density_coefficient: MassDensityPerTemperature::from_unit::<KilogramPerCubicMeterKelvin>(
                -0.5,
            ),
            absorption_coefficient: ReciprocalLengthPerTemperature::from_unit::<PerMeterKelvin>(
                0.015,
            ),
        }
    }

    /// Water coefficients
    /// Reference: IEC 61161:2013
    #[must_use]
    pub fn water() -> Self {
        Self {
            sound_speed_coefficient: VelocityPerTemperature::from_unit::<MeterPerSecondKelvin>(4.0),
            density_coefficient: MassDensityPerTemperature::from_unit::<KilogramPerCubicMeterKelvin>(
                -0.2,
            ),
            absorption_coefficient: ReciprocalLengthPerTemperature::from_unit::<PerMeterKelvin>(
                0.0,
            ),
        }
    }

    /// Blood coefficients
    /// Reference: Gordon et al. (2009)
    #[must_use]
    pub fn blood() -> Self {
        Self {
            sound_speed_coefficient: VelocityPerTemperature::from_unit::<MeterPerSecondKelvin>(2.5),
            density_coefficient: MassDensityPerTemperature::from_unit::<KilogramPerCubicMeterKelvin>(
                -0.6,
            ),
            absorption_coefficient: ReciprocalLengthPerTemperature::from_unit::<PerMeterKelvin>(
                0.02,
            ),
        }
    }

    /// Bone coefficients
    /// Reference: Duck (1990)
    #[must_use]
    pub fn bone() -> Self {
        Self {
            sound_speed_coefficient: VelocityPerTemperature::from_unit::<MeterPerSecondKelvin>(1.0),
            density_coefficient: MassDensityPerTemperature::from_unit::<KilogramPerCubicMeterKelvin>(
                -0.1,
            ),
            absorption_coefficient: ReciprocalLengthPerTemperature::from_unit::<PerMeterKelvin>(
                0.005,
            ),
        }
    }

    /// Sound speed at temperature
    #[must_use]
    pub fn sound_speed(
        &self,
        base_sound_speed: Velocity<f64>,
        temperature: ThermodynamicTemperature<f64>,
        reference_temperature: ThermodynamicTemperature<f64>,
    ) -> Velocity<f64> {
        let delta: TemperatureDifference<f64> = temperature - reference_temperature;
        base_sound_speed + self.sound_speed_coefficient * delta
    }

    /// Density at temperature
    #[must_use]
    pub fn density(
        &self,
        base_density: MassDensity<f64>,
        temperature: ThermodynamicTemperature<f64>,
        reference_temperature: ThermodynamicTemperature<f64>,
    ) -> MassDensity<f64> {
        let delta: TemperatureDifference<f64> = temperature - reference_temperature;
        base_density + self.density_coefficient * delta
    }

    /// Absorption at temperature
    #[must_use]
    pub fn absorption(
        &self,
        base_absorption: ReciprocalLength<f64>,
        temperature: ThermodynamicTemperature<f64>,
        reference_temperature: ThermodynamicTemperature<f64>,
    ) -> ReciprocalLength<f64> {
        let delta: TemperatureDifference<f64> = temperature - reference_temperature;
        let candidate: ReciprocalLength<f64> =
            base_absorption + self.absorption_coefficient * delta;
        ReciprocalLength::from_base(candidate.into_base().max(0.0))
    }
}

impl Default for TemperatureCoefficients {
    fn default() -> Self {
        Self::soft_tissue()
    }
}
