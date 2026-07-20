use aequitas::systems::si::quantities::{
    ReciprocalTemperature, ReciprocalTemperatureSquared, ThermodynamicTemperature,
};
use kwavers_core::constants::thermodynamic::{BODY_TEMPERATURE_K, ROOM_TEMPERATURE_K};
use proteus::{
    ConstantResponse, ConstitutiveLaw, LinearResponse, QuadraticResponse, ResponseSet,
    TemperatureLaw,
};

use super::super::ThermalPropertyData;

type ThermalResponses = ResponseSet<ConstantResponse, LinearResponse<f64>, QuadraticResponse<f64>>;
type ThermalLaw = TemperatureLaw<f64, ThermalResponses>;

/// Temperature-dependent thermal properties backed by the Proteus law.
///
/// ## Model
///
/// - Conductivity: `k(T) = k₀[1 + κ₁(T − T₀) + κ₂(T − T₀)²]`
/// - Specific heat: `c_p(T) = c₀[1 + c₁(T − T₀)]`
/// - Density: constant
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDependentThermal {
    law: ThermalLaw,
    blood_perfusion: Option<f64>,
    blood_specific_heat: Option<f64>,
}

impl TemperatureDependentThermal {
    /// Construct a dimensionally typed thermal response.
    ///
    /// # Errors
    ///
    /// Returns an error when the reference temperature is outside
    /// `[273, 373] K`, a response coefficient is non-finite, or Proteus rejects
    /// the reference state.
    pub fn new(
        base_properties: ThermalPropertyData,
        reference_temperature: f64,
        conductivity_coeff_linear: f64,
        conductivity_coeff_quadratic: f64,
        specific_heat_coefficient: f64,
    ) -> Result<Self, String> {
        if !(273.0..=373.0).contains(&reference_temperature) {
            return Err(format!(
                "Reference temperature {reference_temperature} K is outside valid range [273, 373] K"
            ));
        }

        let responses = ResponseSet::new(
            ConstantResponse,
            LinearResponse::new(ReciprocalTemperature::from_base(specific_heat_coefficient))
                .map_err(|error| error.to_string())?,
            QuadraticResponse::new(
                ReciprocalTemperature::from_base(conductivity_coeff_linear),
                ReciprocalTemperatureSquared::from_base(conductivity_coeff_quadratic),
            )
            .map_err(|error| error.to_string())?,
        );
        let law = TemperatureLaw::new(
            *base_properties.thermophysical(),
            ThermodynamicTemperature::from_base(reference_temperature),
            responses,
        )
        .map_err(|error| error.to_string())?;

        Ok(Self {
            law,
            blood_perfusion: base_properties.blood_perfusion,
            blood_specific_heat: base_properties.blood_specific_heat,
        })
    }

    /// Evaluate the complete thermal property bundle at `temperature` kelvin.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-physical temperature or response factor.
    pub fn properties(&self, temperature: f64) -> Result<ThermalPropertyData, String> {
        let temperature = ThermodynamicTemperature::from_base(temperature);
        let thermophysical = self
            .law
            .properties(&temperature)
            .map_err(|error| error.to_string())?;
        ThermalPropertyData::from_thermophysical(
            thermophysical,
            self.blood_perfusion,
            self.blood_specific_heat,
        )
    }

    pub(super) fn properties_with_density(
        &self,
        temperature: f64,
        density: f64,
    ) -> Result<ThermalPropertyData, String> {
        let properties = self.properties(temperature)?;
        ThermalPropertyData::new(
            properties.conductivity(),
            properties.specific_heat(),
            density,
            self.blood_perfusion,
            self.blood_specific_heat,
        )
    }

    /// Evaluate thermal conductivity in W/(m·K).
    ///
    /// # Errors
    ///
    /// Returns an error for a non-physical temperature or response factor.
    pub fn conductivity(&self, temperature: f64) -> Result<f64, String> {
        self.properties(temperature)
            .map(|properties| properties.conductivity())
    }

    /// Evaluate specific heat capacity in J/(kg·K).
    ///
    /// # Errors
    ///
    /// Returns an error for a non-physical temperature or response factor.
    pub fn specific_heat(&self, temperature: f64) -> Result<f64, String> {
        self.properties(temperature)
            .map(|properties| properties.specific_heat())
    }

    /// Evaluate `α(T) = k(T) / (ρ × c_p(T))` in m²/s.
    ///
    /// The supplied density is the acoustic model's temperature-dependent
    /// density, preserving the combined material contract.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-physical temperature, response factor, or
    /// acoustic density.
    pub fn thermal_diffusivity(&self, temperature: f64, density: f64) -> Result<f64, String> {
        self.properties_with_density(temperature, density)
            .map(|properties| properties.thermal_diffusivity())
    }

    /// Water properties.
    #[must_use]
    pub fn water() -> Self {
        Self::new(
            ThermalPropertyData::water(),
            ROOM_TEMPERATURE_K,
            0.002,
            -1e-5,
            0.0001,
        )
        .expect("water temperature coefficients satisfy the Proteus contract")
    }

    /// Soft-tissue properties.
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self::new(
            ThermalPropertyData::soft_tissue(),
            BODY_TEMPERATURE_K,
            0.001,
            0.0,
            0.0002,
        )
        .expect("soft-tissue temperature coefficients satisfy the Proteus contract")
    }
}
