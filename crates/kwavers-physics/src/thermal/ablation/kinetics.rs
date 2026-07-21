//! Tissue-specific policy over the Asclepius Arrhenius law.
//!
//! Kwavers owns tissue parameter presets and ablation thresholds. Asclepius
//! owns the first-order damage rate, integration, survival, and probability
//! laws described in ADR 044.

use aequitas::systems::si::quantities::{
    MolarEnergy, MolarHeatCapacity, ReciprocalTime, ThermodynamicTemperature, Time,
};
use asclepius::{response::thermal::ArrheniusDamage, DamageIntegral, Probability};
use kwavers_core::{
    constants::{fundamental::GAS_CONSTANT, thermodynamic::KELVIN_OFFSET_C},
    error::{KwaversError, KwaversResult},
};

/// Tissue ablation policy backed by a typed Arrhenius damage law.
#[derive(Debug, Clone, Copy)]
pub struct AblationKinetics {
    law: ArrheniusDamage<f64>,
    damage_threshold: DamageIntegral<f64>,
    ablation_threshold: ThermodynamicTemperature<f64>,
}

impl AblationKinetics {
    /// Construct custom ablation kinetics.
    ///
    /// # Errors
    ///
    /// Returns an error when an Arrhenius parameter, damage threshold, or
    /// absolute ablation temperature is outside its mathematical domain.
    pub fn new(
        frequency_factor: f64,
        activation_energy: f64,
        damage_threshold: f64,
        ablation_threshold_c: f64,
    ) -> KwaversResult<Self> {
        let law = ArrheniusDamage::new(
            ReciprocalTime::from_base(frequency_factor),
            MolarEnergy::from_base(activation_energy),
            MolarHeatCapacity::from_base(GAS_CONSTANT),
        )
        .map_err(|source| {
            KwaversError::InvalidInput(format!("invalid ablation kinetics: {source}"))
        })?;
        let damage_threshold = DamageIntegral::new(damage_threshold).map_err(|source| {
            KwaversError::InvalidInput(format!("invalid ablation damage threshold: {source}"))
        })?;
        let ablation_threshold =
            ThermodynamicTemperature::from_base(ablation_threshold_c + KELVIN_OFFSET_C);
        law.rate(ablation_threshold).map_err(|source| {
            KwaversError::InvalidInput(format!("invalid ablation temperature threshold: {source}"))
        })?;
        Ok(Self {
            law,
            damage_threshold,
            ablation_threshold,
        })
    }

    /// Protein denaturation kinetics from Henriques (1947).
    #[must_use]
    pub fn protein_denaturation() -> Self {
        Self::from_validated_constants(1.0e44, 284_000.0, 1.0, 45.0)
    }

    /// Collagen denaturation kinetics from Lepock et al. (1993).
    #[must_use]
    pub fn collagen_denaturation() -> Self {
        Self::from_validated_constants(1.0e44, 250_000.0, 1.0, 55.0)
    }

    /// HIFU tissue-necrosis kinetics.
    #[must_use]
    pub fn hifu_ablation() -> Self {
        Self::from_validated_constants(1.0e47, 284_000.0, 1.0, 50.0)
    }

    /// Return the typed frequency factor.
    #[must_use]
    pub const fn frequency_factor(&self) -> ReciprocalTime<f64> {
        self.law.frequency_factor()
    }

    /// Return the typed activation energy.
    #[must_use]
    pub const fn activation_energy(&self) -> MolarEnergy<f64> {
        self.law.activation_energy()
    }

    /// Return the validated damage threshold.
    #[must_use]
    pub const fn damage_threshold(&self) -> DamageIntegral<f64> {
        self.damage_threshold
    }

    /// Return the absolute ablation temperature threshold.
    #[must_use]
    pub const fn ablation_threshold(&self) -> ThermodynamicTemperature<f64> {
        self.ablation_threshold
    }

    /// Evaluate the Asclepius damage rate.
    ///
    /// # Errors
    ///
    /// Returns an error when `temperature` is not finite and positive.
    pub fn damage_rate(
        &self,
        temperature: ThermodynamicTemperature<f64>,
    ) -> KwaversResult<ReciprocalTime<f64>> {
        self.law.rate(temperature).map_err(|source| {
            KwaversError::InvalidInput(format!("invalid ablation temperature: {source}"))
        })
    }

    /// Evaluate one Asclepius damage increment.
    ///
    /// # Errors
    ///
    /// Returns an error when the absolute temperature or time step is invalid.
    pub fn damage_increment(
        &self,
        temperature: ThermodynamicTemperature<f64>,
        step: Time<f64>,
    ) -> KwaversResult<DamageIntegral<f64>> {
        self.law.increment(temperature, step).map_err(|source| {
            KwaversError::InvalidInput(format!("invalid ablation observation: {source}"))
        })
    }

    /// Return the viable fraction implied by accumulated damage.
    #[must_use]
    pub fn viability(damage: DamageIntegral<f64>) -> Probability<f64> {
        ArrheniusDamage::survival(damage)
    }

    /// Return whether damage reaches this tissue policy's threshold.
    #[must_use]
    pub fn is_ablated(&self, damage: DamageIntegral<f64>) -> bool {
        damage >= self.damage_threshold
    }

    fn from_validated_constants(
        frequency_factor: f64,
        activation_energy: f64,
        damage_threshold: f64,
        ablation_threshold_c: f64,
    ) -> Self {
        Self::new(
            frequency_factor,
            activation_energy,
            damage_threshold,
            ablation_threshold_c,
        )
        .expect("invariant: published ablation preset parameters are valid")
    }
}

impl Default for AblationKinetics {
    fn default() -> Self {
        Self::hifu_ablation()
    }
}
