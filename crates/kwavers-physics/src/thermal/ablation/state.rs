use super::kinetics::AblationKinetics;
use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::DamageIntegral;
use kwavers_core::{
    constants::thermodynamic::KELVIN_OFFSET_C,
    error::{KwaversError, KwaversResult},
};

/// Tissue ablation state at a point
#[derive(Debug, Clone, Copy)]
pub struct AblationState {
    /// Temperature [°C]
    pub temperature: f64,
    /// Accumulated thermal damage (Ω)
    pub damage: f64,
    /// Tissue viability (0-1, 1=fully viable)
    pub viability: f64,
    /// Is tissue ablated?
    pub ablated: bool,
}

impl AblationState {
    /// Create new ablation state
    #[must_use]
    pub fn new(temperature: f64, _kinetics: &AblationKinetics) -> Self {
        Self {
            temperature,
            damage: 0.0,
            viability: 1.0,
            ablated: false,
        }
    }

    /// Update ablation state with temperature change
    ///
    /// # Errors
    ///
    /// Returns an error without mutating the state when the observation or
    /// existing damage violates the Asclepius contract.
    pub fn update(
        &mut self,
        temperature_c: f64,
        kinetics: &AblationKinetics,
        dt: f64,
    ) -> KwaversResult<()> {
        let temperature = ThermodynamicTemperature::from_base(temperature_c + KELVIN_OFFSET_C);
        let increment = kinetics.damage_increment(temperature, Time::from_base(dt))?;
        let current = DamageIntegral::new(self.damage).map_err(|source| {
            KwaversError::InvalidInput(format!("invalid accumulated ablation damage: {source}"))
        })?;
        let next = if temperature >= kinetics.ablation_threshold() {
            DamageIntegral::new(current.get() + increment.get()).map_err(|source| {
                KwaversError::InvalidInput(format!("invalid accumulated ablation damage: {source}"))
            })?
        } else {
            current
        };
        let viability = AblationKinetics::viability(next).get();
        let ablated = kinetics.is_ablated(next);

        self.temperature = temperature_c;
        self.damage = next.get();
        self.viability = viability;
        self.ablated = ablated;
        Ok(())
    }
}
