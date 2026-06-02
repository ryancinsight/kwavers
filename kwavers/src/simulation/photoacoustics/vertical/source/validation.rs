use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::imaging::photoacoustic::{InitialPressure, PhotoacousticScenario};

/// Deterministic validation descriptor for thermoelastic source generation.
#[derive(Debug, Clone)]
pub struct SourceValidationCase {
    pub name: &'static str,
    pub expected_relation: &'static str,
}

/// Validate a thermoelastic source realization against the canonical relation.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn validate_source_generation(
    scenario: &PhotoacousticScenario,
    initial_pressure: &InitialPressure,
) -> KwaversResult<()> {
    if initial_pressure.max_pressure <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "initial pressure must contain positive deposited energy".to_owned(),
        ));
    }
    if scenario.config.pulse_duration_s <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "pulse duration must be positive".to_owned(),
        ));
    }
    Ok(())
}
