use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::fusion::algorithms::MultiModalFusion;

/// Validation descriptor for weighted-average fusion.
#[derive(Debug, Clone)]
pub struct WeightedAverageValidationCase {
    pub name: &'static str,
    pub expected_modalities: usize,
}

/// Validate the retained weighted-average fusion inputs.
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
///
pub fn validate_weighted_average_inputs(fusion: &MultiModalFusion) -> KwaversResult<()> {
    if fusion.registered_data.len() < 2 {
        return Err(KwaversError::Validation(
            crate::core::error::ValidationError::ConstraintViolation {
                message: "At least two modalities required for weighted-average fusion".to_owned(),
            },
        ));
    }

    let total_weight: f64 = fusion
        .registered_data
        .keys()
        .map(|name| {
            fusion
                .config
                .modality_weights
                .get(name)
                .copied()
                .unwrap_or(1.0)
        })
        .sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(KwaversError::Validation(
            crate::core::error::ValidationError::ConstraintViolation {
                message: "FusionConfig.modality_weights must sum to a positive finite value"
                    .to_owned(),
            },
        ));
    }

    Ok(())
}
