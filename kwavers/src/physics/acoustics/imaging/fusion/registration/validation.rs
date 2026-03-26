use crate::core::error::{KwaversError, KwaversResult};

/// Validate that image dimensions are compatible for registration
///
/// Checks that source and target dimensions are within acceptable ratios
/// to avoid excessive resampling artifacts.
pub fn validate_registration_compatibility(
    source_dims: (usize, usize, usize),
    target_dims: (usize, usize, usize),
) -> KwaversResult<()> {
    const MAX_RATIO: f64 = 10.0;

    for dim in 0..3 {
        let source = [source_dims.0, source_dims.1, source_dims.2][dim] as f64;
        let target = [target_dims.0, target_dims.1, target_dims.2][dim] as f64;
        let ratio = (source / target).max(target / source);

        if ratio > MAX_RATIO {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Incompatible dimensions for registration: ratio {} exceeds maximum {}",
                        ratio, MAX_RATIO
                    ),
                },
            ));
        }
    }

    Ok(())
}
