use kwavers_core::error::{KwaversError, KwaversResult};

/// Validate that image dimensions are compatible for registration
///
/// Checks that source and target dimensions are within acceptable ratios
/// to avoid excessive resampling artifacts.
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
///
pub fn validate_registration_compatibility(
    source_dims: [usize; 3],
    target_dims: [usize; 3],
) -> KwaversResult<()> {
    const MAX_RATIO: f64 = 10.0;

    for dim in 0..3 {
        let source = source_dims[dim] as f64;
        let target = target_dims[dim] as f64;
        let ratio = (source / target).max(target / source);

        if ratio > MAX_RATIO {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::ConstraintViolation {
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
