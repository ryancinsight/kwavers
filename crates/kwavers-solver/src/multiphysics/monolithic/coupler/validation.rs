use super::MonolithicCoupler;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use leto::Array3;
use std::collections::HashMap;

impl MonolithicCoupler {
    /// Validate the solve contract before flattening fields.
    ///
    /// The monolithic state vector stacks complete field volumes in the x-axis,
    /// so every participating field must match the grid dimensions exactly.
    /// Rejecting invalid contracts here preserves panic freedom in
    /// `flatten_fields` and keeps validation cost proportional to field count.
    pub(super) fn validate_solve_inputs(
        &self,
        fields: &HashMap<UnifiedFieldType, Array3<f64>>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "dt".to_owned(),
                value: dt,
                reason: "must be finite and positive".to_owned(),
            }));
        }

        self.newton_config.validate()?;
        self.physics_coefficients.validate()?;

        if fields.is_empty() {
            return Err(KwaversError::Validation(ValidationError::MissingField {
                field: "monolithic fields".to_owned(),
            }));
        }

        let expected = grid.dimensions();
        for (&field_type, field) in fields {
            if field.shape() != expected {
                return Err(KwaversError::Validation(
                    ValidationError::DimensionMismatch {
                        expected: format!("{expected:?}"),
                        actual: format!("{} {:?}", field_type.name(), field.shape()),
                    },
                ));
            }
        }

        Ok(())
    }
}
