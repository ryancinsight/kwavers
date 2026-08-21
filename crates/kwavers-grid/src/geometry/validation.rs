use super::GeometryError;

pub(super) fn validate_measure(kind: &'static str, value: f64) -> Result<(), GeometryError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(GeometryError::InvalidMeasure { kind, value })
    }
}
