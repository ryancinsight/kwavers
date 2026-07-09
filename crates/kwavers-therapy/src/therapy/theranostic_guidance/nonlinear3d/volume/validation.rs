//! Input validation for nonlinear 3-D volume preparation.

use leto::Array3;

use kwavers_core::error::{KwaversError, KwaversResult};

pub(super) fn validate_inputs(
    ct_hu: &Array3<f64>,
    label_volume: Option<&Array3<i16>>,
    spacing_mm: [f64; 3],
) -> KwaversResult<()> {
    if ct_hu.is_empty() {
        return Err(KwaversError::InvalidInput(
            "nonlinear 3-D CT volume is empty".to_owned(),
        ));
    }
    if let Some(labels) = label_volume {
        if labels.dim() != ct_hu.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "CT shape {:?} does not match segmentation shape {:?}",
                ct_hu.dim(),
                labels.dim()
            )));
        }
    }
    if spacing_mm
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "nonlinear 3-D CT spacing must be positive and finite".to_owned(),
        ));
    }
    Ok(())
}
