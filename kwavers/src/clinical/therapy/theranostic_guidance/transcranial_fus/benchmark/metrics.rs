use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::types::PressureFieldMetrics;

pub fn evaluate_pressure_field(
    reference_pressure_pa: &Array3<f32>,
    candidate_pressure_pa: &Array3<f32>,
    spacing_m: [f64; 3],
) -> KwaversResult<PressureFieldMetrics> {
    if reference_pressure_pa.dim() != candidate_pressure_pa.dim() {
        return Err(KwaversError::InvalidInput(
            "reference and candidate pressure fields must have identical shapes".to_owned(),
        ));
    }
    if spacing_m
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "spacing_m values must be positive and finite".to_owned(),
        ));
    }
    let (reference_focus_index, reference_peak_pa) = argmax_pressure(reference_pressure_pa)?;
    let (candidate_focus_index, candidate_peak_pa) = argmax_pressure(candidate_pressure_pa)?;
    let reference_norm = reference_peak_pa.max(f64::MIN_POSITIVE);

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (reference, candidate) in reference_pressure_pa
        .iter()
        .zip(candidate_pressure_pa.iter())
    {
        let r = f64::from(*reference) / reference_norm;
        let c = f64::from(*candidate) / reference_norm;
        let diff = c - r;
        numerator += diff * diff;
        denominator += r * r;
    }
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err(KwaversError::InvalidInput(
            "reference pressure field has zero norm".to_owned(),
        ));
    }

    let dx = (candidate_focus_index[0] as f64 - reference_focus_index[0] as f64) * spacing_m[0];
    let dy = (candidate_focus_index[1] as f64 - reference_focus_index[1] as f64) * spacing_m[1];
    let dz = (candidate_focus_index[2] as f64 - reference_focus_index[2] as f64) * spacing_m[2];
    let focal_position_error_m = (dx * dx + dy * dy + dz * dz).sqrt();
    let max_pressure_error_percent =
        (candidate_peak_pa - reference_peak_pa).abs() / reference_peak_pa * 100.0;

    Ok(PressureFieldMetrics {
        relative_l2: (numerator / denominator).sqrt(),
        focal_position_error_m,
        max_pressure_error_percent,
        reference_peak_pa,
        candidate_peak_pa,
        reference_focus_index,
        candidate_focus_index,
    })
}

fn argmax_pressure(pressure_pa: &Array3<f32>) -> KwaversResult<([usize; 3], f64)> {
    let mut best_index = [0_usize; 3];
    let mut best_value = f64::NEG_INFINITY;
    for ((ix, iy, iz), value) in pressure_pa.indexed_iter() {
        let value = f64::from(*value);
        if !value.is_finite() {
            return Err(KwaversError::InvalidInput(
                "pressure field contains non-finite values".to_owned(),
            ));
        }
        if value > best_value {
            best_value = value;
            best_index = [ix, iy, iz];
        }
    }
    if best_value <= 0.0 || !best_value.is_finite() {
        return Err(KwaversError::InvalidInput(
            "pressure field peak must be positive and finite".to_owned(),
        ));
    }
    Ok((best_index, best_value))
}
