//! Per-frame validation for fixed acquisition plans.

use crate::core::error::{KwaversError, KwaversResult};

pub(super) fn validate_frame_time_shifts(
    time_shifts_s: &[f64],
    rows_available: usize,
) -> KwaversResult<()> {
    if time_shifts_s.len() != rows_available {
        return Err(KwaversError::DimensionMismatch(format!(
            "fixed speed-shift plan expected {rows_available} frame time shifts, got {}",
            time_shifts_s.len()
        )));
    }
    if let Some((idx, value)) = time_shifts_s
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KwaversError::InvalidInput(format!(
            "fixed speed-shift frame row {idx} has nonfinite time shift {value}"
        )));
    }
    Ok(())
}

pub(super) fn validate_frame_batch(
    frame_time_shifts_s: &[&[f64]],
    rows_available: usize,
) -> KwaversResult<()> {
    if frame_time_shifts_s.is_empty() {
        return Err(KwaversError::InvalidInput(
            "fixed speed-shift batch requires at least one frame".to_owned(),
        ));
    }
    for (frame_index, time_shifts_s) in frame_time_shifts_s.iter().enumerate() {
        if time_shifts_s.len() != rows_available {
            return Err(KwaversError::DimensionMismatch(format!(
                "fixed speed-shift batch frame {frame_index} expected {rows_available} time shifts, got {}",
                time_shifts_s.len()
            )));
        }
        if let Some((row, value)) = time_shifts_s
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KwaversError::InvalidInput(format!(
                "fixed speed-shift batch frame {frame_index} row {row} has nonfinite time shift {value}"
            )));
        }
    }
    Ok(())
}
