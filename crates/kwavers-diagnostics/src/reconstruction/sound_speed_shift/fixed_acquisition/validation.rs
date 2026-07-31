//! Per-frame validation for fixed acquisition plans.

use aequitas::systems::si::quantities::Time;
use kwavers_core::error::{KwaversError, KwaversResult};

pub(super) fn validate_frame_time_shifts(
    time_shifts: &[Time],
    rows_available: usize,
) -> KwaversResult<()> {
    if time_shifts.len() != rows_available {
        return Err(KwaversError::DimensionMismatch(format!(
            "fixed speed-shift plan expected {rows_available} frame time shifts, got {}",
            time_shifts.len()
        )));
    }
    if let Some((idx, value)) = time_shifts
        .iter()
        .enumerate()
        .find(|(_, value)| !value.into_base().is_finite())
    {
        return Err(KwaversError::InvalidInput(format!(
            "fixed speed-shift frame row {idx} has nonfinite time shift {}",
            value.into_base()
        )));
    }
    Ok(())
}

pub(super) fn validate_frame_batch(
    frame_time_shifts: &[&[Time]],
    rows_available: usize,
) -> KwaversResult<()> {
    if frame_time_shifts.is_empty() {
        return Err(KwaversError::InvalidInput(
            "fixed speed-shift batch requires at least one frame".to_owned(),
        ));
    }
    for (frame_index, time_shifts) in frame_time_shifts.iter().enumerate() {
        if time_shifts.len() != rows_available {
            return Err(KwaversError::DimensionMismatch(format!(
                "fixed speed-shift batch frame {frame_index} expected {rows_available} time shifts, got {}",
                time_shifts.len()
            )));
        }
        if let Some((row, value)) = time_shifts
            .iter()
            .enumerate()
            .find(|(_, value)| !value.into_base().is_finite())
        {
            return Err(KwaversError::InvalidInput(format!(
                "fixed speed-shift batch frame {frame_index} row {row} has nonfinite time shift {}",
                value.into_base()
            )));
        }
    }
    Ok(())
}
