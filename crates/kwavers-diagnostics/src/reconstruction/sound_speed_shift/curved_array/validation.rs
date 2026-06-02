//! Validation for curved-array speed-shift acquisition.

use std::collections::HashSet;
use std::f64::consts::TAU;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::geometry::CurvedArray2d;
use super::sampling::CurvedArrayShiftScan;

pub(super) fn validate_array(array: &CurvedArray2d) -> KwaversResult<()> {
    validate_finite("curved-array center x", array.center_m.x_m)?;
    validate_finite("curved-array center y", array.center_m.y_m)?;
    validate_positive("curved-array radius", array.radius_m)?;
    validate_finite("curved-array first angle", array.first_angle_rad)?;
    validate_finite("curved-array angular pitch", array.angular_pitch_rad)?;
    if array.element_count < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "curved-array element_count must be at least 2, got {}",
            array.element_count
        )));
    }
    if array.angular_pitch_rad.abs() <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "curved-array angular pitch must be nonzero".to_owned(),
        ));
    }
    if array.aperture_angle_rad().abs() >= TAU {
        return Err(KwaversError::InvalidInput(format!(
            "curved-array endpoint arc must be open and less than 2*pi rad, got {}",
            array.aperture_angle_rad()
        )));
    }
    Ok(())
}

pub(super) fn validate_scan(scan: &CurvedArrayShiftScan) -> KwaversResult<()> {
    validate_array(&scan.array)?;
    if scan.receiver_offsets.is_empty() {
        return Err(KwaversError::InvalidInput(
            "curved-array shift scan requires at least one receiver offset".to_owned(),
        ));
    }
    let mut seen = HashSet::with_capacity(scan.receiver_offsets.len());
    for offset in &scan.receiver_offsets {
        if *offset == 0 || *offset >= scan.array.element_count {
            return Err(KwaversError::InvalidInput(format!(
                "curved-array receiver offset must lie in 1..element_count, got {offset}"
            )));
        }
        if !seen.insert(*offset) {
            return Err(KwaversError::InvalidInput(format!(
                "curved-array receiver offset {offset} is duplicated"
            )));
        }
    }
    Ok(())
}

fn validate_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

fn validate_positive(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}
