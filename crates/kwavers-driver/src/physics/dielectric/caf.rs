//! Relative conductive-anodic-filament (CAF) time-to-failure (Rudra/IPC-TR-476 form).
//!
//! CAF is the HV+humidity failure mode where a copper filament grows along the glass-weave /
//! resin interface between two conductors. The time-to-failure scales as `spacing² / voltage`
//! at fixed humidity/temperature:
//!
//! ```text
//! TTF ∝ spacing² / voltage
//! ```
//!
//! This module exposes the **relative** form
//! `TTF/TTF_ref = (s/s_ref)² · (v_ref/v)` so a contributor can compare a candidate drill
//! margin against a reference design (>1 is safer) without picking absolute humidity / field
//! strengths.

/// Relative conductive-anodic-filament (CAF) time-to-failure between two conductors at
/// `spacing_mm` and `voltage_v`, versus a reference. CAF (an HV+humidity failure where a
/// copper filament grows along the glass-weave) follows `TTF ∝ spacing² / voltage`
/// (Rudra/IPC-TR-476 form). Returns `TTF/TTF_ref` (>1 is safer).
#[must_use]
pub fn caf_ttf_relative(s_ref_mm: f64, v_ref: f64, spacing_mm: f64, voltage_v: f64) -> f64 {
    if spacing_mm <= 0.0 || voltage_v <= 0.0 || s_ref_mm <= 0.0 || v_ref <= 0.0 {
        return f64::INFINITY;
    }
    (spacing_mm / s_ref_mm).powi(2) * (v_ref / voltage_v)
}
