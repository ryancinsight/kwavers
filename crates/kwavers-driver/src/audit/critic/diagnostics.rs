//! EMC/EMI analysis structs and audit functions.
//!
//! Provides charge-recycling efficiency and pulse-skipping interference audits
//! used by the board-level DFM critic.

use crate::board::Board;
use crate::place::FootprintDef;

// ──────────────────────────────────────────────────────────────────────────────
// Charge-recycling efficiency audit
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a charge-recycling efficiency audit.
#[derive(Debug, Clone)]
pub struct ChargeRecyclingReport {
    /// Estimated fraction of dynamic loss recoverable via charge-recycling.
    pub recovery_fraction: f64,
    /// Dynamic loss without recycling (W) — in a 3-level driver.
    pub loss_no_cr_w: f64,
    /// Dynamic loss with N-level + charge-recycling (W).
    pub loss_with_cr_w: f64,
    /// Power saved by charge-recycling (W).
    pub power_saved_w: f64,
    /// Whether charge-recycling is applicable (N-level driver with CR capability).
    pub charge_recycling_applicable: bool,
}

/// Audit the board's driver topology for charge-recycling efficiency.
///
/// Examines pulser IC footprints and estimates the power savings if
/// N-level + charge-recycling drivers are used vs conventional 3-level.
///
/// # Errors
///
/// This function is infallible; it returns a report with zero savings when no
/// N-level ICs are found or when the input power parameters are zero.
#[must_use]
#[allow(clippy::too_many_arguments)] // physics kernel: each argument is irreducible
pub fn charge_recycling_efficiency_audit(
    _board: &Board,
    lib: &[FootprintDef],
    n_levels: usize,
    cr_efficiency: f64,
    c_load_f: f64,
    v_pp: f64,
    freq_hz: f64,
    duty: f64,
) -> ChargeRecyclingReport {
    use crate::five_level::{nlevel_dynamic_loss_w, nlevel_power_saving_w};

    let p_3lv = duty * freq_hz * c_load_f * v_pp * v_pp;
    let p_cr = nlevel_dynamic_loss_w(p_3lv, n_levels, cr_efficiency);
    let saved = nlevel_power_saving_w(p_3lv, n_levels, cr_efficiency);

    let has_nlevel_ic = lib.iter().any(|fp| {
        let name = fp.name.to_uppercase();
        name.contains("STHVUP32") || name.contains("MAX14815") || name.contains("MD1715")
    });

    ChargeRecyclingReport {
        recovery_fraction: if p_3lv > 0.0 { saved / p_3lv } else { 0.0 },
        loss_no_cr_w: p_3lv,
        loss_with_cr_w: p_cr,
        power_saved_w: saved,
        charge_recycling_applicable: has_nlevel_ic && n_levels >= 5,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pulse-skipping pattern interference audit
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a pulse-skipping interference audit.
#[derive(Debug, Clone)]
pub struct PulseSkipInterferenceReport {
    /// Pressure error fraction from pulse skipping.
    pub pressure_error_frac: f64,
    /// Grating-lobe level from skip pattern periodicity (0..1).
    pub grating_lobe_level: f64,
    /// Worst-case tonal spur (dB below carrier).
    pub tonal_spur_dbc: f64,
    /// Recommended maximum skip fraction.
    pub max_skip_fraction: f64,
    /// Whether interference is acceptable.
    pub tolerable: bool,
}

/// Audit the board for interference from pulse-skipping patterns.
///
/// A random (Bernoulli) skip pattern distributes skip events uniformly
/// across channels and time, so grating-lobe and tonal spur artifacts
/// are negligible. This function checks those assumptions.
///
/// # Errors
///
/// This function is infallible; `tolerable` in the returned report indicates
/// whether the computed interference level is within acceptable bounds.
#[must_use]
pub fn pulse_skip_interference_audit(
    n_channels: usize,
    pitch_m: f64,
    lambda_m: f64,
    skip_fraction: f64,
    steer_deg: f64,
    _speed_m_s: f64,
    f_drive_hz: f64,
) -> PulseSkipInterferenceReport {
    use crate::pulse_skip::{
        max_skip_spur_dbc, rms_pressure_error_fraction, skip_induced_grating_lobe,
    };

    let pressure_err = rms_pressure_error_fraction(n_channels, skip_fraction);
    let grating_lobe =
        skip_induced_grating_lobe(n_channels, pitch_m, lambda_m, skip_fraction, steer_deg);
    let tonal_spur = max_skip_spur_dbc(skip_fraction, n_channels, f_drive_hz, f_drive_hz);

    let max_skip = if pressure_err > 0.05 {
        (0.05 * 0.05 * n_channels as f64).min(0.9)
    } else {
        skip_fraction
    };

    PulseSkipInterferenceReport {
        pressure_error_frac: pressure_err,
        grating_lobe_level: grating_lobe,
        tonal_spur_dbc: tonal_spur,
        max_skip_fraction: max_skip,
        tolerable: pressure_err < 0.05 && grating_lobe < 0.01,
    }
}
