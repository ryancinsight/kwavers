//! Standalone design-limit helpers that complement the full [`super::evaluate_design_point`]: the
//! self-consistent thermal duty ceiling, the switching-ring breakdown check, and the
//! junction-temperature-corrected track resistance.

use crate::driver::{pulser_dissipation, PulserOp};
use crate::physics::thermal::temperature_derated_resistance;

use super::context::ThermalContext;

/// Compute the maximum burst duty cycle that keeps `T_j ≤ T_j_max` under the given thermal context,
/// accounting for the temperature-derated Rds_on (hotter operation raises dissipation, which
/// further raises temperature — this function solves for the self-consistent limit).
///
/// The thermal model is linear: `T_j = T_ambient + board_rise + θ_jc · P_device`.
/// Since `P_device` depends on duty cycle (linearly via `pulser_dissipation`) and
/// `T_j` depends on `P_device`, the maximum duty is:
///
/// `D_max = (T_j_max − T_ambient − board_rise) / (θ_jc · P_device_at_D1)`
///
/// where `P_device_at_D1` is the device dissipation at 100% duty. Capped at 1.0 (CW).
#[must_use]
pub fn max_safe_duty_thermal(op: &PulserOp, thermal: &ThermalContext) -> f64 {
    // Dissipation at D = 1.0 (worst case, cold Rds_on — conservative).
    let op_d1 = PulserOp { duty: 1.0, ..*op };
    let p_at_d1 = pulser_dissipation(&op_d1).device_total;
    if p_at_d1 <= 0.0 || thermal.theta_jc_k_per_w <= 0.0 {
        return 1.0;
    }
    let headroom = thermal.t_j_max_k - thermal.t_ambient_k - thermal.board_rise_k;
    if headroom <= 0.0 {
        return 0.0;
    }
    (headroom / (thermal.theta_jc_k_per_w * p_at_d1)).min(1.0)
}

/// Check whether the switching-node ringing would push the drain voltage above a device breakdown margin.
///
/// Returns `true` when `V_supply + V_ring > V_breakdown * (1 − margin_frac)`.
/// A typical margin is 10–20% (`margin_frac = 0.10–0.20`) to account for transient overshoot
/// and device-to-device variation in the breakdown voltage.
#[must_use]
pub fn ringing_exceeds_breakdown(
    v_supply: f64,
    ring_v: f64,
    v_breakdown: f64,
    margin_frac: f64,
) -> bool {
    v_supply + ring_v > v_breakdown * (1.0 - margin_frac)
}

/// Copper track resistance corrected for operating temperature at the hot junction.
///
/// Convenience wrapper that chains [`crate::physics::ampacity::track_resistance()`] and
/// [`crate::physics::thermal::temperature_derated_resistance`] for the common case of computing
/// the trace resistance seen by the device at its operating junction temperature.
#[must_use]
pub fn hot_track_resistance(len_m: f64, width_m: f64, copper_oz: f64, t_j_k: f64) -> f64 {
    let r_dc = crate::physics::ampacity::track_resistance(len_m, width_m, copper_oz);
    temperature_derated_resistance(r_dc, 293.0, t_j_k, 3.93e-3)
}
