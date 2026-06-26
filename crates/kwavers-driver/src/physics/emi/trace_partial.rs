//! Partial self-inductance (Grover/IPC) for a straight rectangular trace — the source of
//! `L·dI/dt` overshoot on the HV switching node: ~6–10 nH/cm.

use super::MU0;

/// Partial self-inductance (nH) of a straight rectangular trace (Grover/IPC):
/// `L = (μ₀/2π)·l·[ ln(2l/(w+t)) + 0.5 + 0.2235·(w+t)/l ]`. ~6–10 nH/cm — the source of `L·dI/dt`
/// overshoot on the HV switching node.
#[must_use]
pub fn trace_partial_inductance_nh(len_m: f64, width_m: f64, thickness_m: f64) -> f64 {
    if len_m <= 0.0 {
        return 0.0;
    }
    let wt = width_m + thickness_m;
    let l = (MU0 / (2.0 * std::f64::consts::PI))
        * len_m
        * ((2.0 * len_m / wt).ln() + 0.5 + 0.2235 * wt / len_m);
    l * 1.0e9
}
