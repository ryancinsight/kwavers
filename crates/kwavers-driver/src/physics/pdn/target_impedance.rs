//! PDN target-impedance sizing kernel.
//!
//! Three free functions that turn a board-level ripple / transient budget into a quantification
//! to compare against. Each takes a small bag of physical inputs and returns the budget figure
//! directly (no internal state, no cross-slice dependency).
//!
//! * [`target_impedance_ohm`] — `Z_target = V_ripple / I_transient`.
//! * [`holdup_capacitance_f`] — `C = I·Δt / ΔV` hold-up cap sizing.
//! * [`max_decoupling_distance_mm`] — derivation of the placement budget from the cap's SRF vs.
//!   the board's connection-loop inductance.

/// PDN target impedance (Ω): `Z_target = V_ripple / I_transient`. The power net's impedance must
/// stay below this across the relevant band for the rail to hold its voltage.
#[must_use]
pub fn target_impedance_ohm(v_ripple: f64, i_transient_a: f64) -> f64 {
    if i_transient_a <= 0.0 {
        return f64::INFINITY;
    }
    v_ripple / i_transient_a
}

/// Bulk hold-up capacitance (F) to bound the rail droop to `dv` while supplying `i` for `dt`:
/// `C = I·Δt / ΔV`.
#[must_use]
pub fn holdup_capacitance_f(i_a: f64, dt_s: f64, dv: f64) -> f64 {
    if dv <= 0.0 {
        return f64::INFINITY;
    }
    i_a * dt_s / dv
}

/// Maximum routed distance (mm) from a decoupling capacitor to the pin it bypasses, such that the
/// loop the placement adds keeps the capacitor's **self-resonant frequency at or above** `f_keep_hz`
/// — the highest frequency the cap must still present a low impedance at.
///
/// The total series inductance is the device/cap mounting `fixed_esl_h` plus the connection loop
/// `loop_nh_per_mm · d`. The SRF stays ≥ `f_keep_hz` while `L ≤ 1/((2π f_keep)² C)`, so the trace
/// budget is `L_budget − fixed_esl`, and the distance budget is that over the per-mm loop inductance.
/// Returns `0.0` if even a zero-length connection already exceeds the budget (the cap can't meet the
/// target), `f64::INFINITY` on degenerate inputs.
#[must_use]
pub fn max_decoupling_distance_mm(
    c_f: f64,
    f_keep_hz: f64,
    loop_nh_per_mm: f64,
    fixed_esl_h: f64,
) -> f64 {
    if c_f <= 0.0 || f_keep_hz <= 0.0 || loop_nh_per_mm <= 0.0 {
        return f64::INFINITY;
    }
    let w = 2.0 * std::f64::consts::PI * f_keep_hz;
    let l_budget = 1.0 / (w * w * c_f); // H
    let trace_budget = l_budget - fixed_esl_h; // H available to the connection loop
    if trace_budget <= 0.0 {
        return 0.0;
    }
    trace_budget / (loop_nh_per_mm * 1.0e-9)
}
