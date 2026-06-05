//! Time-integrated stable / inertial cavitation dose and the closed-loop
//! pressure controller that drives clinical harmonic-dose monitoring.
//!
//! A passive-cavitation-dose controller integrates the band-resolved emission
//! power over the sonication and uses the two accumulated doses to steer drive
//! pressure: grow the *stable* cavitation dose (therapeutic, reversible) while
//! keeping the *inertial* cavitation dose below a damage limit
//! (McDannold 2006; O'Reilly & Hynynen 2012; Arvanitis 2012).

/// Cumulative cavitation dose: trapezoidal time-integral of an emission-power
/// series.
///
/// ```text
///   D[m] = Σ_{i=1..m} ½·(P[i−1] + P[i])·Δt        [emission-power·s]
/// ```
/// `power_arr[k]` is the band emission power measured in the k-th monitoring
/// window (e.g. the stable emission `sub + ultra`, or the broadband emission)
/// and `dt_s` is the window duration. The returned array is the running dose,
/// the same length as `power_arr`, with `D[0] = 0` (no elapsed time yet).
///
/// Negative power samples are clamped to 0 — emission energy cannot reduce an
/// accumulated dose.
///
/// # Reference
/// O'Reilly M.A. & Hynynen K. (2012) *Radiology* 263, 96 (real-time dose control).
#[must_use]
pub fn cumulative_cavitation_dose(power_arr: &[f64], dt_s: f64) -> Vec<f64> {
    let n = power_arr.len();
    let mut out = vec![0.0_f64; n];
    if n < 2 || !(dt_s.is_finite() && dt_s > 0.0) {
        return out;
    }
    let mut acc = 0.0_f64;
    let mut prev = power_arr[0].max(0.0);
    for i in 1..n {
        let cur = power_arr[i].max(0.0);
        acc += 0.5 * (prev + cur) * dt_s;
        out[i] = acc;
        prev = cur;
    }
    out
}

/// One step of the InsighTec-style closed-loop cavitation-dose controller.
///
/// The controller adjusts drive pressure to maximise stable cavitation while
/// holding inertial cavitation below a safety limit. Given the most recent
/// monitoring-window emissions:
/// * **inertial emission above `inertial_limit`** → multiplicatively *back off*
///   pressure by `1 − gain` (clamped to `[p_min, p_max]`). Safety dominates.
/// * **otherwise, stable emission below `stable_target`** → *increase* pressure
///   by `1 + gain` to recruit more stable cavitation.
/// * **otherwise** → hold pressure (in the therapeutic window).
///
/// This is the discrete proportional law used for per-burst pressure stepping;
/// `gain` is the fractional step per burst (e.g. 0.05–0.1).
///
/// # Arguments
/// * `current_p_pa`     – drive pressure applied on the just-monitored burst [Pa]
/// * `stable_emission`  – measured sub+ultra-harmonic emission this burst
/// * `inertial_emission`– measured broadband emission this burst
/// * `stable_target`    – stable-emission set-point (recruit up to this level)
/// * `inertial_limit`   – broadband-emission ceiling (never exceed)
/// * `gain`             – fractional pressure step per burst (≥ 0)
/// * `p_min_pa`, `p_max_pa` – drive-pressure clamp [Pa]
///
/// Returns the drive pressure for the next burst [Pa].
///
/// # Reference
/// McDannold N. et al. (2006) *Phys. Med. Biol.* 51, 793.
/// Arvanitis C.D. et al. (2012) *PLoS ONE* 7, e45783.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn cavitation_controller_pressure(
    current_p_pa: f64,
    stable_emission: f64,
    inertial_emission: f64,
    stable_target: f64,
    inertial_limit: f64,
    gain: f64,
    p_min_pa: f64,
    p_max_pa: f64,
) -> f64 {
    let g = gain.max(0.0);
    let next = if inertial_emission > inertial_limit {
        current_p_pa * (1.0 - g) // safety back-off dominates
    } else if stable_emission < stable_target {
        current_p_pa * (1.0 + g) // recruit more stable cavitation
    } else {
        current_p_pa // hold inside the therapeutic window
    };
    next.clamp(p_min_pa, p_max_pa)
}
