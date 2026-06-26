//! Cross-IC comparison at a single operating point: run a list of `(name, R_on, Q_g)` candidate
//! pulser ICs through the loss model and report each one's device heat, efficiency, thermal duty
//! headroom and matching-tank parameters so the design can pick a part.

use super::pulser::{pulser_dissipation, PulserOp};
use super::rating::max_safe_duty;
use super::reactive::{driver_efficiency, load_quality_factor, tuning_inductor_h};
use super::DEFAULT_THETA_JC_K_PER_W;

/// Cross-component comparison at a single operating point.
#[derive(Debug, Clone)]
pub struct ComponentComparison {
    /// Component part number.
    pub part_number: String,
    /// Total device heat (W): device-side dynamic loss + gate-drive loss + reverse-recovery loss.
    /// This is the full thermal load the IC package must dissipate; it drives the thermal headroom
    /// calculation and the `efficiency` comparison across ICs.
    pub total_w: f64,
    /// Device-side dynamic switching loss (W): the device's R_on fraction of `D·f·C·V²`.
    /// Lower R_on reduces this term; the complement (`dynamic_series_r`) goes into the external
    /// damping resistor and is not affected by which IC is fitted.
    pub device_w: f64,
    /// Power conversion efficiency (0–1).
    pub efficiency: f64,
    /// Maximum safe duty cycle (0–1) at the operating point.
    pub max_duty: f64,
    /// Series tuning inductance that resonates the load capacitance (µH).
    pub tuning_uh: f64,
    /// Loaded Q factor of the resonant tank.
    pub load_q: f64,
}

/// Compare multiple driver ICs at one operating point.
/// Accepts a list of `(name, r_on, q_g)` tuples.
#[must_use]
#[allow(clippy::too_many_arguments)] // physics kernel: each argument is an irreducible input.
pub fn compare_driver_ics_at(
    freq_hz: f64,
    c_load_f: f64,
    v_pp: f64,
    r_series_ohm: f64,
    v_gate: f64,
    q_rr_c: f64,
    p_acoustic_w: f64,
    thermal_limit_k: f64,
    duty: f64,
    ics: &[(&str, f64, f64)],
) -> Vec<ComponentComparison> {
    ics.iter()
        .map(|&(name, r_on, q_g)| {
            let op = PulserOp {
                drive_hz: freq_hz,
                duty,
                c_load_f,
                v_pp,
                r_on_ohm: r_on,
                r_series_ohm,
                q_g_c: q_g,
                v_gate,
                q_rr_c,
            };
            let d = pulser_dissipation(&op);
            // Efficiency for IC comparison uses device-only losses; the external series
            // damping resistor dissipates the same energy regardless of which IC is fitted
            // (it is not part of the IC under evaluation).
            let eta = driver_efficiency(p_acoustic_w, d.device_total);
            let md = max_safe_duty(
                duty,
                d.device_total * DEFAULT_THETA_JC_K_PER_W,
                thermal_limit_k,
            );
            let l_h = tuning_inductor_h(c_load_f, freq_hz) * 1e6;
            let q = load_quality_factor(c_load_f, freq_hz, r_series_ohm);
            ComponentComparison {
                part_number: name.to_string(),
                total_w: d.device_total,
                device_w: d.dynamic_device,
                efficiency: eta,
                max_duty: md,
                tuning_uh: l_h,
                load_q: q,
            }
        })
        .collect()
}
