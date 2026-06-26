//! The frequency-sweep driver-loss optimiser: evaluate the pulser loss model, efficiency, thermal
//! duty headroom and matching-tank parameters across an operating-frequency range and pick the
//! lowest-loss point.

use super::pulser::{pulser_dissipation, PulserOp};
use super::rating::max_safe_duty;
use super::reactive::{driver_efficiency, load_quality_factor, tuning_inductor_h};
use super::DEFAULT_THETA_JC_K_PER_W;

/// A single frequency point in the driver-loss sweep.
#[derive(Debug, Clone, Copy)]
pub struct FreqSweepPoint {
    /// Operating frequency (Hz).
    pub freq_hz: f64,
    /// Dynamic switching loss at this frequency (W).
    pub dynamic_loss_w: f64,
    /// Gate-charge loss at this frequency (W).
    pub gate_loss_w: f64,
    /// Reverse-recovery loss at this frequency (W).
    pub recovery_loss_w: f64,
    /// Total per-device loss: dynamic + gate + recovery (W).
    pub total_device_w: f64,
    /// Power conversion efficiency (0–1).
    pub efficiency: f64,
    /// Maximum safe duty cycle (0–1) at this frequency given thermal limits.
    pub max_safe_duty: f64,
    /// Series tuning inductance that resonates the load capacitance at this frequency (µH).
    pub tuning_l_uh: f64,
    /// Loaded Q factor of the resonant tank at this frequency.
    pub load_q: f64,
}

/// Sweep driver loss over a frequency range, returning a point per frequency step.
/// Useful for comparing transducer-driver matching across operating frequencies.
#[must_use]
#[allow(clippy::too_many_arguments)] // physics kernel: each argument is an irreducible input.
pub fn sweep_driver_loss(
    f_start_hz: f64,
    f_end_hz: f64,
    n_points: usize,
    c_load_f: f64,
    v_pp: f64,
    r_on_ohm: f64,
    r_series_ohm: f64,
    q_g_c: f64,
    v_gate: f64,
    q_rr_c: f64,
    p_acoustic_w: f64,
    thermal_limit_k: f64,
    duty: f64,
) -> Vec<FreqSweepPoint> {
    if n_points < 2 {
        return Vec::new();
    }
    let step = (f_end_hz - f_start_hz) / (n_points - 1) as f64;
    (0..n_points)
        .map(|i| {
            let freq = f_start_hz + i as f64 * step;
            let op = PulserOp {
                drive_hz: freq,
                duty,
                c_load_f,
                v_pp,
                r_on_ohm,
                r_series_ohm,
                q_g_c,
                v_gate,
                q_rr_c,
            };
            let d = pulser_dissipation(&op);
            let eta = driver_efficiency(p_acoustic_w, d.device_total + d.dynamic_series_r);
            let max_d = max_safe_duty(
                duty,
                d.device_total * DEFAULT_THETA_JC_K_PER_W,
                thermal_limit_k,
            );
            let l_h = tuning_inductor_h(c_load_f, freq) * 1e6;
            let q = load_quality_factor(c_load_f, freq, r_series_ohm);
            FreqSweepPoint {
                freq_hz: freq,
                dynamic_loss_w: d.dynamic_total,
                gate_loss_w: d.gate,
                recovery_loss_w: d.recovery,
                total_device_w: d.device_total,
                efficiency: eta,
                max_safe_duty: max_d,
                tuning_l_uh: l_h,
                load_q: q,
            }
        })
        .collect()
}

/// Find the frequency that minimises total device loss (within the swept range).
#[must_use]
pub fn find_best_freq(sweep: &[FreqSweepPoint]) -> Option<&FreqSweepPoint> {
    sweep
        .iter()
        .min_by(|a, b| a.total_device_w.total_cmp(&b.total_device_w))
}
