//! Matching-network, reactive-drive, ringdown and switching-node physics: the reactive energy a
//! tuning inductor would recover, the damping-resistor ↔ ringdown-Q relationship, the
//! electrical→acoustic efficiency, and the LC turn-off ringing overshoot.

/// Reactive power (VA, ~W of loss in a direct drive) cycled into a capacitive load each second:
/// `D·f·C·V²`. This is the energy a matching inductor would recover.
#[must_use]
pub fn reactive_drive_power_w(c_f: f64, v: f64, freq_hz: f64, duty: f64) -> f64 {
    duty * freq_hz * c_f * v * v
}

/// Series/parallel **tuning inductor** (H) that resonates out a transducer clamped capacitance `c0_f`
/// at `f0_hz`: `L = 1/((2π f₀)² C₀)`. Presenting a real load to the source recovers the reactive
/// `C₀·V²·f` energy a direct pulser would dissipate — the core of an output matching network.
#[must_use]
pub fn tuning_inductor_h(c0_f: f64, f0_hz: f64) -> f64 {
    if c0_f <= 0.0 || f0_hz <= 0.0 {
        return f64::INFINITY;
    }
    let w = 2.0 * std::f64::consts::PI * f0_hz;
    1.0 / (w * w * c0_f)
}

/// Electrical quality factor of a series-R / C₀ load at `f0`: `Q = 1/(2π f₀ C₀ R)`. High `Q` = lightly
/// damped (long ringdown, narrow band); low `Q` = heavily damped (short pulse, broadband imaging).
#[must_use]
pub fn load_quality_factor(c0_f: f64, f0_hz: f64, r_series_ohm: f64) -> f64 {
    let xc = 1.0 / (2.0 * std::f64::consts::PI * f0_hz * c0_f);
    if r_series_ohm <= 0.0 {
        return f64::INFINITY;
    }
    xc / r_series_ohm
}

/// Series **damping resistor** (Ω) giving a target ringdown quality factor `q_target` for a `c0_f`
/// load at `f0_hz`: `R = 1/(2π f₀ C₀ Q)`. Short imaging pulses want `Q ≈ 0.5–1`; the present design's
/// 56 Ω is a damping choice, not an impedance match.
#[must_use]
pub fn damping_resistor_ohm(c0_f: f64, f0_hz: f64, q_target: f64) -> f64 {
    if q_target <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (2.0 * std::f64::consts::PI * f0_hz * c0_f * q_target)
}

/// Ringdown duration in carrier cycles for a resonant load of quality factor `q`: the envelope decays
/// to `1/e` in ≈ `Q/π` cycles. Imaging axial resolution improves as this shortens.
#[must_use]
pub fn ringdown_cycles(q: f64) -> f64 {
    q / std::f64::consts::PI
}

/// Electrical→acoustic efficiency `η = P_acoustic / (P_acoustic + P_loss)`. `P_loss` is the total
/// dissipated power (device + resistors); `P_acoustic` is the real power delivered to the radiation
/// resistance. Returns `0` when nothing is delivered.
#[must_use]
pub fn driver_efficiency(p_acoustic_w: f64, p_loss_w: f64) -> f64 {
    let denom = p_acoustic_w + p_loss_w;
    if denom <= 0.0 {
        0.0
    } else {
        p_acoustic_w / denom
    }
}

/// Switching-node LC ringing peak voltage (V) above the supply rail.
///
/// When the HV pulser switches off, the commutation loop forms a series LC tank
/// (`L_loop` — from the commutation-loop partial inductance, `C_sw` — the device's
/// output capacitance + node capacitance). The energy stored in the inductor at the
/// moment of turn-off rings as:
///
/// `V_ring = I_peak · √(L_loop / C_sw)` (the characteristic impedance of the LC tank).
///
/// This overshoot adds directly to the supply rail, so the device must be rated to at
/// least `V_supply + V_ring`. For the HV7355 at 150 V supply with a 10 nH loop and
/// 25 pF node capacitance, `V_ring = 1.5 · √(10e-9/25e-12) ≈ 30 V`, pushing the
/// peak to 180 V — 20% above the nominal 150 V rail.
///
/// - `i_peak_a` — peak current at switch-off (A); for cap loads use
///   [`crate::physics::emi::capacitive_drive_current_a`]
/// - `loop_nh` — commutation-loop partial inductance (nH); from
///   [`crate::physics::emi::loop_inductance_nh`] or [`crate::physics::emi::trace_partial_inductance_nh`]
/// - `c_sw_f` — total switching-node capacitance (F) = device Coss + trace + load
#[must_use]
pub fn switching_node_ringing_v(i_peak_a: f64, loop_nh: f64, c_sw_f: f64) -> f64 {
    if loop_nh <= 0.0 || c_sw_f <= 0.0 {
        return 0.0;
    }
    i_peak_a * (loop_nh * 1.0e-9 / c_sw_f).sqrt()
}
