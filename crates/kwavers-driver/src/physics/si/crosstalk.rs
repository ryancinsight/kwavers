//! Crosstalk coupling and channel operating margin (signal-to-noise margin) for coupled lines.
//!
//! Crosstalk lives here because the physical phenomenon is inter-trace coupling (the dominant
//! signal-integrity noise source in parallel-routed differential buses, see for example the
//! HV-pulser SPI daisy chain on holohv tiles). [`channel_operating_margin_db`] is the natural
//! neighbour: a coupled-line COM is the amplitude-ratio between the dominant signal-edge
//! swing and the worst-case coupled noise from a neighbour — when that ratio falls below
//! the receiver's threshold, the link fails eye-mask. Putting the two fns side-by-side lets
//! the caller compose a budget check (`crosstalk_coupling` for the worst-case coupled edge
//! step → `channel_operating_margin_db` for the budget margin in dB) without crossing
//! submodule boundaries.
//!
//! The COM convention used here is the IEEE-standard amplitude-ratio `20·log10(V_signal / V_noise)`;
//! it is **not** the Jedec COM full statistical formulation (which folds ISI, crosstalk, and
//! jitter into one figure of merit). For the Jedec full COM, pre-compute a measured signal
//! amplitude and the worst-case summed noise from this slice, then call
//! [`channel_operating_margin_db`] with the pair.

/// Backward (near-end) crosstalk coupling coefficient between two parallel microstrips at centre
/// spacing `s` over height `h` to the reference plane. Empirical `k ≈ 1 / (1 + (s/h)²)`: coupling
/// falls off sharply with spacing/height — the physical basis for the router's crosstalk penalty.
#[must_use]
pub fn crosstalk_coupling(spacing_m: f64, height_m: f64) -> f64 {
    if height_m <= 0.0 {
        return 1.0;
    }
    let r = spacing_m / height_m;
    1.0 / (1.0 + r * r)
}

/// Channel operating margin (dB) — the IEEE-standard amplitude-ratio
/// `20·log10(V_signal / V_noise)`.
///
/// Boundary behaviour:
/// * `V_signal == V_noise` ⇒ `COM = 0 dB` (link at threshold).
/// * `V_signal = 10 · V_noise` ⇒ `COM = 20 dB` (one-decade margin).
/// * `V_signal < V_noise` ⇒ `< 0 dB` (link below threshold; will fail eye-mask).
/// * `V_signal <= 0` or `V_noise <= 0` ⇒ degenerate inputs; returns [`f64::NEG_INFINITY`]
///   so the caller can distinguish a budget-bust from a budget-meet without a `NaN` chain.
#[must_use]
pub fn channel_operating_margin_db(signal_v: f64, noise_v: f64) -> f64 {
    if signal_v <= 0.0 || noise_v <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * (signal_v / noise_v).log10()
}
