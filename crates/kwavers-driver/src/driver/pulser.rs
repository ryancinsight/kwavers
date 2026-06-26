//! The core class-D HV-pulser loss model: a [`PulserOp`] operating point yields the duty-weighted
//! per-channel [`PulserDissipation`] breakdown, splitting the dynamic ½CV² loss between the device
//! on-resistance and the series damping resistor so each is a correctly-weighted thermal source.

/// Per-channel pulser dissipation breakdown (W), duty-cycle weighted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulserDissipation {
    /// Total dynamic (charge/discharge) loss `D·f·C·V²`, before the device/resistor split.
    pub dynamic_total: f64,
    /// Dynamic loss dissipated **in the device** (its on-resistance share of the series path).
    pub dynamic_device: f64,
    /// Dynamic loss dissipated **in the series (damping) resistor**.
    pub dynamic_series_r: f64,
    /// Gate-drive loss `D·Q_g·V_drv·f` (device-side).
    pub gate: f64,
    /// Reverse-recovery loss `D·Q_rr·V·f` (device-side clamp/body diode).
    pub recovery: f64,
    /// Total heat in the **device** = dynamic_device + gate + recovery.
    pub device_total: f64,
}

/// Inputs describing one channel's pulser + load operating point.
#[derive(Debug, Clone, Copy)]
pub struct PulserOp {
    /// Drive (carrier) frequency, Hz.
    pub drive_hz: f64,
    /// Duty cycle `0..=1` — burst-on fraction (1.0 for CW).
    pub duty: f64,
    /// Load capacitance charged each cycle (transducer clamped C₀ + output cap), F.
    pub c_load_f: f64,
    /// Pulse voltage swing, V.
    pub v_pp: f64,
    /// Device output on-resistance (charge path), Ω.
    pub r_on_ohm: f64,
    /// Series damping resistor, Ω.
    pub r_series_ohm: f64,
    /// Gate charge, C.
    pub q_g_c: f64,
    /// Gate-drive voltage, V.
    pub v_gate: f64,
    /// Clamp/body-diode reverse-recovery charge, C.
    pub q_rr_c: f64,
}

/// Derive the per-channel pulser dissipation from the operating point.
#[must_use]
pub fn pulser_dissipation(op: &PulserOp) -> PulserDissipation {
    let dynamic_total = op.duty * op.drive_hz * op.c_load_f * op.v_pp * op.v_pp;
    let r_sum = op.r_on_ohm + op.r_series_ohm;
    // The charge current flows through both resistances in series, so the ½CV² per-edge dissipation
    // splits by resistance ratio. If neither is specified, charge all dynamic loss to the device.
    let device_frac = if r_sum > 0.0 {
        op.r_on_ohm / r_sum
    } else {
        1.0
    };
    let dynamic_device = dynamic_total * device_frac;
    let dynamic_series_r = dynamic_total - dynamic_device;
    let gate = op.duty * op.q_g_c * op.v_gate * op.drive_hz;
    let recovery = op.duty * op.q_rr_c * op.v_pp * op.drive_hz;
    PulserDissipation {
        dynamic_total,
        dynamic_device,
        dynamic_series_r,
        gate,
        recovery,
        device_total: dynamic_device + gate + recovery,
    }
}
