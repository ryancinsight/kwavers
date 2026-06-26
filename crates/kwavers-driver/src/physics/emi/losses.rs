//! Switching / gate-drive / reverse-recovery losses — three dissipation channels on a
//! class-D driver IC node.

/// Dynamic switching loss (W) of a class-D node: `P = f·C·V²` (the charge/discharge of the
/// output capacitance each cycle). This is the HV7355's dominant dissipation and the thermal
/// source term.
#[must_use]
pub fn switching_loss_w(freq_hz: f64, c_f: f64, voltage: f64) -> f64 {
    freq_hz * c_f * voltage * voltage
}

/// Gate-drive power (W): `P = Q_g·V_drive·f` — the charge cycled into the power-device gate each
/// period by the MD1822-class gate driver.
#[must_use]
pub fn gate_drive_power_w(qg_c: f64, v_drive: f64, freq_hz: f64) -> f64 {
    qg_c * v_drive * freq_hz
}

/// Reverse-recovery loss (W) in a clamp/body diode: `P = Q_rr·V·f` — the D1–D4 Schottky clamps in
/// the paper exist to keep this (and the ringing) bounded.
#[must_use]
pub fn reverse_recovery_loss_w(qrr_c: f64, voltage: f64, freq_hz: f64) -> f64 {
    qrr_c * voltage * freq_hz
}
