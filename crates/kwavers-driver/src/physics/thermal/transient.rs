//! Lumped + transient thermal functions: time-constant, exponential rise, junction temperature,
//! and temperature-derated copper resistance.

/// Lumped first-order thermal time constant (s): `τ = R_th · C_th`, with `C_th = m·c_p`.
#[must_use]
pub fn thermal_time_constant_s(r_th_k_per_w: f64, heat_capacity_j_per_k: f64) -> f64 {
    r_th_k_per_w * heat_capacity_j_per_k
}

/// Transient temperature rise (K) at `time_s` toward `steady_k` with time constant `tau_s`:
/// `ΔT(t) = ΔT_∞·(1 − e^{−t/τ})`. The paper observes steady state by its 18 s test window.
#[must_use]
pub fn transient_rise_k(steady_k: f64, tau_s: f64, time_s: f64) -> f64 {
    if tau_s <= 0.0 {
        return steady_k;
    }
    steady_k * (1.0 - (-time_s / tau_s).exp())
}

/// Total junction temperature (K) of a device from ambient, board heat rise, and package thermal resistance.
///
/// The complete thermal chain from power supply to silicon:
/// `T_j = T_ambient + ΔT_board + θ_jc · P_device`
///
/// where:
/// - `t_ambient_k` — ambient temperature (K), typically 298–308 K (25–35 °C)
/// - `board_rise_k` — steady-state board temperature rise at the component footprint (K),
///   from [`super::solve_board`] or [`super::solve_electrothermal`]
/// - `theta_jc_k_per_w` — junction-to-case thermal resistance (K/W) from the datasheet
/// - `p_device_w` — total device power dissipation (W)
///
/// The result drives derating: if `T_j > T_j_max` the device is outside its safe operating area.
/// For the HV7355K6-G, datasheet θ_jc ≈ 40 K/W and `T_j_max = 150 °C` (423 K).
#[must_use]
pub fn junction_temperature_k(
    t_ambient_k: f64,
    board_rise_k: f64,
    theta_jc_k_per_w: f64,
    p_device_w: f64,
) -> f64 {
    t_ambient_k + board_rise_k + theta_jc_k_per_w * p_device_w
}

/// Copper conductor resistance (Ω) corrected for operating temperature.
///
/// Copper resistivity has a positive temperature coefficient: `ρ(T) = ρ₀·(1 + α·ΔT)` where
/// `α_Cu ≈ 3.93e-3 K⁻¹` (IEC 60228). A hot board track carries more resistance than the DC
/// value calculated at 20 °C, increasing Joule heating and IR drop. For the HV pulser's
/// 1.5 A peak with 10 Ω tracks this amounts to ~15% resistance increase at 60 °C junction rise.
///
/// - `r_dc_ohm` — DC resistance at reference temperature (Ω), from [`crate::physics::ampacity::track_resistance()`]
/// - `t_ref_k` — reference temperature for `r_dc_ohm` (typically 293 K / 20 °C)
/// - `t_op_k` — operating temperature (K) — use `junction_temperature_k` for the device or the
///   board thermal field peak for traces
/// - `alpha_k` — temperature coefficient of resistance (K⁻¹); default copper = `3.93e-3 K⁻¹`
#[must_use]
pub fn temperature_derated_resistance(
    r_dc_ohm: f64,
    t_ref_k: f64,
    t_op_k: f64,
    alpha_k: f64,
) -> f64 {
    r_dc_ohm * (1.0 + alpha_k * (t_op_k - t_ref_k))
}
