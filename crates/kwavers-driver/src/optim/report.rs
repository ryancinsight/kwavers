//! The [`DesignReport`] — the structured aggregate of every figure of merit
//! [`super::evaluate_design_point`] computes, grouped by physics domain (electrical / thermal /
//! acoustic / PDN / EMI) so a caller can read off the binding constraint directly.

/// Full joint design report.
#[derive(Debug, Clone)]
pub struct DesignReport {
    // ── Electrical ─────────────────────────────────────────────────────────────────────────────
    /// Total device power dissipation (W) at 25 °C (cold model).
    pub p_device_cold_w: f64,
    /// Electrical→acoustic efficiency at 25 °C.
    pub efficiency_cold: f64,
    /// Switching-node peak voltage ringing above the supply rail (V).
    pub switching_ring_v: f64,

    // ── Thermal ────────────────────────────────────────────────────────────────────────────────
    /// Junction temperature (K) at the cold-model dissipation.
    pub t_j_k: f64,
    /// Whether the junction temperature is within the device's rated maximum.
    pub thermal_pass: bool,
    /// Thermally-derated efficiency at the computed junction temperature.
    pub efficiency_derated: f64,
    /// Thermally-derated total device dissipation (W) — accounts for rising Rds_on.
    pub p_device_derated_w: f64,

    // ── Acoustic ───────────────────────────────────────────────────────────────────────────────
    /// Coherent pressure gain at the focus (N × element pressure amplitude).
    pub focal_pressure_gain: f64,
    /// Acoustic intensity at the focus (W/m²) for 1 Pa source (scales with pressure²).
    pub acoustic_intensity_w_m2_per_pa2: f64,
    /// Normalised nonlinear shock parameter σ at the focal depth.
    pub nonlinear_sigma: f64,
    /// Array factor at the steer angle (should be ≈ 1.0 for a properly focused array).
    pub array_factor_at_steer: f64,
    /// Broadside array factor (should be < 0.5 when steered off-axis).
    pub array_factor_broadside: f64,

    // ── PDN ────────────────────────────────────────────────────────────────────────────────────
    /// PDN impedance |Z(f)| at the operating frequency (Ω).
    pub pdn_z_at_freq_ohm: f64,
    /// PDN target impedance (Ω): `V_ripple / I_transient`.
    pub pdn_z_target_ohm: f64,
    /// Whether the PDN impedance is below the target at the operating frequency.
    pub pdn_pass: bool,
    /// Anti-resonance frequency between the bulk and local cap stages (Hz).
    pub anti_resonance_hz: f64,

    // ── EMI ────────────────────────────────────────────────────────────────────────────────────
    /// Estimated radiated emission at the test distance (dBµV/m) at the fundamental frequency.
    pub radiated_emi_dbuv_m: f64,
    /// Whether the estimated emission is within the regulatory limit.
    pub emi_pass: bool,
}
