//! Propagation-delay / length-matching / risetime-degradation kernels for microstrip routing.
//!
//! These three fns cover the **timing** half of signal integrity: a controlled-impedance
//! trace set by [`crate::physics::si::impedance`] rides on a propagation delay set here, and
//! the trace-length budget vs. the timing-resolution budget is what [`within_skew`] enforces.
//! The risetime-loss fn captures the per-metre edge-spread introduced by skin-effect and
//! dielectric loss so a length-matched route can still be checked for **edge-alignment**,
//! not just centre-alignment.
//!
//! All fns accept `f64` lengths / power-of-ten rates; the docstrings say "any consistent
//! length unit". Phase 2 will replace the `w, h, er` parameters with `Meter, Meter, f64` for
//! the dimensioned quantities and return types as `Second_per_meter` / `Second`.
//! **No signature change at Phase 3f** — keeping the API as `f64` preserves every existing
//! call-site and test fixture until the vertical-slice units land.

/// Microstrip propagation delay (s/m): `√εeff / c`.
#[must_use]
pub fn microstrip_delay_s_per_m(w: f64, h: f64, er: f64) -> f64 {
    crate::physics::si::impedance::microstrip_eeff(w, h, er).sqrt() / 2.998e8
}

/// Whether two routed lengths (m) stay within a skew budget set by a timing resolution (s).
#[must_use]
pub fn within_skew(len_a_m: f64, len_b_m: f64, delay_s_per_m: f64, budget_s: f64) -> bool {
    ((len_a_m - len_b_m).abs() * delay_s_per_m) <= budget_s
}

/// Rise-time degradation (ps/m) imposed by skin-effect and dielectric losses on a microstrip.
///
/// A signal edge with rise-time `t_r` propagating along a track of length `L` degrades as
/// `t_r_out ≈ √(t_r_in² + (τ_deg · L)²)`. The per-metre degradation `τ_deg` accounts for:
/// - Skin effect: `τ_skin ≈ 2.2 · δ_skin · √(εeff) / c` — the per-metre delay spread from
///   the frequency-dependent skin depth narrowing the effective conductor.
/// - Dielectric loss: dominates at higher frequencies and adds `τ_diel ≈ tan_δ · √εeff / (c · ln10)`.
///
/// Uses the empirical Magnusson approximation: `τ_deg ≈ sqrt_eeff / (2·c) · (Rs·P/A + tan_δ·f_ghz)`,
/// where `Rs = √(π·f·μ₀·ρ)` is the surface resistance. For 2 MHz on FR4, the skin-effect term
/// is negligible (δ ≈ 46 µm > 1 oz foil), and the dielectric term dominates (~20 ps/m at 2 MHz).
///
/// Returns an empirical estimate for design guidance; not a substitute for a 2.5D field solver.
///
/// - `w`, `h` — width/height (any consistent unit) for Hammerstad Zeff
/// - `er`, `tan_delta` — dielectric constant and loss tangent (FR4: er=4.3, tan_δ≈0.02)
/// - `freq_hz` — operating frequency for skin-effect term
/// - `copper_oz` — copper weight for foil thickness and Rs
#[must_use]
pub fn risetime_degradation_ps_per_m(
    w: f64,
    h: f64,
    er: f64,
    tan_delta: f64,
    freq_hz: f64,
    copper_oz: f64,
) -> f64 {
    use std::f64::consts::PI;
    let eeff = crate::physics::si::impedance::microstrip_eeff(w, h, er);
    let c = 2.998e8;
    // Skin-effect term: Rs·P/(A·c·√eeff) in ps/m.
    // Surface resistance Rs = sqrt(π·f·µ₀·ρ); P/A ~ 2/w (perimeter/area for a rectangular trace).
    let rho_cu = 1.68e-8_f64;
    let mu0 = 1.256_637_062e-6_f64;
    let t_cu = copper_oz * 34.8e-6; // foil thickness in metres
    let rs = (PI * freq_hz * mu0 * rho_cu).sqrt();
    // Use normalised width for perimeter-to-area ratio; w and h in same unit (cancel).
    let pa_ratio = 2.0 / (w * t_cu / h); // rough P/A normalised by h — dimensionally consistent when w,h same unit
    let tau_skin = rs * pa_ratio / (c * eeff.sqrt());
    // Dielectric loss term: tan_δ · √eeff / (c · ln10/10) — semi-empirical.
    let tau_diel = tan_delta * eeff.sqrt() / (c * 10.0_f64.ln() / 10.0);
    (tau_skin + tau_diel) * 1.0e12 // ps/m
}
