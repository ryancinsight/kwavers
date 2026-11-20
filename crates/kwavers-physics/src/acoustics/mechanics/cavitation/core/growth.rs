//! Bubble growth mechanics
//!
//! Models the growth of cavitation bubbles over time, particularly
//! through rectified diffusion.
//!
//! # Mathematical Specification
//!
//! ## Theorem: Rectified Diffusion (Eller & Flynn 1965; Crum & Hansen 1982)
//! Under acoustic excitation, gas bubbles grow by rectified diffusion:
//! during expansion the bubble surface area increases, enhancing inward
//! gas flux; during compression, reduced surface area limits outward flux.
//! The net effect is a monotonic increase in equilibrium radius once the
//! acoustic pressure exceeds a threshold set by the Laplace pressure.
//!
//! The rate implemented here is the *generalised* Eller–Flynn rate of
//! Crum & Hansen (1982) as restated by Crum & Mao (1996, p. 2902)
//! (see also Zhang 2012, PhD thesis, §3.2, eqs. (3.6)–(3.19)).
//! For a bubble of equilibrium radius `R0` in saturated liquid
//! (`C_i / C_0 = 1`), the long-time (quasi-steady) molar flux is
//!
//! ```text
//!   dn/dt = 4π R0 D C_0 ⟨R/R0⟩ ( 1 − ⟨(R/R0)⁴ (P_in/P_0)⟩ / ⟨(R/R0)⁴⟩ )
//! ```
//!
//! where the cycle averages are, to second order in the fractional
//! oscillation amplitude `ε = |α| (P_ac / P_0)` (Crum & Mao 1996,
//! eqs. (3.17)–(3.19)):
//!
//! ```text
//!   ⟨R/R0⟩                 = 1 + K ε²
//!   ⟨(R/R0)⁴⟩              = 1 + (3 + 4K) ε²
//!   ⟨(R/R0)⁴ (P_in/P_0)⟩  = [1 + (3(κ−1)(3κ−4)/4 + (4−3κ)K) ε²] (1 + 2σ/(R0 P_0))
//! ```
//!
//! The response amplitude `α` is the linear driven-oscillator response
//! (Crum & Mao 1996, eq. (3.13)):
//!
//! ```text
//!   α = −(P_0 / (ρ_l R0²)) / √[(ω² − ω_0²)² + (δ_tot ω_0²)²]
//!   ω_0² = [3κ (P_0 + 2σ/R0) − 2σ/R0] / (ρ_l R0²)
//! ```
//!
//! κ is the thermal polytropic exponent κ = Re Φ/3 (Prosperetti 1977),
//! used in every equation — not the adiabatic γ. At 26.6 kHz the gas is
//! far from adiabatic (κ ≈ 1.05 at 30 µm); substituting γ = 1.4
//! understates ε and the growth term by ~30%.
//!
//! with total damping `δ_tot = δ_vis + δ_th + δ_ac` (Devin 1959;
//! Eller 1970): viscous `δ_vis = 4ωμ / [3κ(P_0+2σ/R0) − 2σ/R0]`,
//! acoustic `δ_ac = ρ_l R0³ω³ / (c_l [3κ(P_0+2σ/R0) − 2σ/R0])`, and
//! thermal `δ_th = Im Φ / Re Φ` (Crum & Mao 1996, eq. (3.10)) from the
//! Devin–Prosperetti thermal function
//! `Φ = 3γ / (1 − 3(γ−1) i X [√(i/X) coth √(i/X) − 1])`,
//! `X = D_{g,p} / (ω R0²)` (Devin 1959, eq. 50; Prosperetti et al. 1988,
//! eq. 41; Zhang 2012, eq. (2.44)), which also sets the effective
//! polytropic exponent `κ = Re Φ / 3` (Prosperetti 1977). The resonance
//! factor is `K = [(3κ+1−β²)/4 + (σ/(2R0P_0))(3κ+1−2/(3κ))] /
//! [1 + (2σ/(R0P_0))(1−1/(3κ))]` with `β² = ρ_l ω²R0² /
//! [3κ(P_0+2σ/R0) − 2σ/R0]` (Crum & Mao 1996, eqs. (3.15)–(3.16)).
//!
//! The bracket splits into a rectified *growth* term and a Laplace
//! *dissolution* term:
//!
//! ```text
//!   1 − ⟨(R/R0)⁴ (P_in/P_0)⟩/⟨(R/R0)⁴⟩ = (growth, ∝ ε²) − 2σ/(R0 P_0)
//! ```
//!
//! so the returned flux is *signed*: positive above the rectified-diffusion
//! threshold, negative (net dissolution) below it. The `(1 + 4σ/(3P_0R0))`
//! factor in the `dR0/dt` form cancels against `dn/dR0` in the molar-flux
//! form, so it does not appear above.
//!
//! ## What this replaces (KW-PHYS-001)
//! The previous implementation returned
//! `4π R D C (P_ac/P_0) √Pe` — linear in the pressure ratio with a
//! `√Pe` factor and no amplitude dependence. That is not the Eller–Flynn
//! rate: the genuine rectified term is second order in the oscillation
//! amplitude (`∝ ε²`, with `ε = |α| P_ac/P_0`), has no `√Pe` factor
//! (the `√Pe`-like term in the theory is the *transient*
//! `R0/√(πDt)`, which tends to 1 at long times and is not part of the
//! steady rate), and is opposed by Laplace dissolution. At Eller's
//! (1969) conditions the old formula gave 515–527 µm/min against ~13
//! µm/min measured — a 40× overprediction — while the genuine theory
//! *underpredicts* the same measurement by >20× (Smith & Wang 2022,
//! §1). The old formula was therefore wrong in both directions at once:
//! ~1000× the genuine rectified flux.
//!
//! ## Benchmark: Eller (1969)
//! Conditions (Smith & Wang 2022, §4.2, from Eller 1969): air-saturated
//! water, 26.6 kHz, `P_ac` = 20 kPa, `D` = 2e-9 m²/s. With the
//! benchmark constants (`C` = 0.26 mol/m³, as in the original
//! cross-check), this implementation gives, in `dR/dt` (isothermal ideal
//! gas, `dn/dR = (4πR²P_0 + (16/3)πRσ)/(R_g T)`):
//!
//! | R0 (µm) | net (µm/min) | growth term only (µm/min) | old formula |
//! |--------:|-------------:|--------------------------:|------------:|
//! | 20 | −1.65 | +0.94 | 516 |
//! | 30 | −0.50 | +0.66 | 524 |
//! | 40 | −0.13 | +0.53 | 528 |
//! | 50 | +0.05 | +0.47 | 530 |
//!
//! Measured (Eller 1969, via Smith & Wang 2022, fig. 3): ~13 µm/min,
//! roughly flat from 30–50 µm. Eller's own theoretical values were
//! ~0.6 µm/min (13/20 — the ">20×" gap of Smith & Wang §1). The
//! growth term here (+0.66 µm/min at 30 µm, 0.47–0.94 µm/min across
//! 20–50 µm) reproduces Eller's theoretical value within ~2×; the net
//! rate is small (|net| ≲ 2 µm/min) and changes sign near this theory's
//! threshold radius R0 ≈ 47 µm (P_T ≈ 27 kPa at 30 µm) versus the
//! measured ~27 µm. The remaining >20× gap to the *measured* rate is a
//! formula error: acoustic microstreaming intensifies growth/dissolution
//! up to twentyfold once surface oscillations ("crispations") appear
//! (Gould 1974, abstract), and the exact leading-order theory with the
//! history integral reproduces the measured rapid early growth (Smith &
//! Wang 2022, JFM 939, A28). Church (1988, JASA 84, 1758–1764) gives a
//! method to account for microstreaming; it is not implemented here
//! (see below). No factor has been fitted to force agreement with
//! Eller: the numbers above are the theory as published.
//!
//! ## What is NOT fixed here
//! - **Microstreaming.** The >20× gap between the genuine theory and
//!   Eller's measurement is attributed in the literature to acoustic
//!   microstreaming and surfactants (Smith & Wang 2022, §1; Gould 1974;
//!   Church 1988, JASA 83, 2210–2217, found agreement with Eller 1969
//!   only below 35 µm). Church (1988, JASA 84, 1758–1764) is the
//!   published microstreaming correction, but its full text was not
//!   accessible for this fix, so no streaming term is implemented.
//!   Adding it requires reading Church (1988, JASA 84) and benchmarking
//!   against Eller (1969) — it must not be fitted to the 13 µm/min.
//! - **Early-time growth.** This is the long-time rate. The early-time
//!   rate is asymptotically larger (Smith & Wang 2022, A28); their
//!   two-ODE model (Rayleigh–Plesset + gas-mass ODE with fading-memory
//!   history integral, eqs. (3.27)–(3.29)) is the more accurate
//!   in-regime alternative, and the three-timescale form (Smith & Wang
//!   2022, JFM 951, A12) is the implementable long-timescale version.
//!   Neither is ported here.
//! - **Supersaturation.** The single `concentration` argument is used as
//!   both the saturation concentration `C_0` and the far-field
//!   concentration `C_i` (saturated liquid, `C_i/C_0 = 1`).
//!   Supersaturated or undersaturated liquids need the split.
//!
//! # References
//! - Eller, A. I., & Flynn, H. G. (1965). "Rectified diffusion during
//!   nonlinear pulsations of cavitation bubbles". JASA, 37(3), 493–503.
//! - Eller, A. I. (1969). "Growth of bubbles by rectified diffusion".
//!   JASA, 46(5), 1246–1250.
//! - Crum, L. A., & Hansen, G. M. (1982). "Generalized equations for
//!   rectified diffusion". JASA, 72(5), 1586–1592.
//! - Crum, L. A., & Mao, Y. (1996). "Acoustic cavitation in
//!   biomedical ultrasound" (rectified-diffusion formulas, p. 2902;
//!   as restated in Zhang 2012, §3.2).
//! - Devin, C. (1959). "Survey of thermal, radiation, and viscous
//!   damping of pulsating air bubbles in water". JASA, 31(12),
//!   1654–1667 (thermal function Φ, eq. 50).
//! - Prosperetti, A. (1977). "Thermal effects and damping mechanisms
//!   in the forced radial oscillations of gas bubbles in liquids".
//!   JASA, 61(1), 17–27 (polytropic exponent from Φ).
//! - Prosperetti, A., Crum, L. A., & Commander, K. W. (1988).
//!   "Nonlinear bubble dynamics". JASA, 83(2), 502–514 (Φ, eq. 41).
//! - Gould, R. K. (1974). "Rectified diffusion in the presence of,
//!   and absence of, acoustic streaming". JASA, 56(6), 1740–1746.
//! - Church, C. C. (1988). "Prediction of rectified diffusion during
//!   nonlinear bubble pulsations at biomedical frequencies". JASA,
//!   83(6), 2210–2217.
//! - Church, C. C. (1988). "A method to account for acoustic
//!   microstreaming when predicting bubble growth rates produced by
//!   rectified diffusion". JASA, 84(5), 1758–1764.
//! - Fyrillas, M. M., & Szeri, A. J. (1994). "Dissolution or growth
//!   of soluble spherical oscillating bubbles". J. Fluid Mech., 277,
//!   381–407.
//! - Smith, W. R., & Wang, Q. (2022). "A theoretical model for the
//!   growth of spherical bubbles by rectified diffusion".
//!   J. Fluid Mech., 939, A28. doi:10.1017/jfm.2022.218.
//! - Smith, W. R., & Wang, Q. (2022). "A tractable mathematical model
//!   for rectified diffusion". J. Fluid Mech., 951, A12.
//!   doi:10.1017/jfm.2022.849.
//! - Zhang, Y. (2012). "Analysis of Radial Oscillations of Gas Bubbles
//!   in Newtonian or Viscoelastic Mediums under Acoustic Excitation".
//!   PhD thesis, University of Warwick, §3.2 (Crum's analysis,
//!   eqs. (3.6)–(3.19)) and §2.2 (thermal effects, eqs. (2.44),
//!   (2.48)–(2.52)).

use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Maximum acoustic-to-ambient pressure ratio for which the Eller–Flynn
/// rectified-diffusion law is in its domain of validity.
///
/// The law assumes small-amplitude oscillation (ε = ΔR/R₀ << 1), a
/// quasi-steady gas field, and an infinite reservoir. Keller–Miksis puts
/// ε = 1 at P_ac/P₀ ≈ 3–4 (R₀ = 2 µm); above that the bubble collapses
/// inertially every cycle and the law's flux is meaningless. Stage 6 of the
/// triplet-pulse prototype applied the law at P_ac/P₀ = 66–146 and produced
/// 986/1904 µm growth artifacts; the Stage 6 audit (Nov 17, 2026) retired
/// those numbers. Use [`rectified_diffusion_rate_checked`] to enforce this
/// bound at the call site.
pub const RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO: f64 = 3.0;

/// Specific-heat ratio of air (γ = c_p / c_v), used for the thermal
/// function Φ and the polytropic exponent.
const GAMMA_AIR: f64 = 1.4;

/// Thermal diffusivity of air at constant pressure, ~20 °C [m²/s]
/// (standard value 2.25e-5; the constant-volume value is γ times larger).
/// Sets the thermal parameter X = D_{g,p} / (ω R₀²) in Φ.
const GAS_THERMAL_DIFFUSIVITY_CP: f64 = 2.25e-5;

/// Liquid (water) density [kg/m³] and dynamic viscosity [Pa·s] at ~20 °C.
const LIQUID_DENSITY: f64 = DENSITY_WATER_NOMINAL;
const LIQUID_VISCOSITY: f64 = 1.002e-3;

/// Sound speed in water [m/s] (acoustic-damping term).
const LIQUID_SOUND_SPEED: f64 = SOUND_SPEED_WATER_SIM;

/// Surface tension of water [N/m].
const SURFACE_TENSION: f64 = SURFACE_TENSION_WATER;

/// Devin–Prosperetti thermal function Φ (Devin 1959, eq. 50;
/// Prosperetti et al. 1988, eq. 41; Zhang 2012, eq. (2.44)):
///
/// ```text
///   Φ = 3γ / (1 − 3(γ−1) i X [√(i/X) coth √(i/X) − 1]),  X = D_{g,p}/(ωR₀²)
/// ```
///
/// Returns `(Re Φ, Im Φ)`. The effective polytropic exponent is
/// `κ = Re Φ / 3` and the thermal damping constant is
/// `δ_th = Im Φ / (Re Φ − 2σ/(R₀ P_in,eq))` (Zhang 2012, eq. (2.49),
/// surface-tension corrected).
///
/// Limits: `X → 0` (adiabatic) gives `Φ → 3γ`, `κ → γ`, `δ_th → 0`;
/// `X → ∞` (isothermal) gives `Φ → 3`, `κ → 1`, `δ_th → 0`.
/// Validated against Devin's simplified forms: `δ_th ≈ 3(γ−1)χ/2`
/// for `χ = √X ≤ 0.08` (eq. (2.50)) and
/// `δ_th ≈ (γ−1)/(15γχ)` for `χ ≥ 0.5` (eq. (2.52)); agreement is
/// ~30% at χ < 0.05 and ~1% at χ ≈ 0.15 and χ ≥ 1, which is immaterial
/// here because α is damping-insensitive off resonance.
fn thermal_phi(x: f64, gamma: f64) -> (f64, f64) {
    if !x.is_finite() || x <= 0.0 {
        // No oscillation (ω = 0) or degenerate input: isothermal limit.
        return (3.0, 0.0);
    }
    // z = √(i/X) = (1+i)/√(2X); write z = (s, s) with s = 1/√(2X).
    let s = 1.0 / (2.0 * x).sqrt();
    // coth z, with z = (s, s):
    //   sinh z = (sinh s cos s, cosh s sin s)
    //   cosh z = (cosh s cos s, sinh s sin s)
    // For Re z > 20, coth z = 1 to <1e-17 (avoids sinh/cosh overflow).
    // For |z| tiny, use coth z ≈ 1/z + z/3 (avoids 0/0).
    let (coth_re, coth_im) = if s > 20.0 {
        (1.0, 0.0)
    } else if s < 1e-3 {
        // 1/z = (1/(2s), −1/(2s)); z/3 = (s/3, s/3).
        (1.0 / (2.0 * s) + s / 3.0, -1.0 / (2.0 * s) + s / 3.0)
    } else {
        let (sh, ch) = (s.sinh(), s.cosh());
        let (sin_s, cos_s) = s.sin_cos();
        let (sinh_re, sinh_im) = (sh * cos_s, ch * sin_s);
        let (cosh_re, cosh_im) = (ch * cos_s, sh * sin_s);
        let denom = sinh_re * sinh_re + sinh_im * sinh_im;
        (
            (cosh_re * sinh_re + cosh_im * sinh_im) / denom,
            (cosh_im * sinh_re - cosh_re * sinh_im) / denom,
        )
    };
    // w = z·coth z − 1, with z = (s, s).
    let w_re = s * (coth_re - coth_im) - 1.0;
    let w_im = s * (coth_re + coth_im);
    // denom = 1 − 3(γ−1)·i·X·w = (1 + 3(γ−1)X w_im, −3(γ−1)X w_re).
    let k = 3.0 * (gamma - 1.0) * x;
    let (d_re, d_im) = (1.0 + k * w_im, -k * w_re);
    // Φ = 3γ / denom.
    let norm = d_re * d_re + d_im * d_im;
    (3.0 * gamma * d_re / norm, -3.0 * gamma * d_im / norm)
}

/// Linear radial-response quantities for a bubble driven at `frequency`
/// (Crum & Mao 1996, eqs. (3.13)–(3.16); Zhang 2012, §3.2).
///
/// Returns `(epsilon, kappa, delta_total, beta_squared, k_factor)`:
/// - `epsilon` = `|α| (P_ac/P_0)`, the fractional oscillation amplitude.
/// - `kappa` = effective polytropic exponent `Re Φ / 3`.
/// - `delta_total` = `δ_vis + δ_th + δ_ac`, the total damping constant.
/// - `beta_squared` = `(ω/ω_0)²`.
/// - `k_factor` = the resonance factor `K` of eq. (3.15).
fn oscillation_response(
    radius: f64,
    ambient_pressure: f64,
    acoustic_pressure: f64,
    frequency: f64,
) -> (f64, f64, f64, f64, f64) {
    let omega = 2.0 * std::f64::consts::PI * frequency;
    if omega <= 0.0 || radius <= 0.0 || ambient_pressure <= 0.0 {
        // No oscillation (or degenerate input): isothermal, undamped,
        // zero amplitude; K reduces to (3·1+1)/4 = 1.
        return (0.0, 1.0, 0.0, 0.0, 1.0);
    }
    // Thermal function: X = D_{g,p} / (ω R₀²); κ = Re Φ / 3.
    let x = GAS_THERMAL_DIFFUSIVITY_CP / (omega * radius * radius);
    let (phi_re, phi_im) = thermal_phi(x, GAMMA_AIR);
    let kappa = phi_re / 3.0;
    // Recompute the stiffness with the thermal κ (Prosperetti 1977):
    // ω_0² = [3κ(P_0 + 2σ/R₀) − 2σ/R₀] / (ρ_l R₀²).
    let stiffness_k = 3.0 * kappa * (ambient_pressure + 2.0 * SURFACE_TENSION / radius)
        - 2.0 * SURFACE_TENSION / radius;
    let omega0_sq = stiffness_k / (LIQUID_DENSITY * radius * radius);
    let beta_sq = LIQUID_DENSITY * omega * omega * radius * radius / stiffness_k;
    // Damping constants (Devin 1959; Eller 1970), Crum & Mao (1996)
    // eqs. (3.9)–(3.11). (Thesis eq. (2.49) gives δ_th with a
    // surface-tension correction, Im Φ/(Re Φ − 2σ/(R₀P_in,eq)); the
    // correction is ~1e-6 relative and (3.10) is used here.)
    let delta_vis = 4.0 * omega * LIQUID_VISCOSITY / stiffness_k;
    let delta_th = phi_im / phi_re;
    let delta_ac =
        LIQUID_DENSITY * radius.powi(3) * omega.powi(3) / (LIQUID_SOUND_SPEED * stiffness_k);
    let delta_tot = delta_vis + delta_th + delta_ac;
    // Response amplitude α (eq. (3.13)); ε = |α| (P_ac/P_0).
    let alpha = -(ambient_pressure / (LIQUID_DENSITY * radius * radius))
        / ((omega * omega - omega0_sq).powi(2) + (delta_tot * omega0_sq).powi(2)).sqrt();
    let epsilon = alpha.abs() * (acoustic_pressure / ambient_pressure);
    // Resonance factor K (eq. (3.15)); note the denominator's surface
    // term is 2σ/(R0P_0) = 4·σ/(2R0P_0), not 2·σ/(2R0P_0).
    let sigma_term = SURFACE_TENSION / (2.0 * radius * ambient_pressure);
    let k_factor = ((3.0 * kappa + 1.0 - beta_sq) / 4.0
        + sigma_term * (3.0 * kappa + 1.0 - 2.0 / (3.0 * kappa)))
        / (1.0 + 4.0 * sigma_term * (1.0 - 1.0 / (3.0 * kappa)));
    (epsilon, kappa, delta_tot, beta_sq, k_factor)
}

/// Decomposition of the rectified-diffusion molar flux [mol/s] into its
/// rectified-growth and Laplace-dissolution parts (Crum & Hansen 1982;
/// Crum & Mao 1996, eq. (3.6)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectifiedDiffusionTerms {
    /// Net molar flux [mol/s]; positive = growth, negative = dissolution.
    pub net_flux: f64,
    /// Rectified (growth) term only [mol/s]; ≥ 0 for `P_ac > 0`.
    pub growth_flux: f64,
    /// Laplace dissolution term only [mol/s]; ≤ 0.
    pub dissolution_flux: f64,
    /// Fractional oscillation amplitude `ε = |α| (P_ac/P_0)`.
    /// The law assumes `ε << 1`; check this before trusting the rate.
    pub epsilon: f64,
    /// Effective polytropic exponent `κ = Re Φ / 3` (Prosperetti 1977).
    pub kappa: f64,
    /// Total damping constant `δ_tot = δ_vis + δ_th + δ_ac`.
    pub damping_total: f64,
}

/// Rectified-diffusion molar flux, decomposed into growth and dissolution
/// terms (generalised Eller–Flynn rate; see the module docs).
///
/// The single `concentration` argument is used as both the saturation
/// concentration `C_0` and the far-field concentration `C_i` (saturated
/// liquid, `C_i/C_0 = 1`). Degenerate inputs (non-positive radius,
/// ambient pressure, diffusivity, or concentration) return zero flux
/// rather than NaN; a zero frequency returns the Laplace dissolution
/// flux only (no oscillation, hence no rectified term).
#[must_use]
pub fn rectified_diffusion_terms(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> RectifiedDiffusionTerms {
    let zero = RectifiedDiffusionTerms {
        net_flux: 0.0,
        growth_flux: 0.0,
        dissolution_flux: 0.0,
        epsilon: 0.0,
        kappa: 1.0,
        damping_total: 0.0,
    };
    if radius <= 0.0 || ambient_pressure <= 0.0 || diffusivity <= 0.0 || concentration <= 0.0 {
        return zero;
    }
    let (epsilon, kappa, delta_tot, _beta_sq, k_factor) =
        oscillation_response(radius, ambient_pressure, acoustic_pressure, frequency);
    let eps_sq = epsilon * epsilon;
    // Cycle averages (Crum & Mao 1996, eqs. (3.17)–(3.19)).
    let avg_r = 1.0 + k_factor * eps_sq;
    let avg_r4 = 1.0 + (3.0 + 4.0 * k_factor) * eps_sq;
    let c1 = 3.0 * (kappa - 1.0) * (3.0 * kappa - 4.0) / 4.0 + (4.0 - 3.0 * kappa) * k_factor;
    let laplace = 2.0 * SURFACE_TENSION / (radius * ambient_pressure);
    let avg_r4p = (1.0 + c1 * eps_sq) * (1.0 + laplace);
    // Net bracket (saturated: C_i/C_0 = 1) and its split.
    let bracket_net = 1.0 - avg_r4p / avg_r4;
    let bracket_growth = bracket_net + laplace;
    // The (1 + 4σ/(3P_0R_0)) factor in the dR_0/dt form cancels against
    // dn/dR_0 in the molar-flux form (see module docs).
    let base = FOUR_PI * radius * diffusivity * concentration * avg_r;
    let growth_flux = base * bracket_growth;
    let dissolution_flux = base * (-laplace);
    RectifiedDiffusionTerms {
        net_flux: growth_flux + dissolution_flux,
        growth_flux,
        dissolution_flux,
        epsilon,
        kappa,
        damping_total: delta_tot,
    }
}

/// Rectified diffusion model for bubble growth (Eller & Flynn 1965;
/// generalised by Crum & Hansen 1982; Crum & Mao 1996).
///
/// Returns the *net* molar gas flux rate [mol/s] into the bubble:
/// positive above the rectified-diffusion threshold (growth),
/// negative below it (Laplace dissolution). Use
/// [`rectified_diffusion_terms`] for the growth/dissolution split and
/// the oscillation amplitude ε.
///
/// # Validity
/// The linearized law is valid only for P_ac/P₀ ≲ 3
/// ([`RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO`]) and ε << 1. Outside
/// that regime the returned flux is an extrapolation artifact, not a
/// prediction — see the Stage 6 audit. Prefer
/// [`rectified_diffusion_rate_checked`], which rejects out-of-regime
/// inputs instead of silently returning a flux. This is the long-time
/// (quasi-steady) rate; the early-time rate is larger (Smith & Wang
/// 2022, JFM 939, A28). Microstreaming is not included (see module
/// docs): the genuine theory underpredicts Eller (1969) by >20×.
#[must_use]
pub fn rectified_diffusion_rate(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> f64 {
    rectified_diffusion_terms(
        radius,
        ambient_pressure,
        acoustic_pressure,
        frequency,
        diffusivity,
        concentration,
    )
    .net_flux
}

/// Fractional oscillation amplitude `ε = |α| (P_ac/P_0)` of a bubble
/// driven at `frequency` (linear response; Crum & Mao 1996, eq. (3.13)).
/// The rectified-diffusion law assumes `ε << 1`; a value approaching
/// 1 means the small-amplitude derivation no longer applies even if
/// `P_ac/P_0` is inside the guard bound.
#[must_use]
pub fn oscillation_amplitude_fraction(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
) -> f64 {
    oscillation_response(radius, ambient_pressure, acoustic_pressure, frequency).0
}

/// Checked form of [`rectified_diffusion_rate`]: same molar gas flux rate
/// [mol/s], but rejects inputs outside the law's domain of validity instead
/// of silently returning an extrapolation artifact.
///
/// # Errors
/// - Returns [`KwaversError::Physics`] when `ambient_pressure <= 0` (the
///   pressure ratio is undefined) or when `acoustic_pressure /
///   ambient_pressure` exceeds [`RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO`]
///   (small-amplitude regime violated; the bubble is in inertial collapse
///   and the flux has no meaning).
pub fn rectified_diffusion_rate_checked(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> KwaversResult<f64> {
    if ambient_pressure <= 0.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "ambient_pressure".to_owned(),
            value: ambient_pressure,
            reason: "Ambient pressure must be positive for the pressure ratio to be defined"
                .to_owned(),
        }));
    }
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    if pressure_ratio > RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "acoustic_pressure / ambient_pressure".to_owned(),
            value: pressure_ratio,
            reason: format!(
                "Eller-Flynn rectified-diffusion law is valid only for P_ac/P0 <= {RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO} (small-amplitude oscillation); got {pressure_ratio:.1}"
            ),
        }));
    }
    Ok(rectified_diffusion_rate(
        radius,
        ambient_pressure,
        acoustic_pressure,
        frequency,
        diffusivity,
        concentration,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;
    use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, GAS_CONSTANT};
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use kwavers_core::constants::thermodynamic::T_AMBIENT;

    // Standard air-in-water parameters (Eller & Flynn 1965 regime)
    const R0: f64 = 5e-6; // 5 µm bubble radius
    const P0: f64 = ATMOSPHERIC_PRESSURE; // atmospheric pressure [Pa]
    const P_AC: f64 = 120_000.0; // ~1.18 atm acoustic amplitude
    const FREQ: f64 = MHZ_TO_HZ; // 1 MHz ultrasound
    const D_AIR: f64 = 2.0e-9; // O₂ diffusivity in water [m²/s]
    const C_SAT: f64 = 0.26; // dissolved O₂ at saturation [mol/m³]

    /// Isothermal ideal-gas dn/dR for a bubble (Laplace term included);
    /// converts a molar flux [mol/s] to dR/dt [m/s]. Same conversion as
    /// the Stage 7 cross-check (smith_wang_diffusion.py).
    fn dn_dr_mol_per_m(radius: f64) -> f64 {
        (FOUR_PI * radius * radius * P0
            + (16.0 / 3.0) * std::f64::consts::PI * radius * SURFACE_TENSION)
            / (GAS_CONSTANT * T_AMBIENT)
    }

    fn flux_to_um_per_min(flux: f64, radius: f64) -> f64 {
        flux / dn_dr_mol_per_m(radius) * 1e6 * 60.0
    }

    #[test]
    fn test_rectified_diffusion_positive_flux() {
        // Above the rectified-diffusion threshold the net flux is
        // positive: 30 µm at 35 kPa, 26.6 kHz is above threshold
        // (P_T ≈ 27 kPa there) and grows at ≈ +0.75 µm/min.
        let flux = rectified_diffusion_rate(30e-6, P0, 35e3, 26.6e3, D_AIR, C_SAT);
        assert!(
            flux > 0.0,
            "Above threshold, acoustic pressure must yield positive molar flux"
        );
        // Physical reasonableness: flux should be very small (pico- to nano-moles/s)
        assert!(
            flux < 1.0,
            "Molar flux should be << 1 mol/s for microbubbles"
        );
        // The growth term is positive whenever P_ac > 0, even below
        // threshold: the 5 µm / 120 kPa / 1 MHz point is below
        // threshold (net ≈ −11.6 µm/min) but its rectified term is
        // ≈ +25 µm/min.
        let terms = rectified_diffusion_terms(R0, P0, P_AC, FREQ, D_AIR, C_SAT);
        assert!(
            terms.growth_flux > 0.0,
            "Rectified growth term must be positive for P_ac > 0"
        );
        assert!(
            terms.net_flux < 0.0,
            "5 µm / 120 kPa / 1 MHz is below threshold (net dissolution), got {}",
            terms.net_flux
        );
    }

    #[test]
    fn test_rectified_diffusion_amplitude_scaling() {
        // The genuine rectified term is second order in the oscillation
        // amplitude (∝ ε² ∝ P_ac²), NOT linear in P_ac as the old formula
        // had it. At small amplitude (30 µm, 26.6 kHz, 5 vs 10 kPa,
        // ε ≈ 0.012/0.024) the growth flux must quadruple when P_ac
        // doubles. The NET flux does not scale this way (the Laplace
        // dissolution term is P_ac-independent), so the scaling is
        // asserted on the growth term.
        let r_ell = 30e-6;
        let f_ell = 26.6e3;
        let g5 = rectified_diffusion_terms(r_ell, P0, 5e3, f_ell, D_AIR, C_SAT).growth_flux;
        let g10 = rectified_diffusion_terms(r_ell, P0, 10e3, f_ell, D_AIR, C_SAT).growth_flux;
        assert_relative_eq!(g10 / g5, 4.0, epsilon = 0.02);

        // Flux is linear in concentration (both terms scale with C).
        let base = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT);
        let doubled_c = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, 2.0 * C_SAT);
        assert_relative_eq!(doubled_c / base, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rectified_diffusion_zero_acoustic_pressure() {
        // With no sound there is no rectified term — but the bubble still
        // dissolves by Laplace-driven diffusion, so the NET flux is
        // negative, not zero. (The old test asserted zero flux, which
        // encoded the missing dissolution term flagged by the Stage 6
        // audit.) The growth term itself must be exactly zero.
        let terms = rectified_diffusion_terms(R0, P0, 0.0, FREQ, D_AIR, C_SAT);
        assert_eq!(terms.growth_flux, 0.0, "No sound => no rectified growth");
        assert_eq!(terms.epsilon, 0.0, "No sound => zero oscillation amplitude");
        assert!(
            terms.net_flux < 0.0,
            "No sound => net Laplace dissolution, got {}",
            terms.net_flux
        );
        assert_relative_eq!(terms.net_flux, terms.dissolution_flux, epsilon = 1e-12);
    }

    #[test]
    fn test_rectified_diffusion_zero_diffusivity() {
        // No diffusion => no flux of either sign.
        let flux = rectified_diffusion_rate(R0, P0, P_AC, FREQ, 0.0, C_SAT);
        assert_eq!(
            flux, 0.0,
            "Zero diffusivity must yield zero flux, got {flux}"
        );
        let flux_c0 = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, 0.0);
        assert_eq!(flux_c0, 0.0, "Zero concentration must yield zero flux");
    }

    #[test]
    fn test_rectified_diffusion_checked_accepts_in_regime() {
        // P_AC = 120 kPa is P_ac/P0 = 1.18, inside the small-amplitude regime.
        let flux = rectified_diffusion_rate_checked(R0, P0, P_AC, FREQ, D_AIR, C_SAT)
            .expect("in-regime input must be accepted");
        assert_relative_eq!(
            flux,
            rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_rectified_diffusion_checked_rejects_burst_pressures() {
        // Stage 6 audit: the law was applied at P_ac/P0 = 66 (Iwasaki burst,
        // 6.7 MPa) and 146 (Kuji burst, 14.7 MPa), producing 986/1904 µm
        // growth artifacts. The checked form must refuse those inputs.
        for p_ac in [6.7e6, 14.7e6] {
            let err = rectified_diffusion_rate_checked(R0, P0, p_ac, FREQ, D_AIR, C_SAT)
                .expect_err("burst pressure must be rejected");
            assert!(
                matches!(
                    err,
                    KwaversError::Physics(PhysicsError::InvalidParameter { .. })
                ),
                "expected InvalidParameter, got {err:?}"
            );
        }
    }

    #[test]
    fn test_rectified_diffusion_checked_rejects_nonpositive_ambient() {
        let err = rectified_diffusion_rate_checked(R0, 0.0, P_AC, FREQ, D_AIR, C_SAT)
            .expect_err("zero ambient pressure must be rejected");
        assert!(
            matches!(
                err,
                KwaversError::Physics(PhysicsError::InvalidParameter { .. })
            ),
            "expected InvalidParameter, got {err:?}"
        );
    }

    #[test]
    fn test_thermal_phi_limits() {
        // Adiabatic limit X → 0: Φ → 3γ, κ → γ, δ_th → 0.
        let (re, im) = thermal_phi(1e-6, GAMMA_AIR);
        assert_relative_eq!(re / 3.0, GAMMA_AIR, epsilon = 0.01);
        assert!(
            im / re < 0.02,
            "thermal damping must vanish, got {}",
            im / re
        );
        // Isothermal limit X → ∞: Φ → 3, κ → 1, δ_th → 0.
        let (re, im) = thermal_phi(100.0, GAMMA_AIR);
        assert_relative_eq!(re / 3.0, 1.0, epsilon = 0.01);
        assert!(
            im / re < 0.02,
            "thermal damping must vanish, got {}",
            im / re
        );
        // Devin (1959) simplified forms: δ_th ≈ 3(γ−1)χ/2 for χ ≤ 0.08
        // (eq. (2.50)) and δ_th ≈ (γ−1)/(15γχ) for χ ≥ 0.5 (eq. (2.52)),
        // with χ = √X. Agreement is ~30% at small χ, ~1% at χ ≥ 1.
        let (re, im) = thermal_phi(0.05_f64.powi(2), GAMMA_AIR);
        let d_th = im / re;
        assert!(
            (0.02..0.05).contains(&d_th),
            "δ_th at χ=0.05 must be ~0.03 (Devin (2.50)), got {d_th}"
        );
        let (re, im) = thermal_phi(1.0_f64.powi(2), GAMMA_AIR);
        let d_th = im / re;
        assert!(
            (0.01..0.03).contains(&d_th),
            "δ_th at χ=1 must be ~0.019 (Devin (2.52)), got {d_th}"
        );
    }

    #[test]
    fn test_eller_1969_benchmark() {
        // Eller (1969) conditions (Smith & Wang 2022, §4.2): 26.6 kHz,
        // P_ac = 20 kPa, air-saturated water, D = 2e-9 m²/s. Benchmark
        // constants C = 0.26 mol/m³ as in the Stage 7 cross-check.
        // Measured: ~13 µm/min, roughly flat 30–50 µm. Eller's own
        // theoretical values: ~0.6 µm/min (the >20× gap, Smith & Wang §1).
        // The old formula gave 515–527 µm/min here.
        // Expected values (κ-faithful Crum–Hansen, cross-checked against
        // an independent Python implementation to 6+ digits):
        //   R0:  20 µm   30 µm   40 µm   50 µm
        //   net: −1.65   −0.50   −0.13   +0.05 µm/min
        //   gro: +0.94   +0.66   +0.53   +0.47 µm/min
        //   ε:   0.064   0.065   0.066   0.069
        let f_ell = 26.6e3;
        let p_ell = 20e3;
        for (r_um, net_lo, net_hi) in [
            (20.0, -2.5, -0.5),
            (30.0, -1.0, 0.0),
            (40.0, -0.5, 0.2),
            (50.0, -0.2, 0.3),
        ] {
            let r = r_um * 1e-6;
            let terms = rectified_diffusion_terms(r, P0, p_ell, f_ell, D_AIR, C_SAT);
            let net_um_min = flux_to_um_per_min(terms.net_flux, r);
            let gro_um_min = flux_to_um_per_min(terms.growth_flux, r);
            // Net rate is small (|net| ≲ 2 µm/min) and changes sign near
            // the theory's threshold radius R0 ≈ 47 µm (P_T ≈ 27 kPa at
            // 30 µm) versus ~27 µm measured — the documented theory gap,
            // not a formula error.
            assert!(
                net_um_min > net_lo && net_um_min < net_hi,
                "R={r_um} µm: net rate must be small (|net| ≲ 2 µm/min), got {net_um_min:.2} µm/min"
            );
            // Growth term reproduces Eller's theoretical ~0.6 µm/min
            // within a factor of 2 at every radius (0.47–0.94 µm/min).
            assert!(
                gro_um_min > 0.2 && gro_um_min < 1.2,
                "R={r_um} µm: growth term must be within 2× of Eller's theoretical ~0.6 µm/min, got {gro_um_min:.2}"
            );
            // Oscillation amplitude is small (ε ≈ 0.065), as the law assumes.
            assert!(
                terms.epsilon > 0.05 && terms.epsilon < 0.08,
                "R={r_um} µm: ε must be ≈0.065, got {}",
                terms.epsilon
            );
        }
        // The old 40× overprediction is gone: the net flux magnitude at
        // 30 µm is >100× smaller than the old formula's 4.2e-12 mol/s.
        let old_flux =
            FOUR_PI * 30e-6 * D_AIR * C_SAT * (p_ell / P0) * (30e-6 * 30e-6 * f_ell / D_AIR).sqrt();
        let new_flux = rectified_diffusion_rate(30e-6, P0, p_ell, f_ell, D_AIR, C_SAT).abs();
        assert!(
            new_flux < old_flux / 100.0,
            "new |flux| {new_flux:.2e} must be >100× smaller than old {old_flux:.2e}"
        );
    }

    #[test]
    fn test_eller_threshold_pressure() {
        // Threshold pressure at R0 = 30 µm is ≈ 27 kPa in this theory
        // (net rate crosses zero between 25 and 30 kPa at 26.6 kHz).
        let r = 30e-6;
        let f_ell = 26.6e3;
        let below = rectified_diffusion_rate(r, P0, 25e3, f_ell, D_AIR, C_SAT);
        let above = rectified_diffusion_rate(r, P0, 30e3, f_ell, D_AIR, C_SAT);
        assert!(
            below < 0.0,
            "25 kPa must still dissolve at 30 µm, got {below:.2e}"
        );
        assert!(above > 0.0, "30 kPa must grow at 30 µm, got {above:.2e}");
    }
}
