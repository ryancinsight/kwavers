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
//! KW-PHYS-001 restored the second-order oscillation-amplitude dependence
//! and the Laplace term. The additive transient contribution is proportional
//! to `R0/√(πDt)` and decays to zero in the steady limit.
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
//! The measurements replotted in Smith & Wang (2022, A28, figure 3)
//! vary with initial equilibrium radius. The plotted rates at approximately
//! 30, 36, 42.5, and 45 µm are +0.5, +5, +9, and +16 µm/min.
//! Compare them with the signed net rate. The isolated rectified-growth
//! term measures the acoustic contribution before Laplace dissolution.
//! The saturated steady model predicts net dissolution at all four radii.
//!
//! At finite exposure, the dissolved-gas profile records the pressure
//! and radius history. Smith & Wang (2022, A28) study this transient
//! regime. Near boundaries or with non-spherical motion, microstreaming
//! may also alter the mass transfer (Church 1988; Moreno Soto et al. 2020).
//! The explicit-concentration API allows independent far-field and flat-interface
//! saturation concentrations within the same steady, small-oscillation law.
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

mod saturation;

pub use saturation::rectified_diffusion_rate_with_concentrations;

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
    /// Rectified growth term [mol/s] in the small-amplitude growth regime.
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
    let cycle = cycle_transport(radius, ambient_pressure, acoustic_pressure, frequency);
    let bracket_growth = 1.0 - cycle.pressure_ratio + cycle.laplace_ratio;
    let base = FOUR_PI * radius * diffusivity * concentration * cycle.mean_radius;
    let growth_flux = base * bracket_growth;
    let dissolution_flux = base * (-cycle.laplace_ratio);
    RectifiedDiffusionTerms {
        net_flux: growth_flux + dissolution_flux,
        growth_flux,
        dissolution_flux,
        epsilon: cycle.epsilon,
        kappa: cycle.kappa,
        damping_total: cycle.damping_total,
    }
}

struct CycleTransport {
    pressure_ratio: f64,
    laplace_ratio: f64,
    mean_radius: f64,
    epsilon: f64,
    kappa: f64,
    damping_total: f64,
}

fn cycle_transport(
    radius: f64,
    ambient_pressure: f64,
    acoustic_pressure: f64,
    frequency: f64,
) -> CycleTransport {
    let (epsilon, kappa, damping_total, _, k_factor) =
        oscillation_response(radius, ambient_pressure, acoustic_pressure, frequency);
    let eps_sq = epsilon * epsilon;
    // Cycle averages (Zhang 2012, Eqs. 3.17–3.19).
    let avg_r4 = 1.0 + (3.0 + 4.0 * k_factor) * eps_sq;
    let c1 = 3.0 * (kappa - 1.0) * (3.0 * kappa - 4.0) / 4.0 + (4.0 - 3.0 * kappa) * k_factor;
    let laplace_ratio = 2.0 * SURFACE_TENSION / (radius * ambient_pressure);
    let avg_r4p = (1.0 + c1 * eps_sq) * (1.0 + laplace_ratio);
    CycleTransport {
        pressure_ratio: avg_r4p / avg_r4,
        laplace_ratio,
        mean_radius: 1.0 + k_factor * eps_sq,
        epsilon,
        kappa,
        damping_total,
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
/// that regime the output extrapolates the small-amplitude model. Use
/// [`rectified_diffusion_rate_checked`] for the pressure-ratio guard and
/// [`oscillation_amplitude_fraction`] to assess the amplitude. This is
/// the long-time (quasi-steady) rate for a spherical bubble in bulk water.
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
/// - Returns [`KwaversError::Physics`] when `ambient_pressure` is not
///   positive (the pressure ratio is undefined) or when
///   `acoustic_pressure / ambient_pressure` is not at most
///   [`RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO`] (small-amplitude regime
///   violated; the bubble is in inertial collapse and the flux has no
///   meaning). Both tests carry explicit NaN disjuncts so they also
///   reject NaN, whose comparisons are all false — the convention of
///   the Bader checked variant.
pub fn rectified_diffusion_rate_checked(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> KwaversResult<f64> {
    if ambient_pressure <= 0.0 || ambient_pressure.is_nan() {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "ambient_pressure".to_owned(),
            value: ambient_pressure,
            reason: "Ambient pressure must be positive for the pressure ratio to be defined"
                .to_owned(),
        }));
    }
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    if pressure_ratio > RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO || pressure_ratio.is_nan() {
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
mod tests;
