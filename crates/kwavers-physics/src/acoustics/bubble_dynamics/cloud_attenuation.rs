//! Bubble-cloud attenuation from single-bubble extinction
//! cross-sections (Xia 2020 energy method), the cloud-length growth
//! law of Bader & Holland, and slab shielding factors.
//!
//! # Two attenuation routes — use the right one
//!
//! - **Linear, monodisperse clouds:** [`commander_prosperetti_attenuation`]
//!   in [`bubbly_medium`](super::bubbly_medium) is the canonical
//!   implementation (Commander & Prosperetti 1989). It assumes
//!   small-amplitude oscillation and is a lower bound (thermal damping
//!   omitted there).
//! - **Strongly nonlinear oscillation** (this module): when bubbles
//!   collapse inertially every cycle, the linear dispersion relation
//!   does not apply. Xia (2020, arXiv:2004.05050, Eqs. (8)–(10), (13))
//!   instead extracts the energy a single bubble removes from the
//!   incident wave per cycle from its (nonlinear) trajectory,
//!   `W = |∮ p_ac dV|`, forms the extinction cross-section
//!   `σ_e = W·f/I` with incident intensity `I = P_A²/(2ρc)`, and gets
//!   the amplitude attenuation coefficient `α = N·σ_e/2` for number
//!   density `N`.
//!
//! In the linear regime the two routes agree in order of magnitude:
//! the prototype's Build 9 derivation (formation/lit_build9_derivation.md)
//! found the Xia/Keller–Miksis value within a factor ~0.83 of Medwin's
//! linear cross-section, and the Commander & Prosperetti linear theory
//! within the same order — cross-checks, not identities, since the
//! linear theory is inapplicable at the formation conditions
//! (20 MPa, `Rmax/R0 ≈ 21`).
//!
//! # Shielding factors and the Build 9 negative result
//!
//! [`shielding_factor_coherent`] is the population-averaged influx
//! multiplier for a coherent wave crossing a uniform cloud slab when
//! the local influx scales as `p^m`:
//! `S = ⟨(p/p0)^m⟩ = (1 − e^(−mαL))/(mαL)`.
//! The cloud length grows as the cloud forms: Bader & Holland (2016)
//! §3.4, Eqs. (15)–(17) give `L(n) = (n − n_crit)·λ/3` with
//! `n_crit ≈ 3.5` cycles ([`cloud_length`]).
//!
//! These pieces were assembled in the prototype (Build 9) to test
//! whether cloud shielding can produce a sharp influx cutoff between
//! 25 and 56 trigger cycles — the only functional form that satisfied
//! both the Kuji and Iwasaki protocols. **It cannot, as a derived
//! law:** attenuation gives `S(n) ∝ 1/n`, while the protocol pair
//! needs a decay exponent γ ≈ 1.8–3 (a near-step); the derived
//! coefficients (α = 3150 Np/m Kuji / 2037 Np/m Iwasaki at the
//! formation conditions) turn over at 8–9 cycles in coherent slab
//! models and 31/51 cycles in the most favourable diffuse model, and
//! shielding acts *more* strongly on Kuji than on Iwasaki — the wrong
//! direction for the required differential. The mechanisms here are
//! general physics and are implemented as such; the negative result
//! is recorded so the cutoff hypothesis is not re-derived from
//! scratch. Burst shielding likewise *widens* the formation gap
//! rather than closing it.

use std::f64::consts::PI;

/// Energy [J] a bubble extracts from the incident wave over one
/// cycle: `W = |∮ p_ac dV|`, with `V = (4π/3)·R³`.
///
/// `pressure` is the instantaneous acoustic pressure `p∞(t) − p0`
/// [Pa] (see [`InstantaneousPressure`](super::InstantaneousPressure))
/// and `radius` the radius [m], sampled over exactly one cycle; the
/// loop is closed from the last sample back to the first. In the
/// linear regime `|W|` equals the radiated + viscous dissipated power
/// (the prototype's dissipation check agreed to 0.05%).
#[must_use]
pub fn cycle_work(pressure: &[f64], radius: &[f64]) -> f64 {
    if pressure.len() < 2 || pressure.len() != radius.len() {
        return 0.0;
    }
    let volume: Vec<f64> = radius
        .iter()
        .map(|&r| (4.0 / 3.0) * PI * r.powi(3))
        .collect();
    let n = volume.len();
    let mut work = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        work += 0.5 * (pressure[i] + pressure[j]) * (volume[j] - volume[i]);
    }
    work.abs()
}

/// Extinction cross-section [m²] of one bubble (Xia 2020, Eqs.
/// (8)–(10)): `σ_e = W·f/I` with incident intensity
/// `I = P_A²/(2ρc)`.
///
/// `work_per_cycle` is from [`cycle_work`]; `pressure_amplitude` is
/// the drive pressure *amplitude* `P_A` [Pa].
#[must_use]
pub fn extinction_cross_section(
    work_per_cycle: f64,
    frequency: f64,
    pressure_amplitude: f64,
    density: f64,
    sound_speed: f64,
) -> f64 {
    if work_per_cycle <= 0.0
        || frequency <= 0.0
        || pressure_amplitude <= 0.0
        || density <= 0.0
        || sound_speed <= 0.0
    {
        return 0.0;
    }
    let intensity = pressure_amplitude.powi(2) / (2.0 * density * sound_speed);
    work_per_cycle * frequency / intensity
}

/// Amplitude attenuation coefficient [Np/m] of a bubble cloud:
/// `α = N·σ_e/2` (Xia 2020, Eq. (13)).
///
/// `number_density` is in bubbles/m³. For the linear-regime
/// alternative see
/// [`commander_prosperetti_attenuation`](super::bubbly_medium::commander_prosperetti_attenuation).
#[must_use]
pub fn cloud_attenuation_coefficient(number_density: f64, extinction_cross_section: f64) -> f64 {
    if number_density <= 0.0 || extinction_cross_section <= 0.0 {
        return 0.0;
    }
    number_density * extinction_cross_section / 2.0
}

/// Cloud length [m] after `cycles` cycles of cloud formation:
/// `L(n) = (n − n_crit)·λ/3` (Bader & Holland 2016, §3.4, Eqs.
/// (15)–(17)), with `λ = c/f` and `formation_cycles` the cloud
/// formation delay `n_crit ≈ 3.5`. Zero before the cloud forms.
#[must_use]
pub fn cloud_length(cycles: f64, formation_cycles: f64, frequency: f64, sound_speed: f64) -> f64 {
    if frequency <= 0.0 || sound_speed <= 0.0 {
        return 0.0;
    }
    let grown = (cycles - formation_cycles).max(0.0);
    grown * (sound_speed / frequency) / 3.0
}

/// Population-averaged influx multiplier for a coherent wave crossing
/// a uniform cloud slab of length `length` [m] with attenuation
/// `alpha` [Np/m], when the local influx scales as `p^m`
/// (`influx_exponent` `m`; the Bader influx's D_MAX exponent gives
/// `m ≈ 1.12`):
///
/// ```text
/// S = (1/L) ∫₀^L e^(−mαx) dx = (1 − e^(−mαL)) / (mαL)
/// ```
///
/// `S → 1` as `αL → 0` and `S → 1/(mαL)` for a thick cloud — the
/// `1/n` tail behind the Build 9 negative result in the module docs.
///
/// Evaluated as `−expm1(−x)/x`: `expm1` computes `e^{−x} − 1` to
/// full relative precision at small `x`, where the direct
/// `(1 − exp(−x))/x` loses up to ~1.6e-11 absolute to cancellation
/// (at `x ≈ 1e-6`; the error decays as ~1e-16/x). The previous
/// form's Taylor branch (`1 − x/2` below `x = 1e-6`) additionally
/// undershot the closed form by `x²/6 ≈ 1.7e-13` at the seam,
/// making `S` non-monotone there. The single `expm1` form has no
/// seam, matches the closed form to ≤1 ulp at every `x > 0`, is
/// monotone decreasing, and preserves `0 < S ≤ 1` (physics review
/// 2026-11-27).
#[must_use]
pub fn shielding_factor_coherent(alpha: f64, length: f64, influx_exponent: f64) -> f64 {
    let x = influx_exponent * alpha * length;
    if x <= 0.0 {
        return 1.0;
    }
    -(-x).exp_m1() / x
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;

    #[test]
    fn test_cycle_work_analytic_ellipse() {
        // p = P̂·sin(ωt), V = V0 + V̂·sin(ωt − φ):
        // ∮ p dV = π·P̂·V̂·sin(φ) exactly.
        let n = 2000;
        let p_hat = 1.0e5;
        let r0: f64 = 5e-6;
        let v0 = (4.0 / 3.0) * PI * r0.powi(3);
        let v_hat = 0.1 * v0;
        let phi = PI / 3.0;
        let mut pressure = Vec::with_capacity(n);
        let mut radius = Vec::with_capacity(n);
        for i in 0..n {
            let phase = 2.0 * PI * i as f64 / n as f64;
            pressure.push(p_hat * phase.sin());
            let v = v0 + v_hat * (phase - phi).sin();
            radius.push((3.0 * v / (4.0 * PI)).powf(1.0 / 3.0));
        }
        let w = cycle_work(&pressure, &radius);
        let expected = PI * p_hat * v_hat * phi.sin();
        assert_relative_eq!(w, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_extinction_and_attenuation_algebra() {
        // Build 9 derived α = 3150 Np/m at N = 1.678e11 m⁻³ (Kuji
        // formation conditions): σ_e = 2α/N, so the coefficient
        // must return 3150 from the un-truncated cross-section.
        let sigma_e_full = 2.0 * 3150.0 / 1.678e11;
        let alpha = cloud_attenuation_coefficient(1.678e11, sigma_e_full);
        assert_relative_eq!(alpha, 3150.0, epsilon = 0.01);
        // σ_e from W: at P_A = 1e5 Pa in water, I = 3.33e3 W/m².
        let sigma_e = extinction_cross_section(1e-9, 1.0e6, 1.0e5, 1000.0, 1500.0);
        let intensity = 1.0e10 / (2.0 * 1000.0 * 1500.0);
        assert_relative_eq!(sigma_e, 1e-9 * 1.0e6 / intensity, epsilon = 1e-12);
    }

    #[test]
    fn test_cloud_length_law() {
        // Kuji trigger end: n = 25, n_crit = 3.5, f = 1 MHz, c = 1500.
        let l = cloud_length(25.0, 3.5, 1.0e6, 1500.0);
        assert_relative_eq!(l, 21.5 * 1.5e-3 / 3.0, epsilon = 1e-12);
        // Before formation: zero.
        assert_eq!(cloud_length(2.0, 3.5, 1.0e6, 1500.0), 0.0);
    }

    #[test]
    fn test_shielding_factor_seam_monotone() {
        // The expm1 evaluation has no seam at x = 1e-6: S is
        // non-increasing across it (to rounding) and matches the
        // series 1 − x/2 + x²/6 there (physics review 2026-11-27;
        // the previous Taylor branch jumped up by ~x²/6 at the seam
        // and the direct closed form carried ~1e-11 cancellation
        // noise around it).
        let mut prev = shielding_factor_coherent(9.9e-7, 1.0, 1.0);
        for i in 1..=1000 {
            let x = 9.9e-7 + i as f64 * 2e-13;
            let v = shielding_factor_coherent(x, 1.0, 1.0);
            assert!(v <= prev + 1e-15, "S must be non-increasing: S({x}) = {v} > {prev}");
            prev = v;
        }
        let s_seam = shielding_factor_coherent(1e-6, 1.0, 1.0);
        let series = 1.0 - 1e-6 / 2.0 + 1e-12 / 6.0;
        assert!(
            (s_seam - series).abs() < 1e-15,
            "S(1e-6) = {s_seam}, series {series}"
        );
    }

    #[test]
    fn test_shielding_factor_limits() {
        assert_eq!(shielding_factor_coherent(0.0, 0.01, 1.12), 1.0);
        // mαL = 1.12: S = (1 − e^−1.12)/1.12 ≈ 0.6015.
        let s = shielding_factor_coherent(100.0, 0.01, 1.12);
        assert_relative_eq!(s, (1.0 - (-1.12_f64).exp()) / 1.12, epsilon = 1e-12);
        // Thick cloud: S → 1/(mαL).
        let s_thick = shielding_factor_coherent(3150.0, 0.05, 1.12);
        let asym = 1.0 / (1.12 * 3150.0 * 0.05);
        assert!((s_thick - asym).abs() / asym < 0.01);
    }
}
