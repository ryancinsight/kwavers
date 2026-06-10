//! Acousto-optic diffraction — the complete theory.
//!
//! When light traverses a region carrying a sound wave, the photoelastic effect
//! turns the sound into a moving **phase grating** that diffracts the light into
//! orders `m = 0, ±1, ±2, …`. Which diffraction *model* applies is set by the
//! **Klein–Cook parameter**
//!
//! ```text
//! Q = 2π λ₀ L / (n Λ²)
//! ```
//!
//! (`λ₀` optical vacuum wavelength, `L` interaction length, `n` refractive
//! index, `Λ` acoustic wavelength):
//!
//! - **`Q ≪ 1` — Raman–Nath** (thin grating): many symmetric orders, intensities
//!   `Iₘ = Jₘ²(ν)` where `ν = 2π Δn L / λ₀` is the peak phase modulation.
//! - **`Q ≫ 1` — Bragg** (thick grating): a single first order at the Bragg
//!   angle, efficiency `η = sin²(ν/2)`.
//! - **intermediate `Q`**: neither closed form is exact; the general
//!   **Klein–Cook coupled-wave equations** must be integrated
//!   ([`solve_coupled_orders`]).
//!
//! This module provides the regime classifier, both analytic limits, the
//! diffraction geometry, and the general coupled-wave solver, which reduces to
//! the Raman–Nath Bessel result as `Q → 0` and to the Bragg `sin²` result as
//! `Q → ∞` (both verified by the test suite).
//!
//! # References
//! - Raman, C. V., & Nath, N. S. N. (1935). *Proc. Indian Acad. Sci.* 2, 406.
//! - Klein, W. R., & Cook, B. D. (1967). "Unified approach to ultrasonic light
//!   diffraction." *IEEE Trans. Sonics Ultrason.* SU-14(3), 123–134.
//! - Korpel, A. (1997). *Acousto-Optics* (2nd ed.). Marcel Dekker.
//! - Saleh, B. E. A., & Teich, M. C. (2007). *Fundamentals of Photonics*, §20.

use core::f64::consts::{PI, TAU};
use kwavers_math::special::bessel::jn;
use num_complex::Complex64;

/// Acousto-optic diffraction regime selected by the Klein–Cook parameter `Q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffractionRegime {
    /// Thin-grating (multi-order Bessel) regime, `Q ≤ 0.3`.
    RamanNath,
    /// Transition regime, `0.3 < Q < 4π` (no closed form — use the solver).
    Intermediate,
    /// Thick-grating (single-order Bragg) regime, `Q ≥ 4π`.
    Bragg,
}

/// Klein–Cook parameter `Q = 2π λ₀ L / (n Λ²)`.
///
/// `optical_wavelength_m` is the **vacuum** optical wavelength `λ₀`,
/// `interaction_length_m` the sound-column width `L` the light crosses,
/// `refractive_index` the medium index `n`, `acoustic_wavelength_m` the sound
/// wavelength `Λ`. Returns `+∞` for non-positive `n`/`Λ` (degenerate).
#[must_use]
pub fn klein_cook_parameter(
    optical_wavelength_m: f64,
    interaction_length_m: f64,
    refractive_index: f64,
    acoustic_wavelength_m: f64,
) -> f64 {
    let denom = refractive_index * acoustic_wavelength_m * acoustic_wavelength_m;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    TAU * optical_wavelength_m * interaction_length_m / denom
}

/// Classify the diffraction regime from the Klein–Cook parameter (Klein & Cook
/// 1967 thresholds: `Q ≤ 0.3` Raman–Nath, `Q ≥ 4π` Bragg).
#[must_use]
pub fn diffraction_regime(q: f64) -> DiffractionRegime {
    if q <= 0.3 {
        DiffractionRegime::RamanNath
    } else if q >= 4.0 * PI {
        DiffractionRegime::Bragg
    } else {
        DiffractionRegime::Intermediate
    }
}

/// Raman–Nath peak phase parameter `ν = 2π Δn L / λ₀`.
///
/// `delta_n` is the peak photoelastic refractive-index modulation
/// `Δn = ½ n³ p_e S` (`p_e` photoelastic coefficient, `S` strain amplitude).
#[must_use]
pub fn raman_nath_parameter(
    delta_n: f64,
    interaction_length_m: f64,
    optical_wavelength_m: f64,
) -> f64 {
    if optical_wavelength_m <= 0.0 {
        return 0.0;
    }
    TAU * delta_n * interaction_length_m / optical_wavelength_m
}

/// **Raman–Nath** (thin-grating) order intensities `Iₘ = Jₘ²(ν)` for
/// `m = −max_order ..= +max_order`, returned in a vector indexed by
/// `m + max_order` (so index `max_order` is the zeroth order).
///
/// The orders are symmetric (`I₋ₘ = Iₘ`, since `J₋ₘ = (−1)ᵐ Jₘ`) and conserve
/// energy: `Σₘ Jₘ²(ν) = 1` (Jacobi–Anger / Bessel sum rule). Valid for `Q ≪ 1`.
#[must_use]
pub fn raman_nath_order_intensities(nu: f64, max_order: u32) -> Vec<f64> {
    let span = 2 * max_order as usize + 1;
    let mut out = vec![0.0; span];
    for m in 0..=max_order {
        let jm = jn(m, nu);
        let i_m = jm * jm;
        out[(max_order + m) as usize] = i_m; // +m
        out[(max_order - m) as usize] = i_m; // −m (symmetric)
    }
    out
}

/// **Bragg** (thick-grating) first-order diffraction efficiency `η = sin²(ν/2)`
/// at exact Bragg incidence. The zeroth order carries `cos²(ν/2)`. Valid for
/// `Q ≫ 1`; complete energy transfer occurs at `ν = π`.
#[must_use]
pub fn bragg_diffraction_efficiency(nu: f64) -> f64 {
    let s = (0.5 * nu).sin();
    s * s
}

/// Diffraction angle `θₘ` of order `m` from `sin θₘ = m λ₀ / (n Λ)` (the grating
/// equation in the medium). Returns `None` when the order is evanescent
/// (`|sin θₘ| > 1`) or the geometry is degenerate.
#[must_use]
pub fn diffraction_angle_rad(
    order: i32,
    optical_wavelength_m: f64,
    refractive_index: f64,
    acoustic_wavelength_m: f64,
) -> Option<f64> {
    let denom = refractive_index * acoustic_wavelength_m;
    if denom <= 0.0 {
        return None;
    }
    let sin_theta = order as f64 * optical_wavelength_m / denom;
    if sin_theta.abs() > 1.0 {
        None
    } else {
        Some(sin_theta.asin())
    }
}

/// General **Klein–Cook coupled-wave** solver for acousto-optic diffraction.
///
/// Integrates the coupled-amplitude equations for the complex order amplitudes
/// `Eₗ(ξ)` across the normalised interaction coordinate `ξ ∈ [0, 1]`:
///
/// ```text
/// dEₗ/dξ = −i (ν/2)(Eₗ₋₁ + Eₗ₊₁) − i (Q/2)(l² + 2 l α) Eₗ ,   Eₗ(0) = δₗ₀
/// ```
///
/// where `ν` is the Raman–Nath phase parameter, `Q` the Klein–Cook parameter,
/// and `α` the normalised incidence parameter (`α = 0` normal incidence;
/// `α = −½` exact Bragg incidence onto the +1 order). The first term is the
/// grating coupling between adjacent orders; the second is the propagation
/// (Bragg-mismatch) phase that suppresses off-resonant orders as `Q` grows.
///
/// Returns the exit order intensities `|Eₗ(1)|²` for `l = −max_order ..= max_order`
/// (indexed `l + max_order`). The model is **complete**: as `Q → 0` it reproduces
/// the Raman–Nath result `Jₗ²(ν)`, and at large `Q` with `α = −½` it reproduces
/// the Bragg result `η = sin²(ν/2)`. Integration is RK4 with `n_steps` steps;
/// orders beyond `±max_order` are truncated to zero (choose `max_order` a few
/// above `ν` for accuracy in the Raman–Nath regime).
#[must_use]
pub fn solve_coupled_orders(
    nu: f64,
    q: f64,
    incidence_alpha: f64,
    max_order: u32,
    n_steps: usize,
) -> Vec<f64> {
    let span = 2 * max_order as usize + 1;
    let center = max_order as i64;
    let steps = n_steps.max(1);
    let h = 1.0 / steps as f64;

    // Index map: state[idx] ↔ order l = idx as i64 − center.
    let mut e = vec![Complex64::new(0.0, 0.0); span];
    e[center as usize] = Complex64::new(1.0, 0.0); // incident order 0

    // dEₗ/dξ (ξ-independent here: the equations are autonomous).
    let deriv = |state: &[Complex64]| -> Vec<Complex64> {
        let half_nu = 0.5 * nu;
        let half_q = 0.5 * q;
        let mut d = vec![Complex64::new(0.0, 0.0); span];
        for (idx, dl) in d.iter_mut().enumerate() {
            let l = idx as i64 - center;
            let lo = if idx > 0 {
                state[idx - 1]
            } else {
                Complex64::new(0.0, 0.0)
            };
            let hi = if idx + 1 < span {
                state[idx + 1]
            } else {
                Complex64::new(0.0, 0.0)
            };
            let l_f = l as f64;
            let phase = half_q * l_f.mul_add(l_f, 2.0 * l_f * incidence_alpha);
            // −i[(ν/2)(E₋+E₊) + phase·Eₗ]
            *dl = Complex64::new(0.0, -1.0) * (half_nu * (lo + hi) + phase * state[idx]);
        }
        d
    };

    let axpy = |a: &[Complex64], s: f64, b: &[Complex64]| -> Vec<Complex64> {
        a.iter().zip(b).map(|(x, y)| x + y * s).collect()
    };

    for _ in 0..steps {
        let k1 = deriv(&e);
        let k2 = deriv(&axpy(&e, 0.5 * h, &k1));
        let k3 = deriv(&axpy(&e, 0.5 * h, &k2));
        let k4 = deriv(&axpy(&e, h, &k3));
        for idx in 0..span {
            e[idx] += (k1[idx] + (k2[idx] + k3[idx]) * 2.0 + k4[idx]) * (h / 6.0);
        }
    }

    e.iter().map(num_complex::Complex::norm_sqr).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(v: &[f64]) -> f64 {
        v.iter().sum()
    }

    /// Klein–Cook parameter and regime classification.
    #[test]
    fn klein_cook_and_regime_classification() {
        // λ₀=633nm, n=1.33, Λ=300µm, L=1mm → thin (Raman–Nath).
        let q_thin = klein_cook_parameter(633e-9, 1e-3, 1.33, 300e-6);
        assert!(q_thin < 0.3, "Q={q_thin}");
        assert_eq!(diffraction_regime(q_thin), DiffractionRegime::RamanNath);

        // Short acoustic wavelength + long interaction → thick (Bragg).
        let q_thick = klein_cook_parameter(633e-9, 10e-3, 1.33, 10e-6);
        assert!(q_thick >= 4.0 * PI, "Q={q_thick}");
        assert_eq!(diffraction_regime(q_thick), DiffractionRegime::Bragg);

        assert_eq!(diffraction_regime(1.0), DiffractionRegime::Intermediate);
        assert!(klein_cook_parameter(633e-9, 1e-3, 0.0, 1e-4).is_infinite());
    }

    /// Raman–Nath orders are symmetric, equal the Bessel squares, and conserve
    /// energy `Σ Jₘ²(ν) = 1`.
    #[test]
    fn raman_nath_orders_are_bessel_and_conserve_energy() {
        let nu = 2.5;
        let max = 10;
        let i = raman_nath_order_intensities(nu, max);
        assert_eq!(i.len(), 2 * max as usize + 1);

        // Symmetry I₋ₘ = Iₘ and exact Bessel values.
        for m in 0..=max {
            let jm = jn(m, nu);
            assert!((i[(max + m) as usize] - jm * jm).abs() < 1e-12);
            assert!((i[(max - m) as usize] - i[(max + m) as usize]).abs() < 1e-15);
        }
        // Energy conservation (enough orders captured).
        assert!((sum(&i) - 1.0).abs() < 1e-9, "Σ Iₘ = {}", sum(&i));

        // Small-ν limit: I₀ ≈ 1, I₁ ≈ (ν/2)².
        let small = raman_nath_order_intensities(0.05, 4);
        assert!((small[4] - 1.0).abs() < 1e-2);
        assert!((small[5] - (0.025_f64).powi(2)).abs() < 1e-4);
    }

    /// Bragg efficiency `sin²(ν/2)`: full transfer at ν=π, none at ν=0/2π.
    #[test]
    fn bragg_efficiency_closed_form() {
        assert!(bragg_diffraction_efficiency(0.0).abs() < 1e-15);
        assert!((bragg_diffraction_efficiency(PI) - 1.0).abs() < 1e-12);
        assert!(bragg_diffraction_efficiency(TAU).abs() < 1e-12);
        assert!((bragg_diffraction_efficiency(PI / 2.0) - 0.5).abs() < 1e-12);
    }

    /// Grating-equation diffraction angles, with evanescent cut-off.
    #[test]
    fn diffraction_angle_grating_equation() {
        let (lam0, n, lambda) = (633e-9, 1.33, 50e-6);
        let theta1 = diffraction_angle_rad(1, lam0, n, lambda).unwrap();
        // sin θ₁ = λ₀/(nΛ).
        assert!((theta1.sin() - lam0 / (n * lambda)).abs() < 1e-15);
        // Zeroth order is undeviated.
        assert!(diffraction_angle_rad(0, lam0, n, lambda).unwrap().abs() < 1e-15);
        // A very small acoustic wavelength pushes high orders evanescent.
        assert!(diffraction_angle_rad(5, lam0, n, 100e-9).is_none());
    }

    /// The general coupled-wave solver reduces to the **Raman–Nath** Bessel
    /// result as Q → 0, and conserves energy.
    #[test]
    fn coupled_solver_recovers_raman_nath_at_low_q() {
        let nu = 2.0;
        let max = 12;
        let intensities = solve_coupled_orders(nu, 0.0, 0.0, max, 2000);
        assert!((sum(&intensities) - 1.0).abs() < 1e-6, "energy {}", sum(&intensities));
        for m in 0..=6u32 {
            let analytic = {
                let j = jn(m, nu);
                j * j
            };
            assert!(
                (intensities[(max + m) as usize] - analytic).abs() < 5e-4,
                "order {m}: {} vs Bessel {analytic}",
                intensities[(max + m) as usize]
            );
        }
    }

    /// The general solver reduces to the **Bragg** `sin²(ν/2)` result at large Q
    /// with Bragg incidence (α = −½): energy concentrates in orders 0 and 1.
    #[test]
    fn coupled_solver_recovers_bragg_at_high_q() {
        let nu = 1.2;
        let max = 6;
        let intensities = solve_coupled_orders(nu, 60.0, -0.5, max, 4000);
        assert!((sum(&intensities) - 1.0).abs() < 1e-5, "energy {}", sum(&intensities));

        let i0 = intensities[max as usize];
        let i1 = intensities[(max + 1) as usize];
        let eta = bragg_diffraction_efficiency(nu); // sin²(ν/2)
        assert!((i1 - eta).abs() < 2e-2, "Bragg η: {i1} vs {eta}");
        assert!((i0 - (1.0 - eta)).abs() < 2e-2, "Bragg I₀: {i0} vs {}", 1.0 - eta);
        // Spurious orders (≥2 and the −1 mismatched order) are suppressed.
        assert!(intensities[(max + 2) as usize] < 1e-2);
        assert!(intensities[(max - 1) as usize] < 1e-2);
    }

    /// Energy is conserved throughout the intermediate regime for any ν, Q, α.
    #[test]
    fn coupled_solver_conserves_energy_in_all_regimes() {
        for &(nu, q, alpha) in &[(3.0, 1.0, 0.0), (1.5, 8.0, -0.5), (4.0, 0.5, 0.2)] {
            let s = sum(&solve_coupled_orders(nu, q, alpha, 16, 3000));
            assert!((s - 1.0).abs() < 1e-5, "energy {s} for ν={nu},Q={q},α={alpha}");
        }
    }

    /// The raman_nath_parameter helper composes ν = 2π Δn L / λ₀.
    #[test]
    fn raman_nath_parameter_formula() {
        let nu = raman_nath_parameter(1e-4, 1e-3, 633e-9);
        assert!((nu - TAU * 1e-4 * 1e-3 / 633e-9).abs() < 1e-9);
        assert_eq!(raman_nath_parameter(1e-4, 1e-3, 0.0), 0.0);
    }
}
