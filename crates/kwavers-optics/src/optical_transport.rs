//! Optical transport in the diffusion approximation (Photoacoustics chapter §4).
//!
//! Closed-form light-transport quantities that supply the optical fluence to the
//! photoacoustic forward model `p₀ = Γ·μ_a·Φ`: the reduced scattering
//! coefficient, the diffusion coefficient, the effective attenuation
//! (Theorem 4.1), the penetration depth, and the diffuse fluence decay with
//! depth.
//!
//! All optical coefficients (`μ_a`, `μ_s`, `μ_s'`, `μ_eff`) share consistent
//! inverse-length units (e.g. all m⁻¹ or all cm⁻¹); lengths/depths use the
//! reciprocal unit. The functions are unit-agnostic — consistency is the
//! caller's responsibility.
//!
//! # References
//! - Wang, L. V. & Wu, H. (2007). *Biomedical Optics: Principles and Imaging.*
//! - Cox, B., et al. (2012). "Quantitative spectroscopic photoacoustic imaging."
//!   *Applied Optics* 51(5), 1245–1259.

/// Reduced scattering coefficient `μ_s' = μ_s·(1 − g)` (§4.2), where `g ∈ [−1, 1]`
/// is the scattering anisotropy (mean cosine of the scattering angle). For soft
/// tissue `g ≈ 0.9`, so `μ_s' ≈ 0.1·μ_s`.
#[must_use]
pub fn reduced_scattering(mu_s: f64, g: f64) -> f64 {
    mu_s * (1.0 - g)
}

/// Optical diffusion coefficient `D = 1 / (3·(μ_a + μ_s'))` (§4.2).
///
/// Returns `0.0` when the transport coefficient `μ_a + μ_s'` is non-positive.
#[must_use]
pub fn diffusion_coefficient(mu_a: f64, mu_s_prime: f64) -> f64 {
    let transport = mu_a + mu_s_prime;
    if transport > 0.0 {
        1.0 / (3.0 * transport)
    } else {
        0.0
    }
}

/// Effective attenuation coefficient `μ_eff = √(3·μ_a·(μ_a + μ_s'))`
/// (Theorem 4.1) — the `1/e` decay rate of the diffuse fluence rate, equal to
/// `√(μ_a / D)`.
///
/// The radicand is clamped to `≥ 0` before the square root so non-physical
/// negative inputs return `0` rather than `NaN`.
#[must_use]
pub fn effective_attenuation(mu_a: f64, mu_s_prime: f64) -> f64 {
    (3.0 * mu_a * (mu_a + mu_s_prime)).max(0.0).sqrt()
}

/// Optical penetration depth `δ = 1 / μ_eff` — the `1/e` diffuse-fluence decay
/// length. Returns `f64::INFINITY` for a transparent medium (`μ_eff = 0`).
#[must_use]
pub fn penetration_depth(mu_eff: f64) -> f64 {
    if mu_eff > 0.0 {
        1.0 / mu_eff
    } else {
        f64::INFINITY
    }
}

/// Diffuse optical fluence at depth `z` under planar surface illumination,
/// `F(z) = F₀·exp(−μ_eff·z)` (§9–§10). `z ≥ 0` measures depth below the surface;
/// negative `z` extrapolates the exponential.
#[must_use]
pub fn planar_fluence_at_depth(surface_fluence: f64, mu_eff: f64, z: f64) -> f64 {
    surface_fluence * (-mu_eff * z).exp()
}

/// Photoacoustic initial pressure `p₀ = Γ·μ_a·F` (§1) — the Grüneisen-weighted
/// absorbed optical energy density that seeds the acoustic field, given the
/// dimensionless Grüneisen parameter `Γ`, absorption `μ_a`, and local fluence
/// `F`.
#[must_use]
pub fn initial_pressure(grueneisen: f64, mu_a: f64, fluence: f64) -> f64 {
    grueneisen * mu_a * fluence
}

/// Apparent (depth-biased) absorption coefficient inferred from a raw PA signal
/// **without** fluence compensation: `μ̃_a = μ_a·exp(−μ_eff·z)` (Theorem 10.1).
/// The true `μ_a` is underestimated by `exp(−μ_eff·z)` at depth `z` — e.g. ≈40 %
/// at `z = 1/μ_eff`.
#[must_use]
pub fn apparent_absorption(mu_a_true: f64, mu_eff: f64, z: f64) -> f64 {
    mu_a_true * (-mu_eff * z).exp()
}

/// Quantitative-PA fluence compensation: recover the true absorption coefficient
/// from a measured PA signal by dividing out the Grüneisen-weighted fluence,
/// `μ_a = S / (Γ·F)` (§10.2) — the inverse of [`initial_pressure`]. Returns `0`
/// when `Γ·F` is non-positive (no usable signal to invert).
#[must_use]
pub fn compensate_fluence(signal: f64, grueneisen: f64, fluence: f64) -> f64 {
    let denom = grueneisen * fluence;
    if denom > 0.0 {
        signal / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// μ_s' = μ_s(1 − g); g ≈ 0.9 ⇒ μ_s' ≈ 0.1·μ_s (§4.2).
    #[test]
    fn reduced_scattering_matches_definition() {
        assert!((reduced_scattering(100.0, 0.9) - 10.0).abs() < 1e-12);
        assert!((reduced_scattering(100.0, 0.0) - 100.0).abs() < 1e-12); // isotropic
    }

    /// Theorem 4.1: μ_eff = √(3 μ_a (μ_a + μ_s')) and equivalently √(μ_a / D).
    /// For NIR tissue (μ_a = 0.1, μ_s' = 10 cm⁻¹) ⇒ μ_eff ≈ 1.74 cm⁻¹, inside the
    /// chapter's 0.5–3 cm⁻¹ band.
    #[test]
    fn effective_attenuation_theorem_and_diffusion_consistency() {
        let (mu_a, mu_s_p) = (0.1, 10.0);
        let mu_eff = effective_attenuation(mu_a, mu_s_p);
        assert!((mu_eff - (3.0 * mu_a * (mu_a + mu_s_p)).sqrt()).abs() < 1e-12);
        assert!((0.5..=3.0).contains(&mu_eff), "μ_eff = {mu_eff} cm⁻¹");
        // μ_eff = √(μ_a / D).
        let d = diffusion_coefficient(mu_a, mu_s_p);
        assert!((mu_eff - (mu_a / d).sqrt()).abs() < 1e-12);
        // Degenerate transport ⇒ D = 0.
        assert_eq!(diffusion_coefficient(0.0, 0.0), 0.0);
    }

    /// Penetration depth is the reciprocal of μ_eff; δ ≈ 5.7 mm for the NIR case
    /// (in the chapter's 3–20 mm range), and ∞ for a transparent medium.
    #[test]
    fn penetration_depth_is_reciprocal() {
        let mu_eff = effective_attenuation(0.1, 10.0); // cm⁻¹
        let delta_cm = penetration_depth(mu_eff);
        assert!((delta_cm - 1.0 / mu_eff).abs() < 1e-12);
        assert!((0.3..=2.0).contains(&delta_cm), "δ = {delta_cm} cm");
        assert!(penetration_depth(0.0).is_infinite());
    }

    /// Fluence decays to 1/e of the surface value at one penetration depth, and
    /// equals the surface value at z = 0.
    #[test]
    fn planar_fluence_decays_to_one_over_e_at_penetration_depth() {
        let mu_eff = 2.0;
        let f0 = 50.0;
        assert!((planar_fluence_at_depth(f0, mu_eff, 0.0) - f0).abs() < 1e-12);
        let delta = penetration_depth(mu_eff);
        let f_delta = planar_fluence_at_depth(f0, mu_eff, delta);
        assert!((f_delta - f0 / std::f64::consts::E).abs() < 1e-12);
    }

    /// p₀ = Γ·μ_a·F (§1).
    #[test]
    fn initial_pressure_is_grueneisen_weighted_absorption() {
        // Γ = 0.2, μ_a = 5 m⁻¹, F = 1000 J/m² ⇒ p₀ = 1000 Pa.
        assert!((initial_pressure(0.2, 5.0, 1000.0) - 1000.0).abs() < 1e-9);
        assert_eq!(initial_pressure(0.2, 0.0, 1000.0), 0.0); // no absorber, no source
    }

    /// Theorem 10.1: apparent μ_a is the true value times exp(−μ_eff·z); at one
    /// penetration depth the underestimation factor is 1/e (≈ 37 %).
    #[test]
    fn apparent_absorption_underestimates_by_depth_bias() {
        let (mu_a, mu_eff) = (1.0, 2.0);
        let z = penetration_depth(mu_eff); // = 0.5
        assert!((apparent_absorption(mu_a, mu_eff, z) - mu_a / std::f64::consts::E).abs() < 1e-12);
        // At the surface there is no bias.
        assert!((apparent_absorption(mu_a, mu_eff, 0.0) - mu_a).abs() < 1e-12);
    }

    /// Fluence compensation inverts the forward PA model: dividing the synthetic
    /// signal S = Γ·μ_a·F by Γ·F recovers the true μ_a (§10.2).
    #[test]
    fn compensate_fluence_recovers_true_absorption() {
        let (gamma, mu_a, f) = (0.2, 7.5, 1500.0);
        let signal = initial_pressure(gamma, mu_a, f);
        assert!((compensate_fluence(signal, gamma, f) - mu_a).abs() < 1e-9);
        // Degenerate (zero Grüneisen·fluence) ⇒ 0.
        assert_eq!(compensate_fluence(signal, 0.0, f), 0.0);
    }
}
