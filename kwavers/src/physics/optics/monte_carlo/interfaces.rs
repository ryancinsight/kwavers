//! Fresnel reflection and refraction at optical interfaces
//!
//! Implements the Fresnel equations for unpolarised photon Monte Carlo transport
//! at planar boundaries between optical media of differing refractive index.
//! This is the critical physics that distinguishes MCML (Wang et al. 1995) from
//! a pure diffusion-approximation solver.
//!
//! ---
//!
//! ## Theorem — Fresnel Reflectance (Unpolarised Light)
//!
//! For a photon in medium n₁ incident on a planar interface at angle θᵢ
//! (measured from the surface normal), the power reflectance for unpolarised
//! light is the average of the TE (s-polarised) and TM (p-polarised) Fresnel
//! coefficients:
//!
//! ```text
//! R(θᵢ) = (R_s + R_p) / 2
//!
//! R_s = │ n₁ cos θᵢ − n₂ cos θₜ │²   (TE: E parallel to interface)
//!        ─────────────────────────────
//!        │ n₁ cos θᵢ + n₂ cos θₜ │
//!
//! R_p = │ n₁ cos θₜ − n₂ cos θᵢ │²   (TM: H parallel to interface)
//!        ─────────────────────────────
//!        │ n₁ cos θₜ + n₂ cos θᵢ │
//! ```
//!
//! The transmitted angle θₜ is given by Snell's law:
//!
//! ```text
//! n₁ sin θᵢ = n₂ sin θₜ
//! ```
//!
//! Total internal reflection (TIR) occurs when n₁ > n₂ and
//! sin θᵢ > n₂ / n₁; in that case R = 1 identically.
//!
//! ## Algorithm (per photon–interface interaction)
//!
//! 1. Compute cos θᵢ = |ŝ · n̂| where n̂ is the inward surface normal.
//! 2. Check TIR: if n₁ > n₂ and sin²θᵢ > (n₂/n₁)², return R = 1, reflect.
//! 3. Compute cos θₜ = √(1 − (n₁/n₂)² sin²θᵢ).
//! 4. Evaluate R_s, R_p; average → R.
//! 5. Sample ξ ~ U(0,1): if ξ < R, reflect; else refract.
//!    - Reflect: flip normal component of direction.
//!    - Refract: apply Snell's rotation (see [`refract_direction`]).
//!
//! ## References
//!
//! 1. Wang, L. V., Jacques, S. L., & Zheng, L. (1995). MCML—Monte Carlo
//!    modeling of light transport in multi-layered tissues. *Computer Methods
//!    and Programs in Biomedicine*, **47**(2), 131–146.
//!    <https://doi.org/10.1016/0169-2607(95)01640-F>
//!
//! 2. Born, M., & Wolf, E. (1999). *Principles of Optics* (7th ed.).
//!    Cambridge University Press. (Chapter 1.5: Boundary conditions and Fresnel formulae.)
//!
//! 3. Jacques, S. L. (1998). Light distributions from point, line, and plane
//!    sources for photochemical reactions and fluorescence in turbid biological
//!    tissues. *Photochemistry and Photobiology*, **67**(1), 23–32.

use rand::Rng;

/// Outcome of a photon–interface interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceOutcome {
    /// Photon reflected back into the incident medium.
    Reflected,
    /// Photon transmitted into the second medium (direction refracted by Snell's law).
    Transmitted,
}

/// Fresnel reflectance for unpolarised light.
///
/// Returns the power reflectance R ∈ [0, 1] for a photon travelling in a
/// medium with refractive index `n1` incident on an interface with a medium
/// with refractive index `n2`.
///
/// # Arguments
/// * `n1`        — Refractive index of the incident medium (dimensionless, ≥ 1)
/// * `n2`        — Refractive index of the transmitted medium (dimensionless, ≥ 1)
/// * `cos_theta_i` — cos θᵢ, where θᵢ is the angle of incidence measured from
///                   the inward surface normal; must be in [0, 1]
///
/// # Returns
/// Power reflectance R ∈ [0, 1].  Returns 1.0 for total internal reflection.
#[must_use]
pub fn fresnel_reflectance(n1: f64, n2: f64, cos_theta_i: f64) -> f64 {
    // Clamp cos_theta_i to [0, 1] (numerical safety)
    let cti = cos_theta_i.clamp(0.0, 1.0);

    if (n1 - n2).abs() < 1e-12 {
        return 0.0; // Identical media — no reflection
    }

    // Snell's law: sin²θₜ = (n₁/n₂)² sin²θᵢ = (n₁/n₂)² (1 - cos²θᵢ)
    let ratio = n1 / n2;
    let sin2_theta_i = 1.0 - cti * cti;
    let sin2_theta_t = ratio * ratio * sin2_theta_i;

    if sin2_theta_t >= 1.0 {
        return 1.0; // Total internal reflection
    }

    let cos_theta_t = (1.0 - sin2_theta_t).sqrt();

    // TE (s-polarised): R_s = [(n₁ cosθᵢ − n₂ cosθₜ) / (n₁ cosθᵢ + n₂ cosθₜ)]²
    let n1_ci = n1 * cti;
    let n2_ct = n2 * cos_theta_t;
    let r_s = {
        let num = n1_ci - n2_ct;
        let den = n1_ci + n2_ct;
        if den.abs() < 1e-30 { 1.0 } else { (num / den).powi(2) }
    };

    // TM (p-polarised): R_p = [(n₁ cosθₜ − n₂ cosθᵢ) / (n₁ cosθₜ + n₂ cosθᵢ)]²
    let n1_ct = n1 * cos_theta_t;
    let n2_ci = n2 * cti;
    let r_p = {
        let num = n1_ct - n2_ci;
        let den = n1_ct + n2_ci;
        if den.abs() < 1e-30 { 1.0 } else { (num / den).powi(2) }
    };

    // Unpolarised average
    0.5 * (r_s + r_p)
}

/// Reflect a photon direction about a surface normal.
///
/// The surface normal `n_hat` is assumed to point *into* the incident medium.
/// Returns the reflected direction (unit vector).
#[must_use]
pub fn reflect_direction(dir: [f64; 3], n_hat: [f64; 3]) -> [f64; 3] {
    // r = d − 2(d·n)n
    let dot = dir[0] * n_hat[0] + dir[1] * n_hat[1] + dir[2] * n_hat[2];
    [
        dir[0] - 2.0 * dot * n_hat[0],
        dir[1] - 2.0 * dot * n_hat[1],
        dir[2] - 2.0 * dot * n_hat[2],
    ]
}

/// Refract a photon direction by Snell's law.
///
/// ## Derivation
///
/// Given incident direction dᵢ (unit vector, pointing *into* the interface),
/// inward normal n̂ (pointing into the incident medium), and refractive index
/// ratio η = n₁/n₂:
///
/// ```text
/// dₜ = η·dᵢ + (η·cos θᵢ − cos θₜ)·n̂
/// ```
///
/// where cos θᵢ = −dᵢ·n̂ (positive when d points toward interface),
/// and cos θₜ = √(1 − η² sin²θᵢ).
///
/// **Reference**: Shirley, P. & Morley, R. K. (2003). *Realistic Ray Tracing*
/// (2nd ed.), AK Peters. Eq. (13.1).
///
/// # Returns
/// `None` if total internal reflection occurs; `Some(refracted_dir)` otherwise.
#[must_use]
pub fn refract_direction(dir: [f64; 3], n_hat: [f64; 3], n1: f64, n2: f64) -> Option<[f64; 3]> {
    let eta = n1 / n2;
    // cos θᵢ = −dir·n̂  (dir points toward surface, n̂ points away from surface)
    let cos_i = -(dir[0] * n_hat[0] + dir[1] * n_hat[1] + dir[2] * n_hat[2]);
    let cos_i = cos_i.max(0.0);

    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return None; // Total internal reflection
    }

    let cos_t = (1.0 - sin2_t).sqrt();
    let scale = eta * cos_i - cos_t;
    Some([
        eta * dir[0] + scale * n_hat[0],
        eta * dir[1] + scale * n_hat[1],
        eta * dir[2] + scale * n_hat[2],
    ])
}

/// Apply Fresnel reflection or transmission at an optical interface.
///
/// Samples ξ ~ U(0,1) and reflects with probability R or transmits with
/// probability 1−R. Updates `direction` in-place.
///
/// # Arguments
/// * `direction` — Photon direction (modified in-place)
/// * `n_hat`     — Inward surface normal (pointing into incident medium)
/// * `n1`        — Refractive index of incident medium
/// * `n2`        — Refractive index of transmitted medium
/// * `rng`       — Random number generator
///
/// # Returns
/// [`InterfaceOutcome::Reflected`] or [`InterfaceOutcome::Transmitted`].
pub fn apply_fresnel<R: Rng>(
    direction: &mut [f64; 3],
    n_hat: [f64; 3],
    n1: f64,
    n2: f64,
    rng: &mut R,
) -> InterfaceOutcome {
    // cos θᵢ from the *outward* convention used for the incident side
    let cos_theta_i = {
        let d = *direction;
        let dot = d[0] * n_hat[0] + d[1] * n_hat[1] + d[2] * n_hat[2];
        // n_hat points INTO the incident medium; cos θᵢ = -d·n̂ when d
        // points toward the surface
        (-dot).clamp(0.0, 1.0)
    };

    let r = fresnel_reflectance(n1, n2, cos_theta_i);

    if rng.gen::<f64>() < r {
        // Reflect
        let reflected = reflect_direction(*direction, n_hat);
        *direction = reflected;
        InterfaceOutcome::Reflected
    } else {
        // Transmit (refract)
        match refract_direction(*direction, n_hat, n1, n2) {
            Some(refracted) => {
                *direction = refracted;
                InterfaceOutcome::Transmitted
            }
            None => {
                // Numerical TIR fallback — reflect
                let reflected = reflect_direction(*direction, n_hat);
                *direction = reflected;
                InterfaceOutcome::Reflected
            }
        }
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Fresnel reflectance at normal incidence against the analytic formula:
    /// R₀ = [(n₁ − n₂)/(n₁ + n₂)]²
    #[test]
    fn test_fresnel_normal_incidence() {
        // Water–glass: n₁=1.33, n₂=1.5 → R₀ = ((1.33-1.5)/(1.33+1.5))² ≈ 0.00359
        let r = fresnel_reflectance(1.33, 1.5, 1.0);
        let expected = ((1.33_f64 - 1.5_f64) / (1.33_f64 + 1.5_f64)).powi(2);
        assert!((r - expected).abs() < 1e-10, "R₀ = {r:.6}, expected {expected:.6}");
    }

    /// At grazing incidence (cos θᵢ → 0) reflectance → 1 for any interface.
    #[test]
    fn test_fresnel_grazing_incidence() {
        let r = fresnel_reflectance(1.0, 1.5, 0.0);
        assert!((r - 1.0).abs() < 1e-6, "Grazing reflectance must be 1, got {r}");
    }

    /// Total internal reflection: n₁ > n₂, sin θᵢ > n₂/n₁.
    #[test]
    fn test_total_internal_reflection() {
        let n1 = 1.5;
        let n2 = 1.0;
        let critical_angle = f64::asin(n2 / n1); // ≈ 41.8°
        // Incident angle = critical + 10° → TIR
        let theta_i = critical_angle + 0.174; // +10°
        let cos_i = theta_i.cos();
        let r = fresnel_reflectance(n1, n2, cos_i);
        assert!((r - 1.0).abs() < 1e-10, "TIR: expected R=1, got {r}");
    }

    /// Identical refractive indices → zero reflectance.
    #[test]
    fn test_fresnel_identical_media() {
        let r = fresnel_reflectance(1.4, 1.4, 0.5);
        assert_eq!(r, 0.0, "Same n → R must be 0");
    }

    /// Snell's law verification: refracted direction satisfies
    /// n₁ sin θᵢ = n₂ sin θₜ.
    #[test]
    fn test_snell_law_refraction() {
        let n1 = 1.0;
        let n2 = 1.4;

        // Incident direction at 30° from normal (into +z half-space)
        let theta_i = 30.0_f64.to_radians();
        let dir = [theta_i.sin(), 0.0, theta_i.cos()];
        let n_hat = [0.0, 0.0, 1.0]; // normal in +z direction (incident side)

        let refracted = refract_direction(dir, n_hat, n1, n2).expect("No TIR at 30°");

        // sin θₜ = (n₁/n₂) sin θᵢ
        let sin_theta_i = theta_i.sin();
        let sin_theta_t_expected = (n1 / n2) * sin_theta_i;

        // sin θₜ from the refracted direction's transverse component
        let sin_theta_t_actual = (refracted[0] * refracted[0] + refracted[1] * refracted[1]).sqrt();

        assert!(
            (sin_theta_t_actual - sin_theta_t_expected).abs() < 1e-10,
            "Snell's law violated: sin_t = {sin_theta_t_actual:.8}, expected {sin_theta_t_expected:.8}"
        );
    }

    /// Reflection: direction component along normal reverses.
    #[test]
    fn test_reflect_direction() {
        let dir = [0.5_f64.sqrt(), 0.0, 0.5_f64.sqrt()]; // 45° to normal
        let n_hat = [0.0, 0.0, 1.0];
        let reflected = reflect_direction(dir, n_hat);

        // z-component should flip sign
        assert!((reflected[0] - dir[0]).abs() < 1e-10);
        assert!((reflected[1] - dir[1]).abs() < 1e-10);
        assert!((reflected[2] + dir[2]).abs() < 1e-10); // sign flipped
    }
}
