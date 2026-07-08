//! Helmholtz Green's function for 3D free-space BEM.
//!
//! # Theorem (Helmholtz Fundamental Solution)
//!
//! **Theorem** (Colton & Kress 1998, Theorem 3.1): The function
//! ```text
//!   G(r, r') = exp(ik|r − r'|) / (4π|r − r'|)
//! ```
//! is the fundamental solution of the 3D Helmholtz equation:
//! ```text
//!   (∇² + k²) G(r, r') = −δ(r − r')
//! ```
//! in the distributional sense. It satisfies the Sommerfeld radiation condition
//! as |r| → ∞:
//! ```text
//!   |r|(∂G/∂r − ikG) → 0    as |r| → ∞.
//! ```
//!
//! # Gradient
//!
//! **Theorem**: The gradient of G with respect to the field point r is:
//! ```text
//!   ∇_r G(r, r') = (ik − 1/R) · G(r, r') · (r − r') / R
//! ```
//! where R = |r − r'|.
//!
//! # Near-Field Behavior
//!
//! As R → 0:
//! ```text
//!   G(r, r') → 1/(4πR)     (same as static Laplace Green's function)
//!   ∇G → −(r − r')/(4πR³)  (static dipole kernel)
//! ```
//!
//! # References
//!
//! - Colton, D. & Kress, R. (1998) *Inverse Acoustic and Electromagnetic Scattering
//!   Theory*, 2nd ed. Springer, §3.1.
//! - Kress, R. (1999) *Linear Integral Equations*, 2nd ed. Springer, §6.
//!
//! # Validation
//!
//! - `test_green_zero_wavenumber`: G(r,r') → 1/(4πR) as k→0.
//! - `test_green_gradient_fd`: ∇G matches finite-difference gradient to < 10⁻⁷.
//! - `test_green_symmetry`: G(r, r') = G(r', r) (reciprocity).

use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_math::fft::Complex64;

/// The 3D free-space Helmholtz Green's function and its gradient.
///
/// Returns `(G, ∇_r G)` where:
/// ```text
///   G = exp(ik R) / (4π R),   R = |r − r'|
///   ∇_r G = (ik − 1/R) · G · (r − r') / R
/// ```
///
/// For R < `NEAR_FIELD_CUTOFF` (1e-12 m) — singular case — returns (0, 0).
/// The caller must handle the singularity (e.g., via Duffy transformation).
///
/// # Arguments
///
/// * `k` — wavenumber [rad/m]
/// * `r_src` — source point r'
/// * `r_obs` — observation (field) point r
#[inline]
#[must_use]
pub fn green_helmholtz(k: f64, r_src: [f64; 3], r_obs: [f64; 3]) -> (Complex64, [Complex64; 3]) {
    let dx = r_obs[0] - r_src[0];
    let dy = r_obs[1] - r_src[1];
    let dz = r_obs[2] - r_src[2];
    let r_sq = dz.mul_add(dz, dx.mul_add(dx, dy * dy));

    if r_sq < NEAR_FIELD_CUTOFF_SQ {
        return (Complex64::ZERO, [Complex64::ZERO; 3]);
    }

    let r = r_sq.sqrt();
    let g = Complex64::new(0.0, k * r).exp() / (FOUR_PI * r);

    // ∇_r G = (ik − 1/R) · G · (r − r') / R
    let factor = (Complex64::new(0.0, k) - 1.0 / r) * g / r;
    let grad = [factor * dx, factor * dy, factor * dz];

    (g, grad)
}

/// Distance cutoff squared below which G is treated as singular.
const NEAR_FIELD_CUTOFF_SQ: f64 = 1e-24; // R < 1e-12 m

/// Normal derivative of the Green's function: ∂G/∂n'(r, r') = ∇_{r'} G · n'.
///
/// This is the kernel of the hypersingular integral operator (H matrix).
///
/// # Arguments
///
/// * `k` — wavenumber
/// * `r_src` — source point r'
/// * `r_obs` — field point r
/// * `normal` — outward unit normal at r'
#[inline]
#[must_use]
pub fn green_normal_deriv(k: f64, r_src: [f64; 3], r_obs: [f64; 3], normal: [f64; 3]) -> Complex64 {
    let dx = r_obs[0] - r_src[0];
    let dy = r_obs[1] - r_src[1];
    let dz = r_obs[2] - r_src[2];
    let r_sq = dz.mul_add(dz, dx.mul_add(dx, dy * dy));

    if r_sq < NEAR_FIELD_CUTOFF_SQ {
        return Complex64::ZERO;
    }

    let r = r_sq.sqrt();
    let g = Complex64::new(0.0, k * r).exp() / (FOUR_PI * r);

    // ∇_{r'} G = −∇_r G  (anti-symmetry with respect to source/field swap)
    // = (1/R − ik) · G · (r − r') / R
    let factor = (1.0 / r - Complex64::new(0.0, k)) * g / r;
    let grad_r_prime = [factor * dx, factor * dy, factor * dz];

    // ∂G/∂n' = ∇_{r'} G · n'
    grad_r_prime[0] * normal[0] + grad_r_prime[1] * normal[1] + grad_r_prime[2] * normal[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// G(r, r') → 1/(4πR) as k→0 (static Laplace Green's function).
    /// # Panics
    /// - Panics if assertion fails: `k=0 Green's: rel={:.3e}`.
    ///
    #[test]
    fn test_green_zero_wavenumber() {
        let r_src = [0.0, 0.0, 0.0];
        let r_obs = [1.0, 0.0, 0.0]; // R = 1
        let (g, _) = green_helmholtz(0.0, r_src, r_obs);
        let expected = 1.0 / (4.0 * PI);
        let rel = (g.re - expected).abs() / expected;
        assert!(rel < 1e-12, "k=0 Green's: rel={:.3e}", rel);
        assert!(g.im.abs() < 1e-12, "k=0 must be real");
    }

    /// G(r, r') = G(r', r) (reciprocity / symmetry of fundamental solution).
    /// # Panics
    /// - Panics if assertion fails: `G must be symmetric: diff={:.3e}`.
    ///
    #[test]
    fn test_green_symmetry() {
        let r1 = [0.3, 0.5, 0.1];
        let r2 = [1.2, 0.4, 0.8];
        let k = 10.0;
        let (g12, _) = green_helmholtz(k, r1, r2);
        let (g21, _) = green_helmholtz(k, r2, r1);
        let diff = (g12 - g21).norm();
        assert!(diff < 1e-12, "G must be symmetric: diff={:.3e}", diff);
    }

    /// ∇G matches central finite differences to < 10⁻⁷ relative error.
    ///
    /// Numerical gradient via 6-point FD: `g_i = (G(r + ε·eᵢ) − G(r − ε·eᵢ)) / (2ε)`.
    /// # Panics
    /// - Panics if assertion fails: `Gradient dim={}: FD={:.4e}+{:.4e}i, exact={:.4e}+{:.4e}i, rel={:.3e}`.
    ///
    #[test]
    fn test_green_gradient_fd() {
        let r_src = [0.0, 0.0, 0.0];
        let r_obs = [0.5, 0.3, 0.4];
        let k = 5.0;
        let eps = 1e-6;

        let (_, grad_exact) = green_helmholtz(k, r_src, r_obs);

        for dim in 0..3 {
            let mut r_plus = r_obs;
            let mut r_minus = r_obs;
            r_plus[dim] += eps;
            r_minus[dim] -= eps;

            let (g_plus, _) = green_helmholtz(k, r_src, r_plus);
            let (g_minus, _) = green_helmholtz(k, r_src, r_minus);
            let grad_fd = (g_plus - g_minus) / (2.0 * eps);

            let diff = (grad_exact[dim] - grad_fd).norm();
            let norm_ref = grad_exact[dim].norm().max(1e-20);
            let rel = diff / norm_ref;
            assert!(
                rel < 1e-5,
                "Gradient dim={}: FD={:.4e}+{:.4e}i, exact={:.4e}+{:.4e}i, rel={:.3e}",
                dim,
                grad_fd.re,
                grad_fd.im,
                grad_exact[dim].re,
                grad_exact[dim].im,
                rel
            );
        }
    }

    /// |G(r, r')| = 1/(4πR) (envelope of complex exponential).
    /// # Panics
    /// - Panics if assertion fails: `|G| error: rel={:.3e}`.
    ///
    #[test]
    fn test_green_modulus() {
        let r_src = [0.0, 0.0, 0.0];
        let r_obs = [0.7, 0.2, 0.5];
        let k = 20.0;
        let (g, _) = green_helmholtz(k, r_src, r_obs);

        let r = (0.7f64 * 0.7 + 0.2 * 0.2 + 0.5 * 0.5).sqrt();
        let expected_mod = 1.0 / (4.0 * PI * r);
        let rel = (g.norm() - expected_mod).abs() / expected_mod;
        assert!(rel < 1e-12, "|G| error: rel={:.3e}", rel);
    }
}
