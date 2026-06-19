//! One-dimensional CPML profile kernels.
//!
//! The kernels are axis-agnostic. Callers provide the target arrays, axis
//! length, spacing, PML thickness, k-Wave `pml_alpha`, reference sound speed,
//! and time step. This keeps x/y/z variation data-driven and prevents cloned
//! axis-specific implementations.

use ndarray::Array1;

pub(super) struct CollocatedProfileMut<'a> {
    sigma: &'a mut Array1<f64>,
    kappa: &'a mut Array1<f64>,
    alpha: &'a mut Array1<f64>,
    a_coeff: &'a mut Array1<f64>,
    b_coeff: &'a mut Array1<f64>,
}

/// Scalar inputs for [`compute_collocated_profile`]: axis geometry, the k-Wave σ
/// scale, the time step, and the CFS-PML maxima.
pub(super) struct CollocatedProfileSpec {
    pub dx: f64,
    pub thickness: usize,
    pub pml_alpha: f64,
    pub sound_speed: f64,
    pub dt: f64,
    pub kappa_max: f64,
    pub alpha_max: f64,
}

impl<'a> CollocatedProfileMut<'a> {
    pub(super) fn new(
        sigma: &'a mut Array1<f64>,
        kappa: &'a mut Array1<f64>,
        alpha: &'a mut Array1<f64>,
        a_coeff: &'a mut Array1<f64>,
        b_coeff: &'a mut Array1<f64>,
    ) -> Self {
        Self {
            sigma,
            kappa,
            alpha,
            a_coeff,
            b_coeff,
        }
    }
}

/// Collocated CPML profile with optional complex-frequency-shift (CFS) terms.
///
/// The σ damping profile is the exact k-Wave form
/// `σ(q) = pml_alpha·(c/dx)·q⁴`. On top of σ this computes the two CFS terms
/// (Roden & Gedney 2000; Komatitsch & Martin 2007):
///
/// ```text
/// κ(q) = 1 + (κ_max − 1)·q⁴          real coordinate stretch  (κ_max ≥ 1)
/// α(q) = α_max·(1 − q)               frequency shift          (α_max ≥ 0)
/// ```
///
/// where the normalized depth `q ∈ (0,1]` is `1` at the outer wall and `→0` at
/// the physical-domain interface. Note α grades **opposite** to σ/κ — maximal at
/// the interface, zero at the wall — which is the load-bearing CFS detail
/// (inverting it turns absorption into amplification).
///
/// Recursive-convolution coefficients (canonical SEISMIC_CPML / Roden & Gedney):
///
/// ```text
/// b = exp[−(σ/κ + α)·Δt]
/// a = σ·(b − 1) / [κ·(σ + κ·α)]      (a = 0 where σ + κ·α = 0)
/// ```
///
/// When `κ_max = 1` and `α_max = 0` this reduces **exactly** to the σ-only CPML
/// (`κ = 1`, `α = 0`, `b = exp(−σΔt)`, `a = b − 1`), so the CFS upgrade is opt-in
/// and the legacy default behavior is bit-identical.
///
/// For the left PML, index `i = 0` is the outer wall and uses
/// `q = (thickness - i) / thickness`. For the right PML, `q` increases from
/// `1 / thickness` at the physical-domain interface to `1` at the outer wall.
pub(super) fn compute_collocated_profile(
    profile: CollocatedProfileMut<'_>,
    n: usize,
    spec: &CollocatedProfileSpec,
) {
    let CollocatedProfileSpec {
        dx,
        thickness,
        pml_alpha,
        sound_speed,
        dt,
        kappa_max,
        alpha_max,
    } = *spec;

    if n <= 1 || thickness == 0 {
        set_neutral(profile);
        return;
    }

    let mut assign = |idx: usize, q: f64, sigma_val: f64| {
        let kappa_val = (kappa_max - 1.0).mul_add(q.powi(4), 1.0); // 1 + (κ_max−1)·q⁴
        let alpha_val = alpha_max * (1.0 - q); // α_max·(1−q)
        profile.sigma[idx] = sigma_val;
        profile.kappa[idx] = kappa_val;
        profile.alpha[idx] = alpha_val;

        let b = (-(sigma_val / kappa_val + alpha_val) * dt).exp();
        // a = σ(b−1)/[κ(σ+κα)]; the denominator vanishes only where σ = α = 0
        // (κ ≥ 1), and there σ = 0 ⇒ the cell contributes no memory anyway.
        let denom = kappa_val * (sigma_val + kappa_val * alpha_val);
        let a = if denom > 0.0 {
            sigma_val * (b - 1.0) / denom
        } else {
            0.0
        };
        profile.b_coeff[idx] = b;
        profile.a_coeff[idx] = a;
    };

    for i in 0..thickness.min(n) {
        let q = (thickness - i) as f64 / thickness as f64;
        assign(i, q, pml_alpha * (sound_speed / dx) * q.powi(4));
    }

    let right_start = n.saturating_sub(thickness);
    for i in right_start..n {
        let q = (i - right_start + 1) as f64 / thickness as f64;
        assign(i, q, pml_alpha * (sound_speed / dx) * q.powi(4));
    }
}

/// Exact k-Wave staggered PML profile for velocity components.
///
/// k-Wave shifts the profile by a half cell. The rightmost staggered cell can
/// exceed the collocated wall value because the half-cell point lies outside
/// the physical-domain sample centers; this is required for parity.
pub(super) fn compute_staggered_profile(
    sigma_sg: &mut Array1<f64>,
    n: usize,
    dx: f64,
    thickness: usize,
    pml_alpha: f64,
    sound_speed: f64,
) {
    if n <= 1 || thickness == 0 {
        sigma_sg.fill(0.0);
        return;
    }

    let t = thickness as f64;

    for i in 0..thickness.min(n) {
        let q = (t - i as f64 - 0.5) / t;
        sigma_sg[i] = pml_alpha * (sound_speed / dx) * q.abs().powi(4);
    }

    let right_start = n.saturating_sub(thickness);
    for i in right_start..n {
        let j = (i - right_start) as f64;
        let q = (j + 1.5) / t;
        sigma_sg[i] = pml_alpha * (sound_speed / dx) * q.powi(4);
    }
}

fn set_neutral(profile: CollocatedProfileMut<'_>) {
    profile.sigma.fill(0.0);
    profile.kappa.fill(1.0);
    profile.alpha.fill(0.0);
    profile.a_coeff.fill(0.0);
    profile.b_coeff.fill(1.0);
}
