//! Frequency-domain viscoelastic constitutive models for the shear channel.
//!
//! Three complex-shear-modulus models of increasing fidelity, all expressing the
//! dispersive shear-wave behaviour through the complex wavenumber
//! `k = ω √(ρ / G*(ω))`:
//!
//! - [`KelvinVoigtModel`] — spring ∥ dashpot, `G* = μ + iωη_s`; unbounded
//!   high-frequency dispersion.
//! - [`ZenerModel`] — standard linear solid (one relaxation arm); bounded
//!   dispersion between a relaxed and an unrelaxed modulus.
//! - [`GeneralizedMaxwellModel`] — `N` relaxation arms,
//!   `G* = E_∞ + Σ_j E_j iωτ_j/(1+iωτ_j)`; with weights `E_j ∝ τ_j^{1-y}` it
//!   reproduces an arbitrary power-law absorption `α ∝ ω^y` (Fung 1993), the
//!   discrete analog of the fractional-Laplacian operator (book §4.4.3 / §4.8.3).
//!
//! These are the medium-layer counterpart of the analytical `voigt_complex_modulus`
//! plotting helpers: solvers and inversion kernels query them for the
//! frequency-dependent modulus of a viscoelastic tissue. The shared
//! [`recover_complex_modulus`] inverts the dispersion relation for
//! model-agnostic rheological recovery (shear-wave spectroscopy).
//!
//! # References
//! - Catheline, S., et al. (2004). "Measurement of viscoelastic properties of
//!   homogeneous soft solid using transient elastography." *J. Acoust. Soc. Am.*,
//!   116(6), 3734–3741.
//! - Deffieux, T., et al. (2009). "Shear wave spectroscopy for in vivo
//!   quantification of human soft tissues viscoelasticity." *IEEE TMI*, 28(3).
//! - Fung, Y.C. (1993). *Biomechanics: Mechanical Properties of Living Tissues*,
//!   2nd ed. Springer.

mod generalized_maxwell;
mod kelvin_voigt;
mod zener;

pub use generalized_maxwell::GeneralizedMaxwellModel;
pub use kelvin_voigt::KelvinVoigtModel;
pub use zener::ZenerModel;

use eunomia::Complex64;

/// A measured shear-wave dispersion sample for the `fit_dispersion` inversions.
#[derive(Debug, Clone, Copy)]
pub struct DispersionSample {
    /// Angular frequency `ω` \[rad·s⁻¹].
    pub omega: f64,
    /// Measured phase velocity `c_p(ω)` \[m·s⁻¹].
    pub phase_velocity: f64,
    /// Measured attenuation `α(ω)` \[Np·m⁻¹].
    pub attenuation: f64,
}

/// Recover the complex shear modulus `G*(ω)` \[Pa] from a measured phase
/// velocity and attenuation (model-agnostic rheological inversion).
///
/// Inverts the dispersion relation `k = ω√(ρ/G*)` used by the forward models.
/// On the physical branch `Im(k) < 0` for a dissipative `G* = μ + iωη_s`
/// (`α = |Im k|`), so the measured complex wavenumber is `k = ω/c_p − iα` and
/// `G* = ρ(ω/k)²`. For a lossless medium (`α = 0`) this returns the real
/// `ρ c_p²`. Returns `0` for non-positive `ω`, `c_p`, or `ρ`.
#[must_use]
pub fn recover_complex_modulus(
    omega: f64,
    phase_velocity: f64,
    attenuation: f64,
    density: f64,
) -> Complex64 {
    if omega <= 0.0 || phase_velocity <= 0.0 || density <= 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    let k = Complex64::new(omega / phase_velocity, -attenuation);
    let ratio = Complex64::new(omega, 0.0) / k;
    Complex64::new(density, 0.0) * ratio * ratio
}
