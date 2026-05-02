//! Analytical Green's-function solutions for diffusion-equation validation.
//!
//! Used by the regression test suite as a value-semantic reference against
//! the numerical PCG solver. Both solutions are documented in Contini et al.
//! (1997), "Photon migration through a turbid slab", *Applied Optics*.

use crate::domain::medium::properties::OpticalPropertyData;

/// Infinite medium Green's function solution.
///
/// Point source at origin in infinite homogeneous medium:
/// ```text
/// Φ(r) = (P₀ / (4π D r)) exp(-μ_eff r)
/// ```
/// where `μ_eff = √(3μₐ(μₐ + μₛ'))`.
#[allow(dead_code)]
pub fn infinite_medium_point_source(
    r: f64,
    source_power: f64,
    optical_properties: OpticalPropertyData,
) -> f64 {
    let mu_a = optical_properties.absorption_coefficient;
    let mu_s_prime = optical_properties.reduced_scattering();
    let d = 1.0 / (3.0 * (mu_a + mu_s_prime));

    let mu_eff = (3.0 * mu_a * (mu_a + mu_s_prime)).sqrt();

    if r < 1e-10 {
        return f64::INFINITY;
    }

    (source_power / (4.0 * std::f64::consts::PI * d * r)) * (-mu_eff * r).exp()
}

/// Semi-infinite medium solution (diffuse reflectance).
///
/// Extrapolated boundary source at depth `z₀ = 1/μ_tr`:
/// ```text
/// Φ(ρ, z) = (3P₀ μ_tr / (4π)) [exp(-μ_eff r₁)/r₁ - exp(-μ_eff r₂)/r₂]
/// ```
/// where `r₁ = √(ρ² + (z − z₀)²)` and `r₂ = √(ρ² + (z + z₀ + 4AD)²)`.
#[allow(dead_code)]
pub fn semi_infinite_medium(
    rho: f64,
    z: f64,
    source_power: f64,
    optical_properties: OpticalPropertyData,
    boundary_parameter: f64,
) -> f64 {
    let mu_a = optical_properties.absorption_coefficient;
    let mu_s_prime = optical_properties.reduced_scattering();
    let d = 1.0 / (3.0 * (mu_a + mu_s_prime));
    let mu_tr = mu_a + mu_s_prime;

    let mu_eff = (3.0 * mu_a * mu_tr).sqrt();
    let z0 = 1.0 / mu_tr;

    let r1 = (rho * rho + (z - z0) * (z - z0)).sqrt();
    let r2 = (rho * rho + (z + z0 + 4.0 * boundary_parameter * d).powi(2)).sqrt();

    let prefactor = 3.0 * source_power * mu_tr / (4.0 * std::f64::consts::PI);

    if r1 < 1e-10 && r2 < 1e-10 {
        return 0.0;
    }

    prefactor * ((-mu_eff * r1).exp() / r1.max(1e-10) - (-mu_eff * r2).exp() / r2.max(1e-10))
}
