//! Analytical Green's-function solutions for diffusion-equation validation.
//!
//! Used by the regression test suite as a value-semantic reference against
//! the numerical PCG solver. Both solutions are documented in Contini et al.
//! (1997), "Photon migration through a turbid slab", *Applied Optics*.

#[cfg(test)]
use kwavers_domain::medium::properties::OpticalPropertyData;

/// Infinite medium Green's function solution.
///
/// Point source at origin in infinite homogeneous medium:
/// ```text
/// Φ(r) = (P₀ / (4π D r)) exp(-μ_eff r)
/// ```
/// where `μ_eff = √(3μₐ(μₐ + μₛ'))`.
#[cfg(test)]
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
