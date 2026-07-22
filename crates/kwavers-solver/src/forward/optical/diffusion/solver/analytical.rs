//! Analytical Green's-function solutions for diffusion-equation validation.
//!
//! Used by the regression test suite as a value-semantic reference against
//! the numerical PCG solver. Both solutions are documented in Contini et al.
//! (1997), "Photon migration through a turbid slab", *Applied Optics*.

#[cfg(test)]
use aequitas::systems::si::{quantities::Length, units::PerMeter};
#[cfg(test)]
use hyperion::quantity::PathLength;
#[cfg(test)]
use kwavers_medium::properties::OpticalPropertyData;

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
    let coefficients = optical_properties
        .diffusion_coefficients()
        .expect("fixture must define non-degenerate optical transport");
    let d = coefficients
        .diffusion_coefficient()
        .expect("fixture diffusion coefficient must be finite")
        .into_quantity()
        .into_base();
    let attenuation = coefficients
        .effective_attenuation()
        .expect("fixture effective attenuation must be finite");

    if r < 1e-10 {
        return f64::INFINITY;
    }

    let transmission = attenuation
        .optical_depth(
            PathLength::new(Length::from_base(r)).expect("fixture distance must be non-negative"),
        )
        .expect("fixture optical depth must be finite")
        .transmission()
        .into_quantity()
        .into_base();
    debug_assert!(attenuation.in_unit::<PerMeter>().is_finite());
    (source_power / (4.0 * std::f64::consts::PI * d * r)) * transmission
}
