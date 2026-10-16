//! The Rayleigh integral over planar apertures: quadrature specification and kernel.

use super::aperture::PlanarAperture;
use super::input::{invalid, validate_point};
use super::path::{RayleighLayer, RayleighPropagationPath};
use super::quadrature::gauss_legendre_unit;
use super::vector::{add, cross, dot, norm, scale, subtract};
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use std::f64::consts::TAU;

pub(super) const MAX_SURFACE_SAMPLES: usize = 1 << 16;

/// Disk-quadrature parameters coupled to a validated propagation path.
#[derive(Debug, Clone, PartialEq)]
pub struct RayleighIntegralSpec {
    path: RayleighPropagationPath,
    radial_order: usize,
    azimuthal_order: usize,
}

impl RayleighIntegralSpec {
    /// Construct a propagation specification.
    ///
    /// `radial_order` is the Gauss-Legendre order in normalized squared radius;
    /// `azimuthal_order` is the periodic trapezoidal order around each ring.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the wavenumber is finite and
    /// positive, attenuation is finite and non-negative, radial order is
    /// positive, and azimuthal order is at least three.
    pub fn new(
        wavenumber_rad_m: f64,
        attenuation_np_m: f64,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        Self::from_path(
            RayleighPropagationPath::homogeneous(wavenumber_rad_m, attenuation_np_m)?,
            radial_order,
            azimuthal_order,
        )
    }

    /// Construct an ordered straight-ray layered propagation specification.
    ///
    /// The final layer must be semi-infinite and every preceding layer must
    /// carry a finite thickness. Phase and attenuation integrate segmentwise;
    /// interface reflection and refraction are outside this approximation.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for an empty or structurally invalid
    /// layer sequence, or invalid quadrature work.
    pub fn layered(
        layers: Vec<RayleighLayer>,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        Self::from_path(
            RayleighPropagationPath::layered(layers)?,
            radial_order,
            azimuthal_order,
        )
    }

    /// Construct an integral specification from a validated propagation path.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for invalid or excessive quadrature
    /// work.
    pub fn from_path(
        path: RayleighPropagationPath,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        if radial_order == 0 {
            return Err(invalid("radial_order", radial_order.to_string(), "> 0"));
        }
        if azimuthal_order < 3 {
            return Err(invalid(
                "azimuthal_order",
                azimuthal_order.to_string(),
                ">= 3",
            ));
        }
        let surface_samples = radial_order.checked_mul(azimuthal_order).ok_or_else(|| {
            invalid(
                "quadrature_surface_samples",
                format!("{radial_order} * {azimuthal_order}"),
                "product representable as usize",
            )
        })?;
        if surface_samples > MAX_SURFACE_SAMPLES {
            return Err(invalid(
                "quadrature_surface_samples",
                surface_samples.to_string(),
                &format!("<= {MAX_SURFACE_SAMPLES}"),
            ));
        }
        Ok(Self {
            path,
            radial_order,
            azimuthal_order,
        })
    }

    /// Acoustic wavenumber in radians per metre.
    #[must_use]
    pub fn wavenumber_rad_m(&self) -> f64 {
        self.path.wavenumber_rad_m()
    }

    /// Amplitude attenuation coefficient in nepers per metre.
    #[must_use]
    pub fn attenuation_np_m(&self) -> f64 {
        self.path.attenuation_np_m()
    }
}

/// Evaluate the coherent complex pressure at every observation point.
///
/// Contributions are zero for a point on or behind a piston's baffle plane.
/// Attenuation is applied per source-to-observation path as `exp(-αR)`.
///
/// # Errors
///
/// Returns `KwaversError::Config` if an observation coordinate is non-finite
/// or the quadrature root solver fails to converge.
pub fn rayleigh_pressure(
    points_m: &[[f64; 3]],
    apertures: &[PlanarAperture],
    spec: &RayleighIntegralSpec,
) -> KwaversResult<Vec<Complex64>> {
    for &point in points_m {
        validate_point("observation_point_m", point)?;
    }
    let radial_rule = gauss_legendre_unit(spec.radial_order)?;
    let prefactor = Complex64::new(0.0, -spec.wavenumber_rad_m() / TAU);
    let mut pressure = vec![Complex64::new(0.0, 0.0); points_m.len()];

    for aperture in apertures {
        if aperture.surface_pressure_pa() == Complex64::new(0.0, 0.0) {
            continue;
        }
        let geometry = aperture.geometry();
        let bitangent = cross(geometry.normal(), geometry.first_axis());
        let (inner_radius, outer_radius, start_angle, span_angle) =
            geometry.shape().radial_and_angular_bounds();
        let squared_radius_span = outer_radius * outer_radius - inner_radius * inner_radius;
        let azimuthal_weight = 0.5 * squared_radius_span * span_angle / spec.azimuthal_order as f64;
        for (&point, total) in points_m.iter().zip(&mut pressure) {
            let center_offset = subtract(point, geometry.center_m());
            if dot(center_offset, geometry.normal()) <= 0.0 {
                continue;
            }
            let mut integral = Complex64::new(0.0, 0.0);
            for &(radial_fraction, radial_weight) in &radial_rule {
                let radius =
                    (inner_radius * inner_radius + radial_fraction * squared_radius_span).sqrt();
                let area_weight = azimuthal_weight * radial_weight;
                for azimuth_index in 0..spec.azimuthal_order {
                    let azimuth = start_angle
                        + span_angle * (azimuth_index as f64 + 0.5) / spec.azimuthal_order as f64;
                    let surface_point = add(
                        geometry.center_m(),
                        scale(
                            add(
                                scale(geometry.first_axis(), azimuth.cos()),
                                scale(bitangent, azimuth.sin()),
                            ),
                            radius,
                        ),
                    );
                    let range = norm(subtract(point, surface_point));
                    let (phase, attenuation) = spec.path.propagation_terms(range)?;
                    let amplitude = area_weight * (-attenuation).exp() / range;
                    integral += Complex64::from_polar(amplitude, phase);
                }
            }
            *total += prefactor * aperture.surface_pressure_pa() * integral;
        }
    }
    Ok(pressure)
}
