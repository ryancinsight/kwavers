//! The Rayleigh integral over planar apertures: quadrature specification and kernel.

use super::aperture::PlanarAperture;
use super::input::{invalid, validate_point};
use super::path::{RayleighLayer, RayleighPropagationPath};
use super::quadrature::surface_rule;
use super::vector::{add, cross, dot, norm, scale, subtract};
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use std::f64::consts::TAU;

pub(super) const MAX_SURFACE_SAMPLES: usize = 1 << 16;

/// Surface-quadrature orders coupled to a validated propagation path.
///
/// The two orders resolve the aperture surface in the directions its shape
/// defines (see [`super::PlanarApertureShape`]): radius and angle for a disk
/// or annular sector, width and height for a rectangle.
#[derive(Debug, Clone, PartialEq)]
pub struct RayleighIntegralSpec {
    path: RayleighPropagationPath,
    primary_order: usize,
    secondary_order: usize,
}

impl RayleighIntegralSpec {
    /// Construct a propagation specification.
    ///
    /// For a polar shape `primary_order` is the Gauss-Legendre order in
    /// normalized squared radius and `secondary_order` the periodic midpoint
    /// order around each ring. For a rectangle they are the Gauss-Legendre
    /// orders across its width and its height.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the wavenumber is finite and
    /// positive, attenuation is finite and non-negative, the primary order is
    /// positive, and the secondary order is at least three.
    pub fn new(
        wavenumber_rad_m: f64,
        attenuation_np_m: f64,
        primary_order: usize,
        secondary_order: usize,
    ) -> KwaversResult<Self> {
        Self::from_path(
            RayleighPropagationPath::homogeneous(wavenumber_rad_m, attenuation_np_m)?,
            primary_order,
            secondary_order,
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
        primary_order: usize,
        secondary_order: usize,
    ) -> KwaversResult<Self> {
        Self::from_path(
            RayleighPropagationPath::layered(layers)?,
            primary_order,
            secondary_order,
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
        primary_order: usize,
        secondary_order: usize,
    ) -> KwaversResult<Self> {
        if primary_order == 0 {
            return Err(invalid("primary_order", primary_order.to_string(), "> 0"));
        }
        if secondary_order < 3 {
            return Err(invalid(
                "secondary_order",
                secondary_order.to_string(),
                ">= 3",
            ));
        }
        let surface_samples = primary_order.checked_mul(secondary_order).ok_or_else(|| {
            invalid(
                "quadrature_surface_samples",
                format!("{primary_order} * {secondary_order}"),
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
            primary_order,
            secondary_order,
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
/// Contributions are zero for a point on or behind an aperture's baffle plane.
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
    let prefactor = Complex64::new(0.0, -spec.wavenumber_rad_m() / TAU);
    let mut pressure = vec![Complex64::new(0.0, 0.0); points_m.len()];

    for aperture in apertures {
        if aperture.surface_pressure_pa() == Complex64::new(0.0, 0.0) {
            continue;
        }
        let geometry = aperture.geometry();
        let first_axis = geometry.first_axis();
        let bitangent = cross(geometry.normal(), first_axis);
        let surface: Vec<([f64; 3], f64)> =
            surface_rule(geometry.shape(), spec.primary_order, spec.secondary_order)?
                .into_iter()
                .map(|([u, v], weight)| {
                    let offset = add(scale(first_axis, u), scale(bitangent, v));
                    (add(geometry.center_m(), offset), weight)
                })
                .collect();
        for (&point, total) in points_m.iter().zip(&mut pressure) {
            let center_offset = subtract(point, geometry.center_m());
            if dot(center_offset, geometry.normal()) <= 0.0 {
                continue;
            }
            let mut integral = Complex64::new(0.0, 0.0);
            for &(surface_point, weight) in &surface {
                let range = norm(subtract(point, surface_point));
                let (phase, attenuation) = spec.path.propagation_terms(range)?;
                integral += Complex64::from_polar(weight * (-attenuation).exp() / range, phase);
            }
            *total += prefactor * aperture.surface_pressure_pa() * integral;
        }
    }
    Ok(pressure)
}
