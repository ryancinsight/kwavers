//! Frequency-domain radiation from oriented, rigidly baffled planar apertures.
//!
//! For prescribed normal surface velocity, the Rayleigh first integral is
//!
//! `p(x,ω) = -i ωρ/(2π) ∫_S v_n(s) exp(i k R)/R dS`.
//!
//! This module accepts the equivalent surface-pressure phasor
//! `p_s = ρ c v_n`, so its prefactor is `-i k/(2π)`. Each finite aperture is
//! integrated directly; applying a separate piston-directivity factor would
//! count the same aperture diffraction twice. The convention follows Qin et
//! al., *Ultrasonics* 51 (2011), Eq. 1,
//! <https://doi.org/10.1016/j.ultras.2010.12.011>.

use aequitas::systems::si::quantities::{Angle, Area, Length, ReciprocalLength};
use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use leto_ops::gauss_legendre_nodes_weights;
use std::f64::consts::TAU;

const MAX_SURFACE_SAMPLES: usize = 1 << 16;

/// Radial bounds and angular span of a planar aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanarApertureShape {
    /// Complete disk from the origin to `radius_m`.
    Disk { radius_m: Length },
    /// Annular sector measured counter-clockwise from the aperture's first axis.
    AnnularSector {
        inner_radius_m: Length,
        outer_radius_m: Length,
        start_angle: Angle,
        span_angle: Angle,
    },
}

impl PlanarApertureShape {
    pub(crate) fn radial_and_angular_bounds(self) -> (f64, f64, f64, f64) {
        match self {
            Self::Disk { radius_m } => (0.0, radius_m.into_base(), 0.0, TAU),
            Self::AnnularSector {
                inner_radius_m,
                outer_radius_m,
                start_angle,
                span_angle,
            } => (
                inner_radius_m.into_base(),
                outer_radius_m.into_base(),
                start_angle.into_base(),
                span_angle.into_base(),
            ),
        }
    }

    /// Exact planar area in square metres.
    #[must_use]
    pub fn area(self) -> Area {
        let (inner, outer, _, span) = self.radial_and_angular_bounds();
        Area::from_base(0.5 * (outer * outer - inner * inner) * span)
    }
}

/// Validated position, orientation, and radial/angular bounds of a planar aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarApertureGeometry {
    center_m: [Length; 3],
    normal: [f64; 3],
    first_axis: [f64; 3],
    shape: PlanarApertureShape,
}

impl PlanarApertureGeometry {
    /// Construct complete circular geometry.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for non-finite geometry, a non-positive
    /// radius, or a zero normal.
    pub fn disk(center_m: [Length; 3], normal: [f64; 3], radius_m: Length) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        let (first_axis, _) = plane_basis(normal);
        Self::from_validated(
            center_m,
            normal,
            first_axis,
            PlanarApertureShape::Disk { radius_m },
        )
    }

    /// Construct a planar aperture with an explicit in-plane angular origin.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the shape bounds are valid and
    /// the first axis has a nonzero projection into the aperture plane.
    pub fn oriented(
        center_m: [Length; 3],
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
    ) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        validate_point("first_axis", first_axis)?;
        let planar_axis = subtract(first_axis, scale(normal, dot(first_axis, normal)));
        let axis_norm = norm(planar_axis);
        validate_positive("first_axis_planar_norm", axis_norm)?;
        Self::from_validated(center_m, normal, scale(planar_axis, 1.0 / axis_norm), shape)
    }

    fn from_validated(
        center_m: [Length; 3],
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
    ) -> KwaversResult<Self> {
        validate_point("center_m", center_m.map(Length::into_base))?;
        match shape {
            PlanarApertureShape::Disk { radius_m } => {
                validate_positive("radius_m", radius_m.into_base())?
            }
            PlanarApertureShape::AnnularSector {
                inner_radius_m,
                outer_radius_m,
                start_angle,
                span_angle,
            } => {
                let inner_radius_m = inner_radius_m.into_base();
                let outer_radius_m = outer_radius_m.into_base();
                let start_angle_rad = start_angle.into_base();
                let span_angle_rad = span_angle.into_base();
                if !inner_radius_m.is_finite() || inner_radius_m < 0.0 {
                    return Err(invalid(
                        "inner_radius_m",
                        inner_radius_m.to_string(),
                        "finite and >= 0",
                    ));
                }
                validate_positive("outer_radius_m", outer_radius_m)?;
                if inner_radius_m >= outer_radius_m {
                    return Err(invalid(
                        "annular_radii_m",
                        format!("{inner_radius_m}..{outer_radius_m}"),
                        "inner < outer",
                    ));
                }
                if !start_angle_rad.is_finite() {
                    return Err(invalid(
                        "start_angle_rad",
                        start_angle_rad.to_string(),
                        "finite",
                    ));
                }
                if !(span_angle_rad.is_finite() && span_angle_rad > 0.0 && span_angle_rad <= TAU) {
                    return Err(invalid(
                        "span_angle_rad",
                        span_angle_rad.to_string(),
                        "0 < span <= 2*pi",
                    ));
                }
            }
        }
        Ok(Self {
            center_m,
            normal,
            first_axis,
            shape,
        })
    }

    /// Piston centre in metres.
    #[must_use]
    pub const fn center(&self) -> [Length; 3] {
        self.center_m
    }

    /// Unit normal pointing into the radiating half-space.
    #[must_use]
    pub const fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// Unit in-plane axis from which positive aperture angles are measured.
    #[must_use]
    pub const fn first_axis(&self) -> [f64; 3] {
        self.first_axis
    }

    /// Aperture shape and bounds.
    #[must_use]
    pub const fn shape(&self) -> PlanarApertureShape {
        self.shape
    }

    /// Outer aperture radius in metres.
    #[must_use]
    pub fn outer_radius(&self) -> Length {
        Length::from_base(self.shape.radial_and_angular_bounds().1)
    }
}

/// Uniformly driven planar aperture embedded in an infinite rigid baffle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarAperture {
    geometry: PlanarApertureGeometry,
    surface_pressure_pa: Complex64,
}

impl PlanarAperture {
    /// Construct a complete circular aperture.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for invalid geometry or pressure.
    pub fn disk(
        center_m: [Length; 3],
        normal: [f64; 3],
        radius_m: Length,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        Self::new(
            PlanarApertureGeometry::disk(center_m, normal, radius_m)?,
            surface_pressure_pa,
        )
    }

    /// Construct an oriented aperture with an explicit angular origin.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for invalid geometry or pressure.
    pub fn oriented(
        center_m: [Length; 3],
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        Self::new(
            PlanarApertureGeometry::oriented(center_m, normal, first_axis, shape)?,
            surface_pressure_pa,
        )
    }

    /// Attach a prescribed pressure phasor to validated geometry.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when either phasor component is non-finite.
    pub fn new(
        geometry: PlanarApertureGeometry,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        if !surface_pressure_pa.re.is_finite() || !surface_pressure_pa.im.is_finite() {
            return Err(invalid(
                "surface_pressure_pa",
                format!("{surface_pressure_pa:?}"),
                "finite",
            ));
        }
        Ok(Self {
            geometry,
            surface_pressure_pa,
        })
    }

    /// Validated aperture geometry.
    #[must_use]
    pub const fn geometry(&self) -> PlanarApertureGeometry {
        self.geometry
    }

    /// Piston centre in metres.
    #[must_use]
    pub const fn center(&self) -> [Length; 3] {
        self.geometry.center()
    }

    /// Unit normal pointing into the radiating half-space.
    #[must_use]
    pub const fn normal(&self) -> [f64; 3] {
        self.geometry.normal()
    }

    /// Aperture shape and bounds.
    #[must_use]
    pub const fn shape(&self) -> PlanarApertureShape {
        self.geometry.shape()
    }

    /// Outer aperture radius in metres.
    #[must_use]
    pub fn outer_radius(&self) -> Length {
        self.geometry.outer_radius()
    }

    /// Complex surface-pressure phasor in pascals.
    #[must_use]
    pub const fn surface_pressure_pa(&self) -> Complex64 {
        self.surface_pressure_pa
    }
}

/// Disk-quadrature parameters coupled to a validated propagation path.
#[derive(Debug, Clone, PartialEq)]
pub struct RayleighIntegralSpec {
    path: RayleighPropagationPath,
    radial_order: usize,
    azimuthal_order: usize,
}

/// One straight-ray propagation segment, ordered outward from the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighLayer {
    wavenumber: ReciprocalLength,
    attenuation: ReciprocalLength,
    thickness: Option<Length>,
}

/// Validated straight-ray acoustic propagation path.
///
/// Finite layers consume their configured thickness in order; the final
/// semi-infinite layer receives the remaining distance. The contract models
/// phase and attenuation only: interface reflection and refraction remain
/// outside the straight-ray Rayleigh approximation.
#[derive(Debug, Clone, PartialEq)]
pub struct RayleighPropagationPath {
    layers: Vec<RayleighLayer>,
}

impl RayleighLayer {
    /// Construct a propagation segment; `None` thickness denotes the final
    /// semi-infinite layer.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for non-finite coefficients, a
    /// non-positive wavenumber, negative attenuation, or non-positive finite
    /// thickness.
    pub fn new(
        wavenumber: ReciprocalLength,
        attenuation: ReciprocalLength,
        thickness: Option<Length>,
    ) -> KwaversResult<Self> {
        validate_positive("wavenumber", wavenumber.into_base())?;
        let attenuation_np_m = attenuation.into_base();
        if !attenuation_np_m.is_finite() || attenuation_np_m < 0.0 {
            return Err(invalid(
                "attenuation",
                attenuation_np_m.to_string(),
                "finite and >= 0",
            ));
        }
        if let Some(thickness) = thickness {
            validate_positive("layer_thickness", thickness.into_base())?;
        }
        Ok(Self {
            wavenumber,
            attenuation,
            thickness,
        })
    }
}

impl RayleighPropagationPath {
    /// Construct a homogeneous semi-infinite propagation path.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the wavenumber is finite and
    /// positive and attenuation is finite and non-negative.
    pub fn homogeneous(
        wavenumber: ReciprocalLength,
        attenuation: ReciprocalLength,
    ) -> KwaversResult<Self> {
        Self::layered(vec![RayleighLayer::new(wavenumber, attenuation, None)?])
    }

    /// Construct an ordered straight-ray layered propagation path.
    ///
    /// The final layer must be semi-infinite and every preceding layer must
    /// carry a finite thickness.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for an empty or structurally invalid
    /// layer sequence.
    pub fn layered(layers: Vec<RayleighLayer>) -> KwaversResult<Self> {
        if layers.is_empty() {
            return Err(invalid("layers", "0".to_owned(), "at least one layer"));
        }
        if layers.last().is_some_and(|layer| layer.thickness.is_some()) {
            return Err(invalid(
                "layers",
                "finite final layer".to_owned(),
                "final layer is semi-infinite",
            ));
        }
        if layers[..layers.len() - 1]
            .iter()
            .any(|layer| layer.thickness.is_none())
        {
            return Err(invalid(
                "layers",
                "non-final semi-infinite layer".to_owned(),
                "only final layer is semi-infinite",
            ));
        }
        Ok(Self { layers })
    }

    /// Integrate phase and amplitude attenuation along one straight-ray path.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when `range_m` is not finite or is
    /// negative.
    pub fn propagation_terms(&self, range: Length) -> KwaversResult<(f64, f64)> {
        let range_m = range.into_base();
        if !range_m.is_finite() || range_m < 0.0 {
            return Err(invalid("range_m", range_m.to_string(), "finite and >= 0"));
        }
        let mut remaining = range_m;
        let mut phase = 0.0;
        let mut attenuation = 0.0;
        for layer in &self.layers {
            let segment = layer
                .thickness
                .map_or(remaining, |thickness| remaining.min(thickness.into_base()));
            phase = layer.wavenumber.into_base().mul_add(segment, phase);
            attenuation = layer.attenuation.into_base().mul_add(segment, attenuation);
            remaining -= segment;
            if remaining <= 0.0 {
                break;
            }
        }
        Ok((phase, attenuation))
    }

    fn wavenumber_rad_m(&self) -> ReciprocalLength {
        self.layers[0].wavenumber
    }

    fn attenuation_np_m(&self) -> ReciprocalLength {
        self.layers[0].attenuation
    }
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
        wavenumber: ReciprocalLength,
        attenuation: ReciprocalLength,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        Self::from_path(
            RayleighPropagationPath::homogeneous(wavenumber, attenuation)?,
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
    pub fn wavenumber(&self) -> ReciprocalLength {
        self.path.wavenumber_rad_m()
    }

    /// Amplitude attenuation coefficient in nepers per metre.
    #[must_use]
    pub fn attenuation(&self) -> ReciprocalLength {
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
    points: &[[Length; 3]],
    apertures: &[PlanarAperture],
    spec: &RayleighIntegralSpec,
) -> KwaversResult<Vec<Complex64>> {
    for &point in points {
        validate_point("observation_point", point.map(Length::into_base))?;
    }
    let radial_rule = gauss_legendre_unit(spec.radial_order)?;
    let prefactor = Complex64::new(0.0, -spec.wavenumber().into_base() / TAU);
    let mut pressure = vec![Complex64::new(0.0, 0.0); points.len()];

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
        let center = geometry.center().map(Length::into_base);
        for (&point, total) in points.iter().zip(&mut pressure) {
            let point = point.map(Length::into_base);
            let center_offset = subtract(point, center);
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
                        center,
                        scale(
                            add(
                                scale(geometry.first_axis(), azimuth.cos()),
                                scale(bitangent, azimuth.sin()),
                            ),
                            radius,
                        ),
                    );
                    let range_m = norm(subtract(point, surface_point));
                    validate_positive("source_observation_range", range_m)?;
                    let range = Length::from_base(range_m);
                    let (phase, attenuation) = spec.path.propagation_terms(range)?;
                    let amplitude = area_weight * (-attenuation).exp() / range_m;
                    integral += Complex64::from_polar(amplitude, phase);
                }
            }
            *total += prefactor * aperture.surface_pressure_pa() * integral;
        }
    }
    Ok(pressure)
}

/// Gauss-Legendre quadrature nodes/weights on [0, 1] (mapped from [−1, 1]).
///
/// Delegates to `leto_ops::gauss_legendre_nodes_weights` for the SSOT computation,
/// then maps the standard [-1,1] rule to [0,1]: node x → (1-x)/2, weight w → w/2.
fn gauss_legendre_unit(order: usize) -> KwaversResult<Vec<(f64, f64)>> {
    let (nodes, weights) = gauss_legendre_nodes_weights(order)
        .map_err(|msg| invalid("radial_order", order.to_string(), &msg))?;
    // Map [-1, 1] → [0, 1]: u = (1 - x) / 2, w_unit = w / 2
    Ok(nodes
        .into_iter()
        .zip(weights)
        .map(|(x, w)| (0.5 * (1.0 - x), 0.5 * w))
        .collect())
}

fn plane_basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = unit(subtract(seed, scale(normal, dot(seed, normal))));
    (tangent, cross(normal, tangent))
}

fn normalized_normal(normal: [f64; 3]) -> KwaversResult<[f64; 3]> {
    validate_point("normal", normal)?;
    let normal_norm = norm(normal);
    validate_positive("normal_norm", normal_norm)?;
    Ok(scale(normal, 1.0 / normal_norm))
}

fn validate_positive(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(parameter, value.to_string(), "finite and > 0"))
    }
}

fn validate_point(parameter: &str, point: [f64; 3]) -> KwaversResult<()> {
    if point.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(())
    } else {
        Err(invalid(
            parameter,
            format!("{point:?}"),
            "all coordinates finite",
        ))
    }
}

fn invalid(parameter: &str, value: String, constraint: &str) -> KwaversError {
    KwaversError::Config(ConfigError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        constraint: constraint.to_owned(),
    })
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|value| value * factor)
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}
fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}
fn unit(vector: [f64; 3]) -> [f64; 3] {
    scale(vector, 1.0 / norm(vector))
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto_ops::application::special::j1;

    fn length(value: f64) -> Length {
        Length::from_base(value)
    }

    fn point(x: f64, y: f64, z: f64) -> [Length; 3] {
        [length(x), length(y), length(z)]
    }

    fn wavenumber(value: f64) -> ReciprocalLength {
        ReciprocalLength::from_base(value)
    }

    fn piston(radius_m: f64) -> PlanarAperture {
        PlanarAperture::disk(
            [length(0.0); 3],
            [0.0, 0.0, 1.0],
            length(radius_m),
            Complex64::new(2.5e5, 0.0),
        )
        .unwrap()
    }

    #[test]
    fn quadrature_integrates_disk_area_exactly() {
        let rule = gauss_legendre_unit(7).unwrap();
        let radius = 2.3e-3;
        let area = PI * radius * radius * rule.iter().map(|(_, weight)| weight).sum::<f64>();
        let expected = PI * radius * radius;
        assert!((area - expected).abs() <= 8.0 * f64::EPSILON * expected);
    }

    #[test]
    fn annular_sector_area_matches_radial_angular_measure() {
        let shape = PlanarApertureShape::AnnularSector {
            inner_radius_m: length(1.0e-3),
            outer_radius_m: length(3.0e-3),
            start_angle: Angle::from_base(0.3),
            span_angle: Angle::from_base(std::f64::consts::FRAC_PI_2),
        };
        let expected = std::f64::consts::FRAC_PI_2 * (9.0e-6 - 1.0e-6) / 2.0;
        assert!((shape.area().into_base() - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }

    #[test]
    fn independently_driven_sectors_superpose_to_complete_annulus_on_axis() {
        let pressure = Complex64::from_polar(1.7e5, 0.4);
        let aperture = |start_angle: Angle, span_angle: Angle| {
            PlanarAperture::oriented(
                [length(0.0); 3],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                PlanarApertureShape::AnnularSector {
                    inner_radius_m: length(0.4e-3),
                    outer_radius_m: length(1.2e-3),
                    start_angle,
                    span_angle,
                },
                pressure,
            )
            .unwrap()
        };
        let complete = aperture(Angle::from_base(0.0), Angle::from_base(TAU));
        let sectors: Vec<_> = (0..4)
            .map(|sector| {
                aperture(
                    Angle::from_base(sector as f64 * std::f64::consts::FRAC_PI_2),
                    Angle::from_base(std::f64::consts::FRAC_PI_2),
                )
            })
            .collect();
        let spec = RayleighIntegralSpec::new(
            wavenumber(TAU / 0.7e-3),
            ReciprocalLength::from_base(0.0),
            12,
            32,
        )
        .unwrap();
        let point = [point(0.0, 0.0, 20.0e-3)];
        let complete_pressure = rayleigh_pressure(&point, &[complete], &spec).unwrap()[0];
        let sector_pressure = rayleigh_pressure(&point, &sectors, &spec).unwrap()[0];
        assert!(
            (complete_pressure - sector_pressure).norm()
                <= 32.0 * f64::EPSILON * complete_pressure.norm(),
            "complete={complete_pressure:?}, sectors={sector_pressure:?}"
        );
    }

    #[test]
    fn on_axis_pressure_matches_closed_form() {
        let piston = piston(1.2e-3);
        let k = TAU / 0.75e-3;
        let axial_range = 14.0e-3;
        let spec =
            RayleighIntegralSpec::new(wavenumber(k), ReciprocalLength::from_base(0.0), 24, 24)
                .unwrap();
        let actual =
            rayleigh_pressure(&[point(0.0, 0.0, axial_range)], &[piston], &spec).unwrap()[0];
        let rim_range = axial_range.hypot(piston.outer_radius().into_base());
        let expected = piston.surface_pressure_pa()
            * (Complex64::from_polar(1.0, k * axial_range)
                - Complex64::from_polar(1.0, k * rim_range));
        let relative = (actual - expected).norm() / expected.norm();
        assert!(relative <= 2.0e-12, "relative complex error {relative:e}");
    }

    #[test]
    fn far_field_ratio_matches_circular_piston_directivity() {
        let piston = piston(0.8e-3);
        let k = TAU / 0.6e-3;
        let range = 0.4;
        let angle: f64 = 0.17;
        let points = [
            point(0.0, 0.0, range),
            point(range * angle.sin(), 0.0, range * angle.cos()),
        ];
        let pressure = rayleigh_pressure(
            &points,
            &[piston],
            &RayleighIntegralSpec::new(wavenumber(k), ReciprocalLength::from_base(0.0), 20, 96)
                .unwrap(),
        )
        .unwrap();
        let argument = k * piston.outer_radius().into_base() * angle.sin();
        let expected_ratio = (2.0 * j1(argument) / argument).abs();
        let actual_ratio = pressure[1].norm() / pressure[0].norm();
        assert!(
            (actual_ratio - expected_ratio).abs() <= 2.0e-4,
            "ratio {actual_ratio:e}, far-field oracle {expected_ratio:e}"
        );
    }

    #[test]
    fn rotation_preserves_complex_pressure() {
        let base = piston(0.7e-3);
        let rotated = PlanarAperture::disk(
            [length(0.0); 3],
            [1.0, 0.0, 0.0],
            base.outer_radius(),
            base.surface_pressure_pa(),
        )
        .unwrap();
        let spec = RayleighIntegralSpec::new(
            wavenumber(TAU / 0.8e-3),
            ReciprocalLength::from_base(3.0),
            12,
            48,
        )
        .unwrap();
        let along_z = rayleigh_pressure(&[point(0.2e-3, 0.0, 30.0e-3)], &[base], &spec).unwrap()[0];
        let along_x =
            rayleigh_pressure(&[point(30.0e-3, 0.0, 0.2e-3)], &[rotated], &spec).unwrap()[0];
        assert!((along_z - along_x).norm() <= 64.0 * f64::EPSILON * along_z.norm());
    }

    #[test]
    fn rigid_baffle_suppresses_back_radiation() {
        let pressure = rayleigh_pressure(
            &[point(0.0, 0.0, -0.01)],
            &[piston(0.5e-3)],
            &RayleighIntegralSpec::new(
                wavenumber(TAU / 1.0e-3),
                ReciprocalLength::from_base(0.0),
                4,
                12,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(pressure, vec![Complex64::new(0.0, 0.0)]);
    }

    #[test]
    fn quadrature_work_is_bounded() {
        let error = RayleighIntegralSpec::new(
            wavenumber(1.0),
            ReciprocalLength::from_base(0.0),
            MAX_SURFACE_SAMPLES,
            3,
        )
        .expect_err("surface sample count exceeds the provider budget");
        assert!(error.to_string().contains("quadrature_surface_samples"));
    }

    #[test]
    fn layered_path_integrates_each_segment_exactly() {
        let path = RayleighPropagationPath::layered(vec![
            RayleighLayer::new(
                wavenumber(2.0),
                ReciprocalLength::from_base(3.0),
                Some(length(0.25)),
            )
            .unwrap(),
            RayleighLayer::new(wavenumber(5.0), ReciprocalLength::from_base(7.0), None).unwrap(),
        ])
        .unwrap();
        let (short_phase, short_attenuation) = path.propagation_terms(length(0.1)).unwrap();
        assert!((short_phase - 0.2).abs() <= f64::EPSILON * 0.2);
        assert!((short_attenuation - 0.3).abs() <= f64::EPSILON * 0.3);
        let (long_phase, long_attenuation) = path.propagation_terms(length(1.0)).unwrap();
        assert!((long_phase - 4.25).abs() <= 2.0 * f64::EPSILON * 4.25);
        assert!((long_attenuation - 6.0).abs() <= 2.0 * f64::EPSILON * 6.0);
    }

    #[test]
    fn layered_path_requires_one_final_half_space() {
        let finite = RayleighLayer::new(
            wavenumber(2.0),
            ReciprocalLength::from_base(0.0),
            Some(length(0.25)),
        )
        .unwrap();
        let half_space =
            RayleighLayer::new(wavenumber(3.0), ReciprocalLength::from_base(0.0), None).unwrap();
        let final_half_space =
            RayleighLayer::new(wavenumber(4.0), ReciprocalLength::from_base(0.0), None).unwrap();
        let finite_final = RayleighPropagationPath::layered(vec![finite])
            .expect_err("a finite final layer has no propagation half-space");
        let non_final_half_space =
            RayleighPropagationPath::layered(vec![half_space, final_half_space])
                .expect_err("a semi-infinite layer cannot precede another layer");
        for (error, value, constraint) in [
            (
                finite_final,
                "finite final layer",
                "final layer is semi-infinite",
            ),
            (
                non_final_half_space,
                "non-final semi-infinite layer",
                "only final layer is semi-infinite",
            ),
        ] {
            match error {
                KwaversError::Config(ConfigError::InvalidValue {
                    parameter,
                    value: actual_value,
                    constraint: actual_constraint,
                }) => {
                    assert_eq!(parameter, "layers");
                    assert_eq!(actual_value, value);
                    assert_eq!(actual_constraint, constraint);
                }
                other => panic!("expected layered-path configuration error, got {other:?}"),
            }
        }
    }

    #[test]
    fn propagation_terms_reject_invalid_range() {
        let path =
            RayleighPropagationPath::homogeneous(wavenumber(1.0), ReciprocalLength::from_base(0.0))
                .unwrap();
        for range_m in [-1.0, f64::NAN] {
            let error = path
                .propagation_terms(length(range_m))
                .expect_err("invalid propagation range must be rejected");
            match error {
                KwaversError::Config(ConfigError::InvalidValue {
                    parameter,
                    constraint,
                    ..
                }) => {
                    assert_eq!(parameter, "range_m");
                    assert_eq!(constraint, "finite and >= 0");
                }
                other => panic!("expected invalid range configuration error, got {other:?}"),
            }
        }
    }
}
