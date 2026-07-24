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

use aequitas::systems::si::quantities::{Area, Length, ReciprocalLength};
use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use std::f64::consts::{PI, TAU};

const LEGENDRE_ROOT_STEPS: usize = 64;
const LEGENDRE_ROOT_TOLERANCE: f64 = 8.0 * f64::EPSILON;
const MAX_SURFACE_SAMPLES: usize = 1 << 16;

/// A validated Cartesian position whose three components carry Aequitas
/// lengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianPosition {
    components: [Length; 3],
}

impl CartesianPosition {
    /// Construct a position from SI base-unit metres.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when any coordinate is non-finite.
    pub fn from_base(components: [f64; 3]) -> KwaversResult<Self> {
        if components.iter().all(|component| component.is_finite()) {
            Ok(Self {
                components: components.map(Length::from_base),
            })
        } else {
            Err(invalid(
                "position",
                format!("{components:?}"),
                "all coordinates finite",
            ))
        }
    }

    /// Construct a position from typed length components.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when any coordinate is non-finite.
    pub fn new(components: [Length; 3]) -> KwaversResult<Self> {
        Self::from_base(components.map(Length::into_base))
    }

    /// Return the typed Cartesian components.
    #[must_use]
    pub const fn components(self) -> [Length; 3] {
        self.components
    }

    /// Convert to the scalar representation required by legacy grid and
    /// numerical-kernel boundaries.
    #[must_use]
    pub fn into_base(self) -> [f64; 3] {
        self.components.map(Length::into_base)
    }
}

/// Radial bounds and angular span of a planar aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanarApertureShape {
    /// Complete disk from the origin to `radius`.
    Disk { radius: Length },
    /// Annular sector measured counter-clockwise from the aperture's first axis.
    AnnularSector {
        inner_radius: Length,
        outer_radius: Length,
        start_angle_rad: f64,
        span_angle_rad: f64,
    },
}

impl PlanarApertureShape {
    pub(crate) fn radial_and_angular_bounds(self) -> (Length, Length, f64, f64) {
        match self {
            Self::Disk { radius } => (Length::from_base(0.0), radius, 0.0, TAU),
            Self::AnnularSector {
                inner_radius,
                outer_radius,
                start_angle_rad,
                span_angle_rad,
            } => (inner_radius, outer_radius, start_angle_rad, span_angle_rad),
        }
    }

    /// Exact planar area.
    #[must_use]
    pub fn area(self) -> Area {
        let (inner, outer, _, span) = self.radial_and_angular_bounds();
        (outer * outer - inner * inner) * (0.5 * span)
    }
}

/// Validated position, orientation, and radial/angular bounds of a planar aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarApertureGeometry {
    center: CartesianPosition,
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
    pub fn disk(
        center: CartesianPosition,
        normal: [f64; 3],
        radius: Length,
    ) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        let (first_axis, _) = plane_basis(normal);
        Self::from_validated(
            center,
            normal,
            first_axis,
            PlanarApertureShape::Disk { radius },
        )
    }

    /// Construct a planar aperture with an explicit in-plane angular origin.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the shape bounds are valid and
    /// the first axis has a nonzero projection into the aperture plane.
    pub fn oriented(
        center: CartesianPosition,
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
    ) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        validate_point("first_axis", first_axis)?;
        let planar_axis = subtract(first_axis, scale(normal, dot(first_axis, normal)));
        let axis_norm = norm(planar_axis);
        validate_positive("first_axis_planar_norm", axis_norm)?;
        Self::from_validated(center, normal, scale(planar_axis, 1.0 / axis_norm), shape)
    }

    fn from_validated(
        center: CartesianPosition,
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
    ) -> KwaversResult<Self> {
        validate_position("center", center)?;
        match shape {
            PlanarApertureShape::Disk { radius } => {
                validate_positive("radius", radius.into_base())?;
            }
            PlanarApertureShape::AnnularSector {
                inner_radius,
                outer_radius,
                start_angle_rad,
                span_angle_rad,
            } => {
                let inner_radius_m = inner_radius.into_base();
                let outer_radius_m = outer_radius.into_base();
                if !inner_radius_m.is_finite() || inner_radius_m < 0.0 {
                    return Err(invalid(
                        "inner_radius",
                        inner_radius_m.to_string(),
                        "finite and >= 0",
                    ));
                }
                validate_positive("outer_radius", outer_radius_m)?;
                if inner_radius_m >= outer_radius_m {
                    return Err(invalid(
                        "annular_radii",
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
            center,
            normal,
            first_axis,
            shape,
        })
    }

    /// Piston centre.
    #[must_use]
    pub const fn center(&self) -> CartesianPosition {
        self.center
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

    /// Outer aperture radius.
    #[must_use]
    pub fn outer_radius(&self) -> Length {
        self.shape.radial_and_angular_bounds().1
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
        center: CartesianPosition,
        normal: [f64; 3],
        radius: Length,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        Self::new(
            PlanarApertureGeometry::disk(center, normal, radius)?,
            surface_pressure_pa,
        )
    }

    /// Construct an oriented aperture with an explicit angular origin.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for invalid geometry or pressure.
    pub fn oriented(
        center: CartesianPosition,
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        Self::new(
            PlanarApertureGeometry::oriented(center, normal, first_axis, shape)?,
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

    /// Piston centre.
    #[must_use]
    pub const fn center(&self) -> CartesianPosition {
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

    /// Outer aperture radius.
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
        validate_positive("wavenumber_rad_m", wavenumber.into_base())?;
        let attenuation_base = attenuation.into_base();
        if !attenuation_base.is_finite() || attenuation_base < 0.0 {
            return Err(invalid(
                "attenuation_np_m",
                attenuation_base.to_string(),
                "finite and >= 0",
            ));
        }
        if let Some(thickness) = thickness {
            validate_positive("layer_thickness_m", thickness.into_base())?;
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

    fn wavenumber(&self) -> ReciprocalLength {
        self.layers[0].wavenumber
    }

    fn attenuation(&self) -> ReciprocalLength {
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
        self.path.wavenumber()
    }

    /// Amplitude attenuation coefficient in nepers per metre.
    #[must_use]
    pub fn attenuation(&self) -> ReciprocalLength {
        self.path.attenuation()
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
    points: &[CartesianPosition],
    apertures: &[PlanarAperture],
    spec: &RayleighIntegralSpec,
) -> KwaversResult<Vec<Complex64>> {
    for &point in points {
        validate_position("observation_point", point)?;
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
        let inner_radius = inner_radius.into_base();
        let outer_radius = outer_radius.into_base();
        let center = geometry.center().into_base();
        let squared_radius_span = outer_radius * outer_radius - inner_radius * inner_radius;
        let azimuthal_weight = 0.5 * squared_radius_span * span_angle / spec.azimuthal_order as f64;
        for (&point, total) in points.iter().zip(&mut pressure) {
            let point = point.into_base();
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
                    let range = norm(subtract(point, surface_point));
                    let (phase, attenuation) =
                        spec.path.propagation_terms(Length::from_base(range))?;
                    let amplitude = area_weight * (-attenuation).exp() / range;
                    integral += Complex64::from_polar(amplitude, phase);
                }
            }
            *total += prefactor * aperture.surface_pressure_pa() * integral;
        }
    }
    Ok(pressure)
}

fn gauss_legendre_unit(order: usize) -> KwaversResult<Vec<(f64, f64)>> {
    let mut rule = vec![(0.0, 0.0); order];
    let paired_roots = order.div_ceil(2);
    for root_index in 0..paired_roots {
        let mut root = (PI * (root_index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..LEGENDRE_ROOT_STEPS {
            let (polynomial, previous) = legendre_pair(order, root);
            let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
            let next = root - polynomial / derivative;
            if (next - root).abs() <= LEGENDRE_ROOT_TOLERANCE * next.abs().max(1.0) {
                root = next;
                converged = true;
                break;
            }
            root = next;
        }
        if !converged {
            return Err(invalid(
                "radial_order",
                order.to_string(),
                "Gauss-Legendre roots converge",
            ));
        }
        let (polynomial, previous) = legendre_pair(order, root);
        let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
        let weight = 1.0 / ((1.0 - root * root) * derivative * derivative);
        let lower = root_index;
        let upper = order - 1 - root_index;
        rule[lower] = (0.5 * (1.0 - root), weight);
        rule[upper] = (0.5 * (1.0 + root), weight);
    }
    Ok(rule)
}

fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if order == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for degree in 2..=order {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    (current, previous)
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

fn validate_position(parameter: &str, position: CartesianPosition) -> KwaversResult<()> {
    validate_point(parameter, position.into_base())
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

    fn inverse_m(value: f64) -> ReciprocalLength {
        ReciprocalLength::from_base(value)
    }

    fn length_m(value: f64) -> Length {
        Length::from_base(value)
    }

    fn position(value: [f64; 3]) -> CartesianPosition {
        CartesianPosition::from_base(value).unwrap()
    }
    use kwavers_math::special::bessel::j1;

    fn piston(radius_m: f64) -> PlanarAperture {
        PlanarAperture::disk(
            position([0.0; 3]),
            [0.0, 0.0, 1.0],
            length_m(radius_m),
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
            inner_radius: length_m(1.0e-3),
            outer_radius: length_m(3.0e-3),
            start_angle_rad: 0.3,
            span_angle_rad: std::f64::consts::FRAC_PI_2,
        };
        let expected = std::f64::consts::FRAC_PI_2 * (9.0e-6 - 1.0e-6) / 2.0;
        assert!((shape.area().into_base() - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }

    #[test]
    fn independently_driven_sectors_superpose_to_complete_annulus_on_axis() {
        let pressure = Complex64::from_polar(1.7e5, 0.4);
        let aperture = |start_angle_rad, span_angle_rad| {
            PlanarAperture::oriented(
                position([0.0; 3]),
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                PlanarApertureShape::AnnularSector {
                    inner_radius: length_m(0.4e-3),
                    outer_radius: length_m(1.2e-3),
                    start_angle_rad,
                    span_angle_rad,
                },
                pressure,
            )
            .unwrap()
        };
        let complete = aperture(0.0, TAU);
        let sectors: Vec<_> = (0..4)
            .map(|sector| {
                aperture(
                    sector as f64 * std::f64::consts::FRAC_PI_2,
                    std::f64::consts::FRAC_PI_2,
                )
            })
            .collect();
        let spec =
            RayleighIntegralSpec::new(inverse_m(TAU / 0.7e-3), inverse_m(0.0), 12, 32).unwrap();
        let point = [position([0.0, 0.0, 20.0e-3])];
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
        let spec = RayleighIntegralSpec::new(inverse_m(k), inverse_m(0.0), 24, 24).unwrap();
        let actual =
            rayleigh_pressure(&[position([0.0, 0.0, axial_range])], &[piston], &spec).unwrap()[0];
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
            position([0.0, 0.0, range]),
            position([range * angle.sin(), 0.0, range * angle.cos()]),
        ];
        let pressure = rayleigh_pressure(
            &points,
            &[piston],
            &RayleighIntegralSpec::new(inverse_m(k), inverse_m(0.0), 20, 96).unwrap(),
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
            position([0.0; 3]),
            [1.0, 0.0, 0.0],
            base.outer_radius(),
            base.surface_pressure_pa(),
        )
        .unwrap();
        let spec =
            RayleighIntegralSpec::new(inverse_m(TAU / 0.8e-3), inverse_m(3.0), 12, 48).unwrap();
        let along_z =
            rayleigh_pressure(&[position([0.2e-3, 0.0, 30.0e-3])], &[base], &spec).unwrap()[0];
        let along_x =
            rayleigh_pressure(&[position([30.0e-3, 0.0, 0.2e-3])], &[rotated], &spec).unwrap()[0];
        assert!((along_z - along_x).norm() <= 64.0 * f64::EPSILON * along_z.norm());
    }

    #[test]
    fn rigid_baffle_suppresses_back_radiation() {
        let pressure = rayleigh_pressure(
            &[position([0.0, 0.0, -0.01])],
            &[piston(0.5e-3)],
            &RayleighIntegralSpec::new(inverse_m(TAU / 1.0e-3), inverse_m(0.0), 4, 12).unwrap(),
        )
        .unwrap();
        assert_eq!(pressure, vec![Complex64::new(0.0, 0.0)]);
    }

    #[test]
    fn quadrature_work_is_bounded() {
        let error =
            RayleighIntegralSpec::new(inverse_m(1.0), inverse_m(0.0), MAX_SURFACE_SAMPLES, 3)
                .expect_err("surface sample count exceeds the provider budget");
        assert!(error.to_string().contains("quadrature_surface_samples"));
    }

    #[test]
    fn layered_path_integrates_each_segment_exactly() {
        let path = RayleighPropagationPath::layered(vec![
            RayleighLayer::new(inverse_m(2.0), inverse_m(3.0), Some(length_m(0.25))).unwrap(),
            RayleighLayer::new(inverse_m(5.0), inverse_m(7.0), None).unwrap(),
        ])
        .unwrap();
        let (short_phase, short_attenuation) = path.propagation_terms(length_m(0.1)).unwrap();
        assert!((short_phase - 0.2).abs() <= f64::EPSILON * 0.2);
        assert!((short_attenuation - 0.3).abs() <= f64::EPSILON * 0.3);
        let (long_phase, long_attenuation) = path.propagation_terms(length_m(1.0)).unwrap();
        assert!((long_phase - 4.25).abs() <= 2.0 * f64::EPSILON * 4.25);
        assert!((long_attenuation - 6.0).abs() <= 2.0 * f64::EPSILON * 6.0);
    }

    #[test]
    fn layered_path_requires_one_final_half_space() {
        let finite =
            RayleighLayer::new(inverse_m(2.0), inverse_m(0.0), Some(length_m(0.25))).unwrap();
        let half_space = RayleighLayer::new(inverse_m(3.0), inverse_m(0.0), None).unwrap();
        let final_half_space = RayleighLayer::new(inverse_m(4.0), inverse_m(0.0), None).unwrap();
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
        let path = RayleighPropagationPath::homogeneous(inverse_m(1.0), inverse_m(0.0)).unwrap();
        for range_m in [-1.0, f64::NAN] {
            let error = path
                .propagation_terms(length_m(range_m))
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
