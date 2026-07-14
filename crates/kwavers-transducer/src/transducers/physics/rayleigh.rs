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

use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use std::f64::consts::{PI, TAU};

const LEGENDRE_ROOT_STEPS: usize = 64;
const LEGENDRE_ROOT_TOLERANCE: f64 = 8.0 * f64::EPSILON;
const MAX_SURFACE_SAMPLES: usize = 1 << 16;

/// Radial bounds and angular span of a planar aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanarApertureShape {
    /// Complete disk from the origin to `radius_m`.
    Disk { radius_m: f64 },
    /// Annular sector measured counter-clockwise from the aperture's first axis.
    AnnularSector {
        inner_radius_m: f64,
        outer_radius_m: f64,
        start_angle_rad: f64,
        span_angle_rad: f64,
    },
}

impl PlanarApertureShape {
    fn radial_and_angular_bounds(self) -> (f64, f64, f64, f64) {
        match self {
            Self::Disk { radius_m } => (0.0, radius_m, 0.0, TAU),
            Self::AnnularSector {
                inner_radius_m,
                outer_radius_m,
                start_angle_rad,
                span_angle_rad,
            } => (
                inner_radius_m,
                outer_radius_m,
                start_angle_rad,
                span_angle_rad,
            ),
        }
    }

    /// Exact planar area in square metres.
    #[must_use]
    pub fn area_m2(self) -> f64 {
        let (inner, outer, _, span) = self.radial_and_angular_bounds();
        0.5 * (outer * outer - inner * inner) * span
    }
}

/// Uniformly driven planar aperture embedded in an infinite rigid baffle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarAperture {
    center_m: [f64; 3],
    normal: [f64; 3],
    first_axis: [f64; 3],
    shape: PlanarApertureShape,
    surface_pressure_pa: Complex64,
}

impl PlanarAperture {
    /// Construct a complete circular aperture.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Config`] for non-finite geometry or pressure,
    /// a non-positive radius, or a zero normal.
    pub fn disk(
        center_m: [f64; 3],
        normal: [f64; 3],
        radius_m: f64,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        let (first_axis, _) = plane_basis(normal);
        Self::from_validated(
            center_m,
            normal,
            first_axis,
            PlanarApertureShape::Disk { radius_m },
            surface_pressure_pa,
        )
    }

    /// Construct a planar aperture with an explicit in-plane angular origin.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Config`] unless the shape bounds are valid and
    /// the first axis has a nonzero projection into the aperture plane.
    pub fn oriented(
        center_m: [f64; 3],
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        validate_point("first_axis", first_axis)?;
        let planar_axis = subtract(first_axis, scale(normal, dot(first_axis, normal)));
        let axis_norm = norm(planar_axis);
        validate_positive("first_axis_planar_norm", axis_norm)?;
        Self::from_validated(
            center_m,
            normal,
            scale(planar_axis, 1.0 / axis_norm),
            shape,
            surface_pressure_pa,
        )
    }

    fn from_validated(
        center_m: [f64; 3],
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        validate_point("center_m", center_m)?;
        match shape {
            PlanarApertureShape::Disk { radius_m } => validate_positive("radius_m", radius_m)?,
            PlanarApertureShape::AnnularSector {
                inner_radius_m,
                outer_radius_m,
                start_angle_rad,
                span_angle_rad,
            } => {
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
        if !surface_pressure_pa.re.is_finite() || !surface_pressure_pa.im.is_finite() {
            return Err(invalid(
                "surface_pressure_pa",
                format!("{surface_pressure_pa:?}"),
                "finite",
            ));
        }
        Ok(Self {
            center_m,
            normal,
            first_axis,
            shape,
            surface_pressure_pa,
        })
    }

    /// Piston centre in metres.
    #[must_use]
    pub const fn center_m(&self) -> [f64; 3] {
        self.center_m
    }

    /// Unit normal pointing into the radiating half-space.
    #[must_use]
    pub const fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// Aperture shape and bounds.
    #[must_use]
    pub const fn shape(&self) -> PlanarApertureShape {
        self.shape
    }

    /// Outer aperture radius in metres.
    #[must_use]
    pub fn outer_radius_m(&self) -> f64 {
        self.shape.radial_and_angular_bounds().1
    }

    /// Complex surface-pressure phasor in pascals.
    #[must_use]
    pub const fn surface_pressure_pa(&self) -> Complex64 {
        self.surface_pressure_pa
    }
}

/// Homogeneous-medium and disk-quadrature parameters for the Rayleigh integral.
#[derive(Debug, Clone, PartialEq)]
pub struct RayleighIntegralSpec {
    layers: Vec<RayleighLayer>,
    radial_order: usize,
    azimuthal_order: usize,
}

/// One straight-ray propagation segment, ordered outward from the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighLayer {
    wavenumber_rad_m: f64,
    attenuation_np_m: f64,
    thickness_m: Option<f64>,
}

impl RayleighLayer {
    /// Construct a propagation segment; `None` thickness denotes the final
    /// semi-infinite layer.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Config`] for non-finite coefficients, a
    /// non-positive wavenumber, negative attenuation, or non-positive finite
    /// thickness.
    pub fn new(
        wavenumber_rad_m: f64,
        attenuation_np_m: f64,
        thickness_m: Option<f64>,
    ) -> KwaversResult<Self> {
        validate_positive("wavenumber_rad_m", wavenumber_rad_m)?;
        if !attenuation_np_m.is_finite() || attenuation_np_m < 0.0 {
            return Err(invalid(
                "attenuation_np_m",
                attenuation_np_m.to_string(),
                "finite and >= 0",
            ));
        }
        if let Some(thickness) = thickness_m {
            validate_positive("layer_thickness_m", thickness)?;
        }
        Ok(Self {
            wavenumber_rad_m,
            attenuation_np_m,
            thickness_m,
        })
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
    /// Returns [`KwaversError::Config`] unless the wavenumber is finite and
    /// positive, attenuation is finite and non-negative, radial order is
    /// positive, and azimuthal order is at least three.
    pub fn new(
        wavenumber_rad_m: f64,
        attenuation_np_m: f64,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        let layer = RayleighLayer::new(wavenumber_rad_m, attenuation_np_m, None)?;
        Self::layered(vec![layer], radial_order, azimuthal_order)
    }

    /// Construct an ordered straight-ray layered propagation specification.
    ///
    /// The final layer must be semi-infinite and every preceding layer must
    /// carry a finite thickness. Phase and attenuation integrate segmentwise;
    /// interface reflection and refraction are outside this approximation.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Config`] for an empty or structurally invalid
    /// layer sequence, or invalid quadrature work.
    pub fn layered(
        layers: Vec<RayleighLayer>,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        if layers.is_empty() {
            return Err(invalid("layers", "0".to_owned(), "at least one layer"));
        }
        if layers
            .last()
            .is_some_and(|layer| layer.thickness_m.is_some())
        {
            return Err(invalid(
                "layers",
                "finite final layer".to_owned(),
                "final layer is semi-infinite",
            ));
        }
        if layers[..layers.len() - 1]
            .iter()
            .any(|layer| layer.thickness_m.is_none())
        {
            return Err(invalid(
                "layers",
                "non-final semi-infinite layer".to_owned(),
                "only final layer is semi-infinite",
            ));
        }
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
            layers,
            radial_order,
            azimuthal_order,
        })
    }

    /// Acoustic wavenumber in radians per metre.
    #[must_use]
    pub fn wavenumber_rad_m(&self) -> f64 {
        self.layers[0].wavenumber_rad_m
    }

    /// Amplitude attenuation coefficient in nepers per metre.
    #[must_use]
    pub fn attenuation_np_m(&self) -> f64 {
        self.layers[0].attenuation_np_m
    }

    fn path_terms(&self, range_m: f64) -> (f64, f64) {
        let mut remaining = range_m;
        let mut phase = 0.0;
        let mut attenuation = 0.0;
        for layer in &self.layers {
            let segment = layer
                .thickness_m
                .map_or(remaining, |thickness| remaining.min(thickness));
            phase = layer.wavenumber_rad_m.mul_add(segment, phase);
            attenuation = layer.attenuation_np_m.mul_add(segment, attenuation);
            remaining -= segment;
            if remaining <= 0.0 {
                break;
            }
        }
        (phase, attenuation)
    }
}

/// Evaluate the coherent complex pressure at every observation point.
///
/// Contributions are zero for a point on or behind a piston's baffle plane.
/// Attenuation is applied per source-to-observation path as `exp(-αR)`.
///
/// # Errors
///
/// Returns [`KwaversError::Config`] if an observation coordinate is non-finite
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
        if aperture.surface_pressure_pa == Complex64::new(0.0, 0.0) {
            continue;
        }
        let bitangent = cross(aperture.normal, aperture.first_axis);
        let (inner_radius, outer_radius, start_angle, span_angle) =
            aperture.shape.radial_and_angular_bounds();
        let squared_radius_span = outer_radius * outer_radius - inner_radius * inner_radius;
        let azimuthal_weight = 0.5 * squared_radius_span * span_angle / spec.azimuthal_order as f64;
        for (&point, total) in points_m.iter().zip(&mut pressure) {
            let center_offset = subtract(point, aperture.center_m);
            if dot(center_offset, aperture.normal) <= 0.0 {
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
                        aperture.center_m,
                        scale(
                            add(
                                scale(aperture.first_axis, azimuth.cos()),
                                scale(bitangent, azimuth.sin()),
                            ),
                            radius,
                        ),
                    );
                    let range = norm(subtract(point, surface_point));
                    let (phase, attenuation) = spec.path_terms(range);
                    let amplitude = area_weight * (-attenuation).exp() / range;
                    integral += Complex64::from_polar(amplitude, phase);
                }
            }
            *total += prefactor * aperture.surface_pressure_pa * integral;
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
    use kwavers_math::special::bessel::j1;

    fn piston(radius_m: f64) -> PlanarAperture {
        PlanarAperture::disk(
            [0.0; 3],
            [0.0, 0.0, 1.0],
            radius_m,
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
            inner_radius_m: 1.0e-3,
            outer_radius_m: 3.0e-3,
            start_angle_rad: 0.3,
            span_angle_rad: std::f64::consts::FRAC_PI_2,
        };
        let expected = std::f64::consts::FRAC_PI_2 * (9.0e-6 - 1.0e-6) / 2.0;
        assert!((shape.area_m2() - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }

    #[test]
    fn independently_driven_sectors_superpose_to_complete_annulus_on_axis() {
        let pressure = Complex64::from_polar(1.7e5, 0.4);
        let aperture = |start_angle_rad, span_angle_rad| {
            PlanarAperture::oriented(
                [0.0; 3],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                PlanarApertureShape::AnnularSector {
                    inner_radius_m: 0.4e-3,
                    outer_radius_m: 1.2e-3,
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
        let spec = RayleighIntegralSpec::new(TAU / 0.7e-3, 0.0, 12, 32).unwrap();
        let point = [[0.0, 0.0, 20.0e-3]];
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
        let spec = RayleighIntegralSpec::new(k, 0.0, 24, 24).unwrap();
        let actual = rayleigh_pressure(&[[0.0, 0.0, axial_range]], &[piston], &spec).unwrap()[0];
        let rim_range = axial_range.hypot(piston.outer_radius_m());
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
            [0.0, 0.0, range],
            [range * angle.sin(), 0.0, range * angle.cos()],
        ];
        let pressure = rayleigh_pressure(
            &points,
            &[piston],
            &RayleighIntegralSpec::new(k, 0.0, 20, 96).unwrap(),
        )
        .unwrap();
        let argument = k * piston.outer_radius_m() * angle.sin();
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
            [0.0; 3],
            [1.0, 0.0, 0.0],
            base.outer_radius_m(),
            base.surface_pressure_pa(),
        )
        .unwrap();
        let spec = RayleighIntegralSpec::new(TAU / 0.8e-3, 3.0, 12, 48).unwrap();
        let along_z = rayleigh_pressure(&[[0.2e-3, 0.0, 30.0e-3]], &[base], &spec).unwrap()[0];
        let along_x = rayleigh_pressure(&[[30.0e-3, 0.0, 0.2e-3]], &[rotated], &spec).unwrap()[0];
        assert!((along_z - along_x).norm() <= 64.0 * f64::EPSILON * along_z.norm());
    }

    #[test]
    fn rigid_baffle_suppresses_back_radiation() {
        let pressure = rayleigh_pressure(
            &[[0.0, 0.0, -0.01]],
            &[piston(0.5e-3)],
            &RayleighIntegralSpec::new(TAU / 1.0e-3, 0.0, 4, 12).unwrap(),
        )
        .unwrap();
        assert_eq!(pressure, vec![Complex64::new(0.0, 0.0)]);
    }

    #[test]
    fn quadrature_work_is_bounded() {
        let error = RayleighIntegralSpec::new(1.0, 0.0, MAX_SURFACE_SAMPLES, 3)
            .expect_err("surface sample count exceeds the provider budget");
        assert!(error.to_string().contains("quadrature_surface_samples"));
    }

    #[test]
    fn layered_path_integrates_each_segment_exactly() {
        let spec = RayleighIntegralSpec::layered(
            vec![
                RayleighLayer::new(2.0, 3.0, Some(0.25)).unwrap(),
                RayleighLayer::new(5.0, 7.0, None).unwrap(),
            ],
            1,
            6,
        )
        .unwrap();
        let (short_phase, short_attenuation) = spec.path_terms(0.1);
        assert!((short_phase - 0.2).abs() <= f64::EPSILON * 0.2);
        assert!((short_attenuation - 0.3).abs() <= f64::EPSILON * 0.3);
        let (long_phase, long_attenuation) = spec.path_terms(1.0);
        assert!((long_phase - 4.25).abs() <= 2.0 * f64::EPSILON * 4.25);
        assert!((long_attenuation - 6.0).abs() <= 2.0 * f64::EPSILON * 6.0);
    }

    #[test]
    fn layered_path_requires_one_final_half_space() {
        let finite = RayleighLayer::new(2.0, 0.0, Some(0.25)).unwrap();
        let half_space = RayleighLayer::new(3.0, 0.0, None).unwrap();
        assert!(RayleighIntegralSpec::layered(vec![finite], 1, 6).is_err());
        assert!(RayleighIntegralSpec::layered(vec![half_space, finite], 1, 6).is_err());
    }
}
