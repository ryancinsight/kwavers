//! Frequency-domain radiation from oriented, rigidly baffled circular pistons.
//!
//! For prescribed normal surface velocity, the Rayleigh first integral is
//!
//! `p(x,ω) = -i ωρ/(2π) ∫_S v_n(s) exp(i k R)/R dS`.
//!
//! This module accepts the equivalent surface-pressure phasor
//! `p_s = ρ c v_n`, so its prefactor is `-i k/(2π)`. Each finite piston face is
//! integrated directly; applying a separate piston-directivity factor would
//! count the same aperture diffraction twice. The convention follows Qin et
//! al., *Ultrasonics* 51 (2011), Eq. 1,
//! <https://doi.org/10.1016/j.ultras.2010.12.011>.

use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use std::f64::consts::{PI, TAU};

const LEGENDRE_ROOT_STEPS: usize = 64;
const LEGENDRE_ROOT_TOLERANCE: f64 = 8.0 * f64::EPSILON;

/// Uniformly driven circular piston embedded in an infinite rigid baffle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularPiston {
    center_m: [f64; 3],
    normal: [f64; 3],
    radius_m: f64,
    surface_pressure_pa: Complex64,
}

impl CircularPiston {
    /// Construct a piston, normalizing its radiating-face normal.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::Config`] for non-finite geometry or pressure,
    /// a non-positive radius, or a zero normal.
    pub fn new(
        center_m: [f64; 3],
        normal: [f64; 3],
        radius_m: f64,
        surface_pressure_pa: Complex64,
    ) -> KwaversResult<Self> {
        validate_point("center_m", center_m)?;
        validate_positive("radius_m", radius_m)?;
        if !surface_pressure_pa.re.is_finite() || !surface_pressure_pa.im.is_finite() {
            return Err(invalid(
                "surface_pressure_pa",
                format!("{surface_pressure_pa:?}"),
                "finite",
            ));
        }
        let norm = normal
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();
        validate_positive("normal_norm", norm)?;
        Ok(Self {
            center_m,
            normal: normal.map(|component| component / norm),
            radius_m,
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

    /// Piston radius in metres.
    #[must_use]
    pub const fn radius_m(&self) -> f64 {
        self.radius_m
    }

    /// Complex surface-pressure phasor in pascals.
    #[must_use]
    pub const fn surface_pressure_pa(&self) -> Complex64 {
        self.surface_pressure_pa
    }
}

/// Homogeneous-medium and disk-quadrature parameters for the Rayleigh integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighIntegralSpec {
    wavenumber_rad_m: f64,
    attenuation_np_m: f64,
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
    /// Returns [`KwaversError::Config`] unless the wavenumber is finite and
    /// positive, attenuation is finite and non-negative, radial order is
    /// positive, and azimuthal order is at least three.
    pub fn new(
        wavenumber_rad_m: f64,
        attenuation_np_m: f64,
        radial_order: usize,
        azimuthal_order: usize,
    ) -> KwaversResult<Self> {
        validate_positive("wavenumber_rad_m", wavenumber_rad_m)?;
        if !attenuation_np_m.is_finite() || attenuation_np_m < 0.0 {
            return Err(invalid(
                "attenuation_np_m",
                attenuation_np_m.to_string(),
                "finite and >= 0",
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
        Ok(Self {
            wavenumber_rad_m,
            attenuation_np_m,
            radial_order,
            azimuthal_order,
        })
    }

    /// Acoustic wavenumber in radians per metre.
    #[must_use]
    pub const fn wavenumber_rad_m(&self) -> f64 {
        self.wavenumber_rad_m
    }

    /// Amplitude attenuation coefficient in nepers per metre.
    #[must_use]
    pub const fn attenuation_np_m(&self) -> f64 {
        self.attenuation_np_m
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
    pistons: &[CircularPiston],
    spec: RayleighIntegralSpec,
) -> KwaversResult<Vec<Complex64>> {
    for &point in points_m {
        validate_point("observation_point_m", point)?;
    }
    let radial_rule = gauss_legendre_unit(spec.radial_order)?;
    let prefactor = Complex64::new(0.0, -spec.wavenumber_rad_m / TAU);
    let mut pressure = vec![Complex64::new(0.0, 0.0); points_m.len()];

    for piston in pistons {
        if piston.surface_pressure_pa == Complex64::new(0.0, 0.0) {
            continue;
        }
        let (tangent, bitangent) = plane_basis(piston.normal);
        let azimuthal_weight = PI * piston.radius_m * piston.radius_m / spec.azimuthal_order as f64;
        for (&point, total) in points_m.iter().zip(&mut pressure) {
            let center_offset = subtract(point, piston.center_m);
            if dot(center_offset, piston.normal) <= 0.0 {
                continue;
            }
            let mut integral = Complex64::new(0.0, 0.0);
            for &(squared_radius, radial_weight) in &radial_rule {
                let radius = piston.radius_m * squared_radius.sqrt();
                let area_weight = azimuthal_weight * radial_weight;
                for azimuth_index in 0..spec.azimuthal_order {
                    let azimuth = TAU * azimuth_index as f64 / spec.azimuthal_order as f64;
                    let surface_point = add(
                        piston.center_m,
                        scale(
                            add(
                                scale(tangent, azimuth.cos()),
                                scale(bitangent, azimuth.sin()),
                            ),
                            radius,
                        ),
                    );
                    let range = norm(subtract(point, surface_point));
                    let amplitude = area_weight * (-spec.attenuation_np_m * range).exp() / range;
                    integral += Complex64::from_polar(amplitude, spec.wavenumber_rad_m * range);
                }
            }
            *total += prefactor * piston.surface_pressure_pa * integral;
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

    fn piston(radius_m: f64) -> CircularPiston {
        CircularPiston::new(
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
    fn on_axis_pressure_matches_closed_form() {
        let piston = piston(1.2e-3);
        let k = TAU / 0.75e-3;
        let axial_range = 14.0e-3;
        let spec = RayleighIntegralSpec::new(k, 0.0, 24, 24).unwrap();
        let actual = rayleigh_pressure(&[[0.0, 0.0, axial_range]], &[piston], spec).unwrap()[0];
        let rim_range = axial_range.hypot(piston.radius_m());
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
            RayleighIntegralSpec::new(k, 0.0, 20, 96).unwrap(),
        )
        .unwrap();
        let argument = k * piston.radius_m() * angle.sin();
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
        let rotated = CircularPiston::new(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            base.radius_m(),
            base.surface_pressure_pa(),
        )
        .unwrap();
        let spec = RayleighIntegralSpec::new(TAU / 0.8e-3, 3.0, 12, 48).unwrap();
        let along_z = rayleigh_pressure(&[[0.2e-3, 0.0, 30.0e-3]], &[base], spec).unwrap()[0];
        let along_x = rayleigh_pressure(&[[30.0e-3, 0.0, 0.2e-3]], &[rotated], spec).unwrap()[0];
        assert!((along_z - along_x).norm() <= 64.0 * f64::EPSILON * along_z.norm());
    }

    #[test]
    fn rigid_baffle_suppresses_back_radiation() {
        let pressure = rayleigh_pressure(
            &[[0.0, 0.0, -0.01]],
            &[piston(0.5e-3)],
            RayleighIntegralSpec::new(TAU / 1.0e-3, 0.0, 4, 12).unwrap(),
        )
        .unwrap();
        assert_eq!(pressure, vec![Complex64::new(0.0, 0.0)]);
    }
}
