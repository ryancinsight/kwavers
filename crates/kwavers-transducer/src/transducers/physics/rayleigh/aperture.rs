//! Validated planar aperture shapes, oriented geometry and surface drive.

use super::input::{invalid, validate_point, validate_positive};
use super::vector::{dot, norm, normalized_normal, plane_basis, scale, subtract};
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use std::f64::consts::TAU;

/// Outline of a planar aperture in its local frame.
///
/// Polar shapes are measured from the aperture centre counter-clockwise from
/// the first axis; a rectangle is centred on the aperture centre with its
/// width along the first axis and its height along `normal × first_axis`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
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
    /// Rectangle of `width_m` along the first axis and `height_m` across it,
    /// the element outline of a linear or matrix array.
    Rectangle { width_m: f64, height_m: f64 },
}

impl PlanarApertureShape {
    /// Inner radius, outer radius, start angle and angular span of a polar
    /// shape; `None` for a rectangle.
    pub(crate) fn polar_bounds(self) -> Option<(f64, f64, f64, f64)> {
        match self {
            Self::Disk { radius_m } => Some((0.0, radius_m, 0.0, TAU)),
            Self::AnnularSector {
                inner_radius_m,
                outer_radius_m,
                start_angle_rad,
                span_angle_rad,
            } => Some((
                inner_radius_m,
                outer_radius_m,
                start_angle_rad,
                span_angle_rad,
            )),
            Self::Rectangle { .. } => None,
        }
    }

    /// Exact planar area in square metres.
    #[must_use]
    pub fn area_m2(self) -> f64 {
        match self {
            Self::Disk { radius_m } => 0.5 * radius_m * radius_m * TAU,
            Self::AnnularSector {
                inner_radius_m,
                outer_radius_m,
                span_angle_rad,
                ..
            } => {
                0.5 * (outer_radius_m * outer_radius_m - inner_radius_m * inner_radius_m)
                    * span_angle_rad
            }
            Self::Rectangle { width_m, height_m } => width_m * height_m,
        }
    }

    /// Radius of the smallest centred disk that contains the aperture, metres.
    #[must_use]
    pub fn outer_radius_m(self) -> f64 {
        match self {
            Self::Disk { radius_m } => radius_m,
            Self::AnnularSector { outer_radius_m, .. } => outer_radius_m,
            Self::Rectangle { width_m, height_m } => 0.5 * width_m.hypot(height_m),
        }
    }
}

/// Validated position, orientation, and outline of a planar aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarApertureGeometry {
    center_m: [f64; 3],
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
    pub fn disk(center_m: [f64; 3], normal: [f64; 3], radius_m: f64) -> KwaversResult<Self> {
        let normal = normalized_normal(normal)?;
        let (first_axis, _) = plane_basis(normal);
        Self::from_validated(
            center_m,
            normal,
            first_axis,
            PlanarApertureShape::Disk { radius_m },
        )
    }

    /// Construct a planar aperture with an explicit in-plane first axis: the
    /// angular origin of a polar shape, the width direction of a rectangle.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the shape bounds are valid and
    /// the first axis has a nonzero projection into the aperture plane.
    pub fn oriented(
        center_m: [f64; 3],
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
        center_m: [f64; 3],
        normal: [f64; 3],
        first_axis: [f64; 3],
        shape: PlanarApertureShape,
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
            PlanarApertureShape::Rectangle { width_m, height_m } => {
                validate_positive("width_m", width_m)?;
                validate_positive("height_m", height_m)?;
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
    pub const fn center_m(&self) -> [f64; 3] {
        self.center_m
    }

    /// Unit normal pointing into the radiating half-space.
    #[must_use]
    pub const fn normal(&self) -> [f64; 3] {
        self.normal
    }

    /// Unit in-plane axis: the angular origin of a polar shape, the width
    /// direction of a rectangle.
    #[must_use]
    pub const fn first_axis(&self) -> [f64; 3] {
        self.first_axis
    }

    /// Aperture shape and bounds.
    #[must_use]
    pub const fn shape(&self) -> PlanarApertureShape {
        self.shape
    }

    /// Radius of the smallest centred disk that contains the aperture, metres.
    #[must_use]
    pub fn outer_radius_m(&self) -> f64 {
        self.shape.outer_radius_m()
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
        center_m: [f64; 3],
        normal: [f64; 3],
        radius_m: f64,
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
        center_m: [f64; 3],
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
    pub const fn center_m(&self) -> [f64; 3] {
        self.geometry.center_m()
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

    /// Radius of the smallest centred disk that contains the aperture, metres.
    #[must_use]
    pub fn outer_radius_m(&self) -> f64 {
        self.geometry.outer_radius_m()
    }

    /// Complex surface-pressure phasor in pascals.
    #[must_use]
    pub const fn surface_pressure_pa(&self) -> Complex64 {
        self.surface_pressure_pa
    }
}
