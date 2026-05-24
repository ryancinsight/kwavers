//! Angular-span constructors for focused bowl source layouts.
//!
//! A physical aperture diameter identifies only minor spherical caps. For fixed
//! arrays whose active surface spans a hemisphere or major cap, the angular
//! aperture is the complete source-domain parameter. These constructors route
//! that parameter through the canonical equal-area spherical-cap layout.

use std::f64::consts::PI;

use super::super::validation::{field_validation_error, validate_element_count};
use super::{add3, normalize3, scale3, sub3, BowlConfig, BowlTransducer};
use crate::core::error::KwaversResult;
use crate::domain::source::transducers::focused::{SphericalCapConfig, SphericalCapLayout};

/// Validated polar-angle coverage for a focused bowl aperture.
///
/// Angles are measured from the vertex-to-focus axis. Keeping this as a source
/// value object lets clinical placement code request "a cap covering these
/// normalized axial bounds" without duplicating trigonometric conversion or
/// aperture-domain validation outside the bowl source boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BowlAngularBounds {
    theta_min_rad: f64,
    theta_max_rad: f64,
}

impl BowlAngularBounds {
    /// Construct explicit polar-angle bounds.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] when the bounds
    /// do not satisfy `0 <= theta_min < theta_max <= pi`.
    pub fn new(theta_min_rad: f64, theta_max_rad: f64) -> KwaversResult<Self> {
        validate_polar_bounds(theta_min_rad, theta_max_rad)?;
        Ok(Self {
            theta_min_rad,
            theta_max_rad,
        })
    }

    /// Construct a cap over `0 <= theta <= theta_max_rad`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] when
    /// `theta_max_rad` is outside the physical focused-bowl domain.
    pub fn polar_span(theta_max_rad: f64) -> KwaversResult<Self> {
        Self::new(0.0, theta_max_rad)
    }

    /// Construct the full `0 <= theta <= pi/2` hemispherical aperture.
    #[must_use]
    pub fn hemisphere() -> Self {
        Self {
            theta_min_rad: 0.0,
            theta_max_rad: PI / 2.0,
        }
    }

    /// Construct bounds from normalized aperture-axis projections.
    ///
    /// `axis_projection = dot(position - focus, vertex - focus) / R^2`, so a
    /// hemispherical cap is `[0, 1]` and a major cap crossing the equator has a
    /// negative lower bound. Since `axis_projection = cos(theta)`, the smaller
    /// projection maps to the larger polar angle.
    ///
    /// # Theorem
    ///
    /// For a spherical bowl of radius `R` centered on the focus, the cap area
    /// over projection bounds `[u_min, u_max]` is
    /// `2 pi R^2 (u_max - u_min)`. The conversion
    /// `theta_min = acos(u_max)`, `theta_max = acos(u_min)` preserves this area
    /// exactly under the focused-cap equal-area sampler.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] unless
    /// `-1 <= axis_projection_min < axis_projection_max <= 1`.
    pub fn from_axis_projection_bounds(
        axis_projection_min: f64,
        axis_projection_max: f64,
    ) -> KwaversResult<Self> {
        if axis_projection_min.is_finite()
            && axis_projection_max.is_finite()
            && axis_projection_min >= -1.0
            && axis_projection_min < axis_projection_max
            && axis_projection_max <= 1.0
        {
            Self::new(axis_projection_max.acos(), axis_projection_min.acos())
        } else {
            Err(field_validation_error(
                "axis_projection_bounds",
                format!("[{axis_projection_min}, {axis_projection_max}]"),
                "must satisfy -1 <= min < max <= 1",
            ))
        }
    }

    /// Lower polar-angle bound [rad].
    #[must_use]
    pub fn theta_min_rad(self) -> f64 {
        self.theta_min_rad
    }

    /// Upper polar-angle bound [rad].
    #[must_use]
    pub fn theta_max_rad(self) -> f64 {
        self.theta_max_rad
    }

    /// Lower normalized aperture-axis projection.
    #[must_use]
    pub fn axis_projection_min(self) -> f64 {
        self.theta_max_rad.cos()
    }

    /// Upper normalized aperture-axis projection.
    #[must_use]
    pub fn axis_projection_max(self) -> f64 {
        self.theta_min_rad.cos()
    }
}

impl BowlTransducer {
    /// Create a fixed-count bowl over `0 <= theta <= theta_max_rad`.
    ///
    /// Use this constructor when the aperture is specified by polar coverage
    /// rather than projected diameter. The `BowlConfig::diameter` field remains
    /// validated as a finite positive source parameter but does not determine
    /// the polar span because projected diameter cannot distinguish a
    /// hemispherical cap from a major cap.
    ///
    /// # Theorem
    ///
    /// For radius `R` and polar span `[0, theta_max]`, the covered surface area
    /// is `2 pi R^2 (1 - cos(theta_max))`. The generated element weights sum to
    /// this value because [`SphericalCapLayout`] samples uniformly in
    /// `cos(theta)`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] when the count,
    /// config, or polar span violates the focused-bowl source domain.
    pub fn with_polar_span(
        config: BowlConfig,
        theta_max_rad: f64,
        element_count: usize,
    ) -> KwaversResult<Self> {
        let bounds = BowlAngularBounds::polar_span(theta_max_rad)?;
        Self::with_angular_bounds(config, bounds, element_count)
    }

    /// Create a fixed-count bowl over `theta_min_rad <= theta <= theta_max_rad`.
    ///
    /// This supports annular focused bowls and cap layouts with a central
    /// cutout. Angles are measured from the bowl vertex-to-focus axis.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] when the count,
    /// config, or angular bounds violate the focused-bowl source domain.
    pub fn with_polar_bounds(
        config: BowlConfig,
        theta_min_rad: f64,
        theta_max_rad: f64,
        element_count: usize,
    ) -> KwaversResult<Self> {
        let bounds = BowlAngularBounds::new(theta_min_rad, theta_max_rad)?;
        Self::with_angular_bounds(config, bounds, element_count)
    }

    /// Create a fixed-count bowl from normalized aperture-axis projections.
    ///
    /// This is the source-domain constructor for hemispherical and major-cap
    /// aperture requests expressed in unit-radius coordinates rather than
    /// radians. It prevents clinical adapters from reimplementing the
    /// `acos` conversion and keeps the equal-area theorem under one source API.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] when the count,
    /// config, or projection bounds violate the focused-bowl source domain.
    pub fn with_axis_projection_bounds(
        config: BowlConfig,
        axis_projection_min: f64,
        axis_projection_max: f64,
        element_count: usize,
    ) -> KwaversResult<Self> {
        let bounds = BowlAngularBounds::from_axis_projection_bounds(
            axis_projection_min,
            axis_projection_max,
        )?;
        Self::with_angular_bounds(config, bounds, element_count)
    }

    /// Create a fixed-count bowl from validated angular bounds.
    ///
    /// # Errors
    ///
    /// Returns [`crate::core::error::KwaversError::Validation`] when the count
    /// or config violates the focused-bowl source domain.
    pub fn with_angular_bounds(
        config: BowlConfig,
        bounds: BowlAngularBounds,
        element_count: usize,
    ) -> KwaversResult<Self> {
        validate_element_count(element_count)?;
        Self::validate_config(&config)?;

        let radius_m = config.radius_of_curvature;
        let axis = sub3(config.focus, config.center);
        let axis_unit = normalize3(axis).ok_or_else(|| {
            field_validation_error(
                "focus",
                format!("{:?}", config.focus),
                "must differ from center to define the bowl acoustic axis",
            )
        })?;
        let curvature_center = add3(config.center, scale3(axis_unit, radius_m));
        let layout = SphericalCapLayout::new(SphericalCapConfig::focused_cap(
            element_count,
            radius_m,
            curvature_center,
            axis,
            bounds.theta_min_rad,
            bounds.theta_max_rad,
        ))?;

        let mut element_positions = Vec::with_capacity(layout.elements().len());
        let mut element_normals = Vec::with_capacity(layout.elements().len());
        let mut element_areas = Vec::with_capacity(layout.elements().len());
        for element in layout.elements() {
            element_positions.push(element.position_m);
            element_normals.push(element.normal_to_focus);
            element_areas.push(element.area_weight_m2);
        }

        Ok(Self {
            config,
            element_positions,
            element_normals,
            element_areas,
        })
    }
}

fn validate_polar_bounds(theta_min_rad: f64, theta_max_rad: f64) -> KwaversResult<()> {
    if theta_min_rad.is_finite()
        && theta_max_rad.is_finite()
        && theta_min_rad >= 0.0
        && theta_min_rad < theta_max_rad
        && theta_max_rad <= PI
    {
        Ok(())
    } else {
        Err(field_validation_error(
            "polar_bounds",
            format!("[{theta_min_rad}, {theta_max_rad}]"),
            "must satisfy 0 <= theta_min < theta_max <= pi",
        ))
    }
}
