//! Angular-span constructors for focused bowl source layouts.
//!
//! A physical aperture diameter identifies only minor spherical caps. For fixed
//! arrays whose active surface spans a hemisphere or major cap, the angular
//! aperture is the complete source-domain parameter. These constructors route
//! that parameter through the canonical equal-area spherical-cap layout.

use std::f64::consts::PI;

use super::{
    add3, field_validation_error, normalize3, scale3, sub3, validate_element_count, BowlConfig,
    BowlTransducer,
};
use crate::core::error::KwaversResult;
use crate::domain::source::transducers::focused::{SphericalCapConfig, SphericalCapLayout};

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
        Self::with_polar_bounds(config, 0.0, theta_max_rad, element_count)
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
        validate_element_count(element_count)?;
        Self::validate_config(&config)?;
        validate_polar_bounds(theta_min_rad, theta_max_rad)?;

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
            theta_min_rad,
            theta_max_rad,
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
