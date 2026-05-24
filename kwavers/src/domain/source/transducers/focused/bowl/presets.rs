//! Focused bowl configuration constructors.

use crate::core::error::KwaversResult;

use super::super::validation::{
    field_validation_error, positive_finite, validate_finite_vector, validate_positive_finite_field,
};
use super::BowlConfig;

impl BowlConfig {
    /// Build a spherical-cap bowl from its vertex and acoustic focus.
    ///
    /// The radius of curvature is the Euclidean distance from `vertex_m` to
    /// `focus_m`. `diameter_m` controls the cap aperture; use
    /// [`Self::hemispherical`] when the intended aperture is a full hemisphere.
    #[must_use]
    pub fn from_vertex_focus(
        vertex_m: [f64; 3],
        focus_m: [f64; 3],
        diameter_m: f64,
        frequency_hz: f64,
        amplitude_pa: f64,
    ) -> Self {
        Self {
            radius_of_curvature: distance3(vertex_m, focus_m),
            diameter: diameter_m,
            center: vertex_m,
            focus: focus_m,
            frequency: frequency_hz,
            amplitude: amplitude_pa,
            ..Self::default()
        }
    }

    /// Build a hemispherical focused bowl from its vertex and acoustic focus.
    ///
    /// This source-domain constructor supports fixed-count hemispherical arrays
    /// without naming the source after a clinical anatomy or device topology.
    #[must_use]
    pub fn hemispherical(
        vertex_m: [f64; 3],
        focus_m: [f64; 3],
        frequency_hz: f64,
        amplitude_pa: f64,
    ) -> Self {
        let radius_m = distance3(vertex_m, focus_m);
        Self::from_vertex_focus(
            vertex_m,
            focus_m,
            2.0 * radius_m,
            frequency_hz,
            amplitude_pa,
        )
    }

    /// Build a spherical-cap bowl from an axis reference, focus, and radius.
    ///
    /// The axis reference fixes the vertex-to-focus axis but does not have to
    /// be the bowl vertex. This supports clinical placements that choose an
    /// anatomical contact point to orient the aperture while selecting a larger
    /// curvature radius from an outside-body rim constraint.
    ///
    /// # Theorem
    ///
    /// Let `A` be the axis reference, `F` be the acoustic focus, `R > 0`, and
    /// `d = normalize(F - A)`. The constructed vertex is `V = F - R d`; hence
    /// `normalize(F - V) = d` and `||F - V|| = R`. Any layout produced by
    /// [`super::BowlTransducer::with_angular_bounds`] from this config is
    /// therefore centered on the requested focus with the requested radius and
    /// the axis implied by `A`.
    ///
    /// # Errors
    ///
    /// Returns a validation error when coordinates are non-finite, `radius_m`
    /// is not positive finite, or the axis reference equals the focus.
    pub fn from_axis_reference_focus(
        axis_reference_m: [f64; 3],
        focus_m: [f64; 3],
        radius_m: f64,
        frequency_hz: f64,
        amplitude_pa: f64,
    ) -> KwaversResult<Self> {
        validate_finite_vector("axis_reference", axis_reference_m)?;
        validate_finite_vector("focus", focus_m)?;
        validate_positive_finite_field("radius_of_curvature", radius_m)?;

        let axis = [
            focus_m[0] - axis_reference_m[0],
            focus_m[1] - axis_reference_m[1],
            focus_m[2] - axis_reference_m[2],
        ];
        let axis_norm = (axis[0])
            .mul_add(axis[0], axis[1].mul_add(axis[1], axis[2] * axis[2]))
            .sqrt();
        if !positive_finite(axis_norm) {
            return Err(field_validation_error(
                "axis_reference",
                format!("{axis_reference_m:?}"),
                "must differ from focus to define the bowl acoustic axis",
            ));
        }

        let inv_axis_norm = axis_norm.recip();
        let vertex_m = [
            focus_m[0] - radius_m * axis[0] * inv_axis_norm,
            focus_m[1] - radius_m * axis[1] * inv_axis_norm,
            focus_m[2] - radius_m * axis[2] * inv_axis_norm,
        ];

        Ok(Self {
            radius_of_curvature: radius_m,
            diameter: 2.0 * radius_m,
            center: vertex_m,
            focus: focus_m,
            frequency: frequency_hz,
            amplitude: amplitude_pa,
            ..Self::default()
        })
    }

    /// Return a config using the requested discretization element size [m].
    #[must_use]
    pub fn with_element_size(mut self, element_size_m: f64) -> Self {
        self.element_size = Some(element_size_m);
        self
    }
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0])
        .mul_add(
            a[0] - b[0],
            (a[1] - b[1]).mul_add(a[1] - b[1], (a[2] - b[2]) * (a[2] - b[2])),
        )
        .sqrt()
}
