//! Shared focused-bowl placement adapters for clinical theranostic geometry.
//!
//! Clinical modules derive anatomy-specific focus, radius, and aperture bounds.
//! Source-domain [`BowlTransducer`] owns spherical-cap validation, equal-area
//! sampling, and element placement.

use crate::core::error::KwaversResult;
use crate::domain::source::transducers::focused::{BowlAngularBounds, BowlConfig, BowlTransducer};

use super::Point3;

const GEOMETRY_UNIT_FREQUENCY_HZ: f64 = 1.0;
const GEOMETRY_UNIT_AMPLITUDE_PA: f64 = 1.0;

/// Focused-bowl vertex orientation relative to the focus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FocusedBowlVertexDirection {
    /// Vertex lies at `focus.z + radius`.
    PositiveZ,
    /// Vertex lies at `focus.z - radius`.
    NegativeZ,
}

impl FocusedBowlVertexDirection {
    #[must_use]
    pub(crate) fn from_superior_positive(superior_positive: bool) -> Self {
        if superior_positive {
            Self::PositiveZ
        } else {
            Self::NegativeZ
        }
    }

    #[must_use]
    fn sign(self) -> f64 {
        match self {
            Self::PositiveZ => 1.0,
            Self::NegativeZ => -1.0,
        }
    }
}

/// Source-domain focused-bowl cap placement request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FocusedBowlCapSpec {
    pub element_count: usize,
    pub focus_m: Point3,
    pub radius_m: f64,
    pub vertex_direction: FocusedBowlVertexDirection,
    pub angular_bounds: BowlAngularBounds,
    pub frequency_hz: f64,
    pub amplitude_pa: f64,
}

impl FocusedBowlCapSpec {
    #[must_use]
    pub(crate) fn geometry_only(
        element_count: usize,
        focus_m: Point3,
        radius_m: f64,
        vertex_direction: FocusedBowlVertexDirection,
        angular_bounds: BowlAngularBounds,
    ) -> Self {
        Self {
            element_count,
            focus_m,
            radius_m,
            vertex_direction,
            angular_bounds,
            frequency_hz: GEOMETRY_UNIT_FREQUENCY_HZ,
            amplitude_pa: GEOMETRY_UNIT_AMPLITUDE_PA,
        }
    }
}

/// Generate focused-bowl cap points through the source-domain bowl transducer.
///
/// # Theorem
///
/// For focus `F`, radius `R`, vertex direction `s e_z`, and angular bounds
/// `[theta_min, theta_max]`, this constructs a bowl vertex
/// `V = F + s R e_z` and delegates to
/// [`BowlTransducer::with_angular_bounds`]. Therefore every returned point lies
/// on `||x - F|| = R`, and equal-area weights and angular-domain validation are
/// identical to direct source-domain bowl construction.
///
/// # Errors
///
/// Returns any source-domain validation error for invalid count, radius,
/// focus/vertex degeneracy, or angular bounds.
pub(crate) fn focused_bowl_cap_points(spec: FocusedBowlCapSpec) -> KwaversResult<Vec<Point3>> {
    let focus_m = [spec.focus_m.x_m, spec.focus_m.y_m, spec.focus_m.z_m];
    let vertex_m = [
        spec.focus_m.x_m,
        spec.focus_m.y_m,
        spec.focus_m.z_m + spec.vertex_direction.sign() * spec.radius_m,
    ];
    let config = BowlConfig::from_vertex_focus(
        vertex_m,
        focus_m,
        2.0 * spec.radius_m,
        spec.frequency_hz,
        spec.amplitude_pa,
    );
    let bowl =
        BowlTransducer::with_angular_bounds(config, spec.angular_bounds, spec.element_count)?;

    Ok(bowl
        .element_positions()
        .iter()
        .map(|position| Point3 {
            x_m: position[0],
            y_m: position[1],
            z_m: position[2],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_points_follow_positive_vertex_axis() {
        let focus = Point3 {
            x_m: 0.01,
            y_m: -0.02,
            z_m: 0.03,
        };
        let radius = 0.12;
        let points = focused_bowl_cap_points(FocusedBowlCapSpec::geometry_only(
            48,
            focus,
            radius,
            FocusedBowlVertexDirection::PositiveZ,
            BowlAngularBounds::new(0.2, 0.9).unwrap(),
        ))
        .unwrap();

        assert_eq!(points.len(), 48);
        for point in points {
            let distance = ((point.x_m - focus.x_m).powi(2)
                + (point.y_m - focus.y_m).powi(2)
                + (point.z_m - focus.z_m).powi(2))
            .sqrt();
            assert!((distance - radius).abs() < 1.0e-12);
            assert!(point.z_m > focus.z_m);
        }
    }

    #[test]
    fn cap_points_follow_negative_vertex_axis() {
        let focus = Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        };
        let points = focused_bowl_cap_points(FocusedBowlCapSpec::geometry_only(
            16,
            focus,
            0.08,
            FocusedBowlVertexDirection::NegativeZ,
            BowlAngularBounds::polar_span(0.6).unwrap(),
        ))
        .unwrap();

        assert_eq!(points.len(), 16);
        assert!(points.iter().all(|point| point.z_m < focus.z_m));
    }
}
