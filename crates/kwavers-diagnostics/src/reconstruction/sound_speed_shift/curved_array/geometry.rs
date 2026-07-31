//! Circular-arc element coordinates for 2-D curved arrays.

use aequitas::systems::si::quantities::{Angle, Length};
use kwavers_core::error::KwaversResult;
use kwavers_solver::inverse::same_aperture::PlanarPoint;

use super::validation::validate_array;

/// Circular-arc transducer geometry in the reconstruction plane.
///
/// Element `i` is placed at
/// `center + radius * [cos(first_angle + i * angular_pitch),
/// sin(first_angle + i * angular_pitch)]`. The endpoint form is available
/// through [`CurvedArray2d::from_arc_endpoints`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvedArray2d {
    /// Arc center `m`.
    pub center_m: PlanarPoint,
    /// Arc radius.
    pub radius: Length,
    /// Angle of the first element.
    pub first_angle: Angle,
    /// Angle increment between adjacent elements.
    pub angular_pitch: Angle,
    /// Number of physical array elements.
    pub element_count: usize,
}

impl CurvedArray2d {
    /// Build a curved array by specifying both arc endpoint angles.
    #[must_use]
    pub fn from_arc_endpoints(
        center_m: PlanarPoint,
        radius: Length,
        start_angle: Angle,
        end_angle: Angle,
        element_count: usize,
    ) -> Self {
        let angular_pitch = if element_count > 1 {
            Angle::from_base(
                (end_angle.into_base() - start_angle.into_base()) / (element_count - 1) as f64,
            )
        } else {
            Angle::from_base(0.0)
        };
        Self {
            center_m,
            radius,
            first_angle: start_angle,
            angular_pitch,
            element_count,
        }
    }

    /// Return all element coordinates in deterministic element-index order.
    ///
    /// # Errors
    /// Returns [`kwavers_core::error::KwaversError`] when the arc geometry is
    /// nonfinite, degenerate, or aliases a full closed ring endpoint.
    pub fn elements(&self) -> KwaversResult<Vec<PlanarPoint>> {
        validate_array(self)?;
        Ok((0..self.element_count)
            .map(|idx| self.element(idx))
            .collect())
    }

    /// Total signed angular aperture from first to last element.
    #[must_use]
    pub fn aperture_angle(&self) -> Angle {
        Angle::from_base(
            self.angular_pitch.into_base() * (self.element_count.saturating_sub(1)) as f64,
        )
    }

    pub(super) fn element(&self, index: usize) -> PlanarPoint {
        let angle = self.first_angle.into_base() + index as f64 * self.angular_pitch.into_base();
        let radius = self.radius.into_base();
        PlanarPoint {
            x_m: self.center_m.x_m + radius * angle.cos(),
            y_m: self.center_m.y_m + radius * angle.sin(),
        }
    }
}
