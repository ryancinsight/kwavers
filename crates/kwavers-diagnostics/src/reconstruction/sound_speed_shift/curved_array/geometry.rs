//! Circular-arc element coordinates for 2-D curved arrays.

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
    /// Arc center [m].
    pub center_m: PlanarPoint,
    /// Arc radius [m].
    pub radius_m: f64,
    /// Angle of the first element [rad].
    pub first_angle_rad: f64,
    /// Angle increment between adjacent elements [rad].
    pub angular_pitch_rad: f64,
    /// Number of physical array elements.
    pub element_count: usize,
}

impl CurvedArray2d {
    /// Build a curved array by specifying both arc endpoint angles.
    #[must_use]
    pub fn from_arc_endpoints(
        center_m: PlanarPoint,
        radius_m: f64,
        start_angle_rad: f64,
        end_angle_rad: f64,
        element_count: usize,
    ) -> Self {
        let angular_pitch_rad = if element_count > 1 {
            (end_angle_rad - start_angle_rad) / (element_count - 1) as f64
        } else {
            0.0
        };
        Self {
            center_m,
            radius_m,
            first_angle_rad: start_angle_rad,
            angular_pitch_rad,
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

    /// Total signed angular aperture from first to last element [rad].
    #[must_use]
    pub fn aperture_angle_rad(&self) -> f64 {
        self.angular_pitch_rad * (self.element_count.saturating_sub(1)) as f64
    }

    pub(super) fn element(&self, index: usize) -> PlanarPoint {
        let angle = self.first_angle_rad + index as f64 * self.angular_pitch_rad;
        PlanarPoint {
            x_m: self.center_m.x_m + self.radius_m * angle.cos(),
            y_m: self.center_m.y_m + self.radius_m * angle.sin(),
        }
    }
}
