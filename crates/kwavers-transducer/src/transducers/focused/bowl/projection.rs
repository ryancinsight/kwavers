//! Planar projection helpers for focused bowl source layouts.

use std::f64::consts::TAU;

use super::super::validation::{
    validate_element_count, validate_finite_vector, validate_positive_finite_field,
};
use super::BowlTransducer;
use kwavers_core::error::KwaversResult;

impl BowlTransducer {
    /// Build the transverse planar projection of an axisymmetric focused bowl.
    ///
    /// # Theorem
    ///
    /// For focus `f`, radius `R`, and element index `i`, this returns
    /// `x_i = f + R [cos(2 pi i / N), sin(2 pi i / N)]`. Thus every projected
    /// element satisfies `||x_i - f||_2 = R`, and adjacent azimuths are separated
    /// by exactly `2 pi / N`. Clinical adapters can use this as the 2-D slice
    /// projection of a focused bowl without duplicating source geometry.
    ///
    /// # Errors
    /// - Returns a validation error unless `focus_m` is finite.
    /// - Returns a validation error unless `radius_m` is positive and finite.
    /// - Returns a validation error unless `element_count` is nonzero.
    pub fn transverse_projection_ring(
        focus_m: [f64; 2],
        radius_m: f64,
        element_count: usize,
    ) -> KwaversResult<Vec<[f64; 2]>> {
        validate_finite_vector("focus_m", focus_m)?;
        validate_positive_finite_field("radius_m", radius_m)?;
        validate_element_count(element_count)?;

        Ok((0..element_count)
            .map(|idx| {
                let theta = TAU * idx as f64 / element_count as f64;
                [
                    theta.cos().mul_add(radius_m, focus_m[0]),
                    theta.sin().mul_add(radius_m, focus_m[1]),
                ]
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::error::KwaversError;

    #[test]
    fn transverse_projection_ring_is_focus_centered() {
        let focus = [0.012, -0.006];
        let radius = 0.145;
        let points = BowlTransducer::transverse_projection_ring(focus, radius, 16).unwrap();

        assert_eq!(points.len(), 16);
        for point in points {
            let distance = (point[0] - focus[0]).hypot(point[1] - focus[1]);
            assert_close(distance, radius);
        }
    }

    #[test]
    fn transverse_projection_ring_rejects_invalid_domain() {
        assert!(matches!(
            BowlTransducer::transverse_projection_ring([0.0, 0.0], 0.0, 16).unwrap_err(),
            KwaversError::Validation(_)
        ));
        assert!(matches!(
            BowlTransducer::transverse_projection_ring([f64::NAN, 0.0], 0.1, 16).unwrap_err(),
            KwaversError::Validation(_)
        ));
        assert!(matches!(
            BowlTransducer::transverse_projection_ring([0.0, 0.0], 0.1, 0).unwrap_err(),
            KwaversError::Validation(_)
        ));
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}
