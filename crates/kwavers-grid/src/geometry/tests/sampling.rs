use tyche_core::{Design, Seed};

use super::super::sampling::DesignSamplingExt;
use super::super::{
    GeometricDomain, GeometryError, PointLocation, RectangularDomain, SphericalDomain,
};
use super::assert_within_absolute_error;

#[test]
fn all_domain_samplers_have_exact_cardinality_replay_and_membership() {
    let seed = Seed::new(42);
    let rectangle = RectangularDomain::new_2d(-1.0, 2.0, 3.0, 5.0).expect("valid rectangle");
    let disk = SphericalDomain::new_2d(1.0, -2.0, 3.0).expect("valid disk");
    let ball = SphericalDomain::new_3d(1.0, -2.0, 3.0, 4.0).expect("valid ball");

    for domain in [&rectangle as &dyn GeometricDomain, &disk, &ball] {
        let interior = domain
            .sample_interior(257, seed)
            .expect("addressable output");
        let replay = domain
            .sample_interior(257, seed)
            .expect("addressable replay");
        assert_eq!(interior, replay);
        assert_eq!(interior.shape(), [257, domain.dimension().as_usize()]);
        for row in 0..interior.shape()[0] {
            let point = (0..interior.shape()[1])
                .map(|column| interior[[row, column]])
                .collect::<Vec<_>>();
            assert_eq!(domain.classify_point(&point, 0.0), PointLocation::Interior);
        }

        let boundary = domain
            .sample_boundary(257, seed)
            .expect("addressable output");
        let boundary_replay = domain
            .sample_boundary(257, seed)
            .expect("addressable replay");
        assert_eq!(boundary, boundary_replay);
        assert_eq!(boundary.shape(), [257, domain.dimension().as_usize()]);
        let tolerance = 64.0 * f64::EPSILON * domain.maximum_extent();
        for row in 0..boundary.shape()[0] {
            let point = (0..boundary.shape()[1])
                .map(|column| boundary[[row, column]])
                .collect::<Vec<_>>();
            assert_eq!(
                domain.classify_point(&point, tolerance),
                PointLocation::Boundary
            );
        }
    }
}

#[test]
fn rectangular_boundary_faces_follow_surface_measure() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 9.0).expect("valid rectangle");
    let sample_count = 10_000;
    let points = domain
        .sample_boundary(sample_count, Seed::new(91))
        .expect("addressable output");
    let vertical_faces = (0..sample_count)
        .filter(|&row| points[[row, 0]] == 0.0 || points[[row, 0]] == 1.0)
        .count();

    // The two vertical faces have total measure 18 and the two horizontal
    // faces measure 2, so p=0.9. Six binomial standard deviations give
    // 6*sqrt(n*p*(1-p)) = 180 observations.
    let expected = 9_000_i64;
    let observed = i64::try_from(vertical_faces).expect("sample count fits i64");
    assert!((observed - expected).abs() <= 180);
}

#[derive(Debug, Clone, Copy)]
struct OneDimensionalDesign;

impl Design<1> for OneDimensionalDesign {
    fn sample_count(&self) -> usize {
        1
    }

    fn sample_unit_into(
        &self,
        index: usize,
        output: &mut [f64; 1],
    ) -> Result<(), tyche_core::SampleIndexError> {
        if index != 0 {
            return Err(tyche_core::SampleIndexError::new(index, 1));
        }
        output[0] = 0.5;
        Ok(())
    }
}

#[test]
fn design_bridge_rejects_dimension_mismatch_before_collection() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0).expect("valid rectangle");
    assert!(matches!(
        domain.sample_design(&OneDimensionalDesign),
        Err(GeometryError::DimensionMismatch {
            role: "design",
            expected: 2,
            actual: 1,
        })
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn addressable_but_unreservable_output_returns_a_typed_error() {
    let domain = RectangularDomain::new_1d(0.0, 1.0).expect("valid interval");
    assert!(matches!(
        domain.sample_interior(usize::MAX, Seed::new(3)),
        Err(GeometryError::AllocationFailed {
            element_count: usize::MAX
        })
    ));
}

#[test]
fn sampled_spherical_radial_moments_match_uniform_measure() {
    let disk = SphericalDomain::new_2d(0.0, 0.0, 1.0).expect("valid disk");
    let ball = SphericalDomain::new_3d(0.0, 0.0, 0.0, 1.0).expect("valid ball");
    let sample_count = 8_192;
    let disk_points = disk
        .sample_interior(sample_count, Seed::new(7))
        .expect("addressable disk output");
    let ball_points = ball
        .sample_interior(sample_count, Seed::new(7))
        .expect("addressable ball output");

    let count = f64::from(u32::try_from(sample_count).expect("sample count fits u32"));
    let disk_mean_radius_squared = (0..sample_count)
        .map(|row| {
            disk_points[[row, 0]].mul_add(disk_points[[row, 0]], disk_points[[row, 1]].powi(2))
        })
        .sum::<f64>()
        / count;
    let ball_mean_radius_cubed = (0..sample_count)
        .map(|row| {
            let squared = ball_points[[row, 0]].mul_add(
                ball_points[[row, 0]],
                ball_points[[row, 1]].mul_add(ball_points[[row, 1]], ball_points[[row, 2]].powi(2)),
            );
            squared.sqrt().powi(3)
        })
        .sum::<f64>()
        / count;

    // The direct inverse transforms make r^2/R^2 and r^3/R^3 equal to the
    // underlying discrete uniform coordinate. Its mean differs from 1/2 by at
    // most the six-sigma Monte Carlo bound used here.
    let bound = 6.0 / (12.0 * count).sqrt();
    assert_within_absolute_error(disk_mean_radius_squared, 0.5, bound);
    assert_within_absolute_error(ball_mean_radius_cubed, 0.5, bound);
}
