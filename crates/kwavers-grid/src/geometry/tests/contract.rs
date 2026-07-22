use super::super::{
    GeometricDomain, GeometryDimension, GeometryError, PointLocation, RectangularDomain,
    SphericalDomain,
};
use super::assert_within_absolute_error;

#[test]
fn rectangular_domain_classifies_each_supported_dimension() {
    let interval = RectangularDomain::new_1d(0.0, 10.0).expect("valid interval");
    assert_eq!(interval.dimension(), GeometryDimension::One);
    assert!(interval.contains(&[5.0]));
    assert!(!interval.contains(&[15.0]));
    assert!(!interval.contains(&[5.0, 0.0]));

    let rectangle = RectangularDomain::new_2d(0.0, 1.0, 0.0, 2.0).expect("valid rectangle");
    assert_eq!(rectangle.dimension(), GeometryDimension::Two);
    assert!(rectangle.contains(&[0.5, 1.0]));
    assert!(!rectangle.contains(&[1.5, 1.0]));

    let cuboid = RectangularDomain::new_3d(-1.0, 1.0, -2.0, 2.0, -3.0, 3.0).expect("valid cuboid");
    assert_eq!(cuboid.dimension(), GeometryDimension::Three);
    assert_within_absolute_error(cuboid.measure(), 48.0, 8.0 * f64::EPSILON);
    assert_eq!(cuboid.maximum_extent(), 6.0);
}

#[test]
fn constructors_reject_non_finite_unordered_and_unrepresentable_domains() {
    assert!(matches!(
        RectangularDomain::new_1d(f64::NAN, 1.0),
        Err(GeometryError::InvalidBounds { axis: 0, .. })
    ));
    assert!(matches!(
        RectangularDomain::new_1d(1.0, 1.0),
        Err(GeometryError::InvalidBounds { axis: 0, .. })
    ));
    assert!(matches!(
        RectangularDomain::new_1d(0.0, 0.0_f64.next_up()),
        Err(GeometryError::InvalidBounds { axis: 0, .. })
    ));
    assert!(matches!(
        SphericalDomain::new_2d(f64::INFINITY, 0.0, 1.0),
        Err(GeometryError::InvalidCenter { axis: 0, .. })
    ));
    assert!(matches!(
        SphericalDomain::new_3d(0.0, 0.0, 0.0, 0.0),
        Err(GeometryError::InvalidRadius { .. })
    ));
    assert!(matches!(
        SphericalDomain::new_2d(f64::MAX, 0.0, 1.0),
        Err(GeometryError::InvalidRadius { .. })
    ));
}

#[test]
fn rectangular_mapping_is_failure_atomic_and_strictly_interior() {
    let domain = RectangularDomain::new_2d(-2.0, 2.0, 10.0, 14.0).expect("valid rectangle");
    let mut output = [41.0, 43.0];
    let error = domain
        .map_unit_interior(&[0.5, 1.0], &mut output)
        .expect_err("one is outside the half-open unit interval");
    assert!(matches!(
        error,
        GeometryError::InvalidUnitCoordinate { axis: 1, .. }
    ));
    assert_eq!(output, [41.0, 43.0]);

    domain
        .map_unit_interior(&[0.0, 0.5], &mut output)
        .expect("valid normalized point");
    assert_eq!(output[0], (-2.0_f64).next_up());
    assert_eq!(output[1], 12.0);
    assert_eq!(domain.classify_point(&output, 0.0), PointLocation::Interior);
}

#[test]
fn disk_and_ball_maps_follow_inverse_measure_reference_cases() {
    let disk = SphericalDomain::new_2d(1.0, -1.0, 2.0).expect("valid disk");
    assert_eq!(disk.maximum_extent(), 4.0);
    let mut disk_point = [0.0; 2];
    disk.map_unit_interior(&[0.25, 0.0], &mut disk_point)
        .expect("valid disk coordinate");
    assert_within_absolute_error(disk_point[0], 2.0, 8.0 * f64::EPSILON);
    assert_within_absolute_error(disk_point[1], -1.0, 8.0 * f64::EPSILON);

    let ball = SphericalDomain::new_3d(1.0, -1.0, 3.0, 2.0).expect("valid ball");
    let mut ball_point = [0.0; 3];
    ball.map_unit_interior(&[0.125, 0.5, 0.0], &mut ball_point)
        .expect("valid ball coordinate");
    assert_within_absolute_error(ball_point[0], 2.0, 16.0 * f64::EPSILON);
    assert_within_absolute_error(ball_point[1], -1.0, 16.0 * f64::EPSILON);
    assert_within_absolute_error(ball_point[2], 3.0, 16.0 * f64::EPSILON);
}

#[test]
fn spherical_mapping_corrects_translated_boundary_roundoff() {
    let center = 1.0e16;
    let radius = 4.0;
    let radial_unit = 1.0_f64.next_down();

    let disk = SphericalDomain::new_2d(center, -center, radius).expect("valid translated disk");
    let mut disk_point = [0.0; 2];
    disk.map_unit_interior(&[radial_unit, 0.0], &mut disk_point)
        .expect("valid disk coordinate");
    assert_eq!(
        disk.classify_point(&disk_point, 0.0),
        PointLocation::Interior
    );
    assert!(disk_point[0] > center);

    let ball =
        SphericalDomain::new_3d(center, -center, center, radius).expect("valid translated ball");
    let mut ball_point = [0.0; 3];
    ball.map_unit_interior(&[radial_unit, 0.0, 0.0], &mut ball_point)
        .expect("valid ball coordinate");
    assert_eq!(
        ball.classify_point(&ball_point, 0.0),
        PointLocation::Interior
    );
    assert!(ball_point[2] > center);
}

#[test]
fn normals_and_invalid_queries_preserve_total_classification() {
    let rectangle = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0).expect("valid rectangle");
    let normal = rectangle.normal(&[0.0, 0.5], 0.0).expect("boundary normal");
    assert_eq!(normal[0], -1.0);
    assert_eq!(normal[1], 0.0);
    assert_eq!(
        rectangle.classify_point(&[0.5], 0.0),
        PointLocation::Exterior
    );
    assert_eq!(
        rectangle.classify_point(&[0.5, 0.5], f64::NAN),
        PointLocation::Exterior
    );

    let disk = SphericalDomain::new_2d(0.0, 0.0, 1.0).expect("valid disk");
    assert!(disk.contains(&[0.5, 0.0]));
    assert!(!disk.contains(&[1.5, 0.0]));
    assert!(!disk.contains(&[f64::NAN, 0.0]));
    let approximate_normal = disk
        .normal(&[0.5, 0.0], 0.5)
        .expect("non-central point lies within the boundary tolerance");
    assert_eq!(approximate_normal[0], 1.0);
    assert_eq!(approximate_normal[1], 0.0);
    assert!(disk.normal(&[0.0, 0.0], 1.0).is_none());
}
