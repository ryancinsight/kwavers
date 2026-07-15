use std::f64::consts::PI;

use super::*;

#[test]
fn test_rectangular_1d() {
    let domain = RectangularDomain::new_1d(0.0, 10.0);
    assert_eq!(domain.dimension(), GeometryDimension::One);
    assert!(domain.contains(&[5.0]));
    assert!(!domain.contains(&[15.0]));
    assert_eq!(domain.measure(), 10.0);
}

#[test]
fn test_rectangular_2d() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 2.0);
    assert_eq!(domain.dimension(), GeometryDimension::Two);
    assert!(domain.contains(&[0.5, 1.0]));
    assert!(!domain.contains(&[1.5, 1.0]));
    assert_eq!(domain.measure(), 2.0);
}

#[test]
fn test_boundary_classification() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
    let tol = 1e-6;

    assert_eq!(
        domain.classify_point(&[0.5, 0.5], tol),
        PointLocation::Interior
    );
    assert_eq!(
        domain.classify_point(&[0.0, 0.5], tol),
        PointLocation::Boundary
    );
    assert_eq!(
        domain.classify_point(&[1.5, 0.5], tol),
        PointLocation::Exterior
    );
}

#[test]
fn test_normal_computation() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
    let tol = 1e-6;

    let normal = domain.normal(&[0.0, 0.5], tol).unwrap();
    assert!((normal[0] + 1.0).abs() < tol);
    assert!(normal[1].abs() < tol);

    let normal = domain.normal(&[1.0, 0.5], tol).unwrap();
    assert!((normal[0] - 1.0).abs() < tol);
}

#[test]
fn test_spherical_2d() {
    let domain = SphericalDomain::new_2d(0.0, 0.0, 1.0);
    assert_eq!(domain.dimension(), GeometryDimension::Two);
    assert!(domain.contains(&[0.5, 0.0]));
    assert!(!domain.contains(&[1.5, 0.0]));
    assert!((domain.measure() - PI).abs() < 1e-10);
}

#[test]
fn test_sampling() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
    let interior = domain.sample_interior(100, Some(42));
    let boundary = domain.sample_boundary(100, Some(42));

    assert_eq!(interior.shape(), [100, 2]);
    assert_eq!(boundary.shape(), [100, 2]);

    for i in 0..interior.shape()[0] {
        let point = [interior[[i, 0]], interior[[i, 1]]];
        assert!(domain.contains(&point));
    }

    let tol = 1e-10;
    for i in 0..boundary.shape()[0] {
        let point = [boundary[[i, 0]], boundary[[i, 1]]];
        assert_eq!(domain.classify_point(&point, tol), PointLocation::Boundary);
    }
}
