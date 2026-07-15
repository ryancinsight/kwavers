//! Tests for `Geometry3D` spatial domain primitives.

use super::*;

#[test]
fn test_rectangular_geometry() {
    let geom = Geometry3D::rectangular(0.0, 2.0, 1.0, 3.0, -1.0, 1.0);
    if let Geometry3D::Rectangular {
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    } = geom
    {
        assert_eq!(x_min, 0.0);
        assert_eq!(x_max, 2.0);
        assert_eq!(y_min, 1.0);
        assert_eq!(y_max, 3.0);
        assert_eq!(z_min, -1.0);
        assert_eq!(z_max, 1.0);
    } else {
        panic!("Expected Rectangular geometry");
    }
}

#[test]
fn test_spherical_geometry() {
    let geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    if let Geometry3D::Spherical {
        x_center,
        y_center,
        z_center,
        radius,
    } = geom
    {
        assert_eq!(x_center, 0.5);
        assert_eq!(y_center, 0.5);
        assert_eq!(z_center, 0.5);
        assert_eq!(radius, 0.3);
    } else {
        panic!("Expected Spherical geometry");
    }
}

#[test]
fn test_cylindrical_geometry() {
    let geom = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);
    if let Geometry3D::Cylindrical {
        x_center,
        y_center,
        z_min,
        z_max,
        radius,
    } = geom
    {
        assert_eq!(x_center, 0.0);
        assert_eq!(y_center, 0.0);
        assert_eq!(z_min, -1.0);
        assert_eq!(z_max, 1.0);
        assert_eq!(radius, 0.5);
    } else {
        panic!("Expected Cylindrical geometry");
    }
}

#[test]
fn test_bounding_box_rectangular() {
    let geom = Geometry3D::rectangular(0.0, 2.0, 1.0, 3.0, -1.0, 1.0);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
    assert_eq!(x_min, 0.0);
    assert_eq!(x_max, 2.0);
    assert_eq!(y_min, 1.0);
    assert_eq!(y_max, 3.0);
    assert_eq!(z_min, -1.0);
    assert_eq!(z_max, 1.0);
}

#[test]
fn test_bounding_box_spherical() {
    let geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
    assert!((x_min - 0.2).abs() < 1e-10);
    assert!((x_max - 0.8).abs() < 1e-10);
    assert!((y_min - 0.2).abs() < 1e-10);
    assert!((y_max - 0.8).abs() < 1e-10);
    assert!((z_min - 0.2).abs() < 1e-10);
    assert!((z_max - 0.8).abs() < 1e-10);
}

#[test]
fn test_bounding_box_cylindrical() {
    let geom = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
    assert_eq!(x_min, -0.5);
    assert_eq!(x_max, 0.5);
    assert_eq!(y_min, -0.5);
    assert_eq!(y_max, 0.5);
    assert_eq!(z_min, -1.0);
    assert_eq!(z_max, 1.0);
}

#[test]
fn test_contains_rectangular() {
    let geom = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    assert!(geom.contains(0.5, 0.5, 0.5));
    assert!(geom.contains(0.0, 0.0, 0.0));
    assert!(geom.contains(1.0, 1.0, 1.0));

    assert!(!geom.contains(1.5, 0.5, 0.5));
    assert!(!geom.contains(0.5, 1.5, 0.5));
    assert!(!geom.contains(0.5, 0.5, 1.5));
    assert!(!geom.contains(-0.1, 0.5, 0.5));
}

#[test]
fn test_contains_spherical() {
    let geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);

    assert!(geom.contains(0.5, 0.5, 0.5));
    assert!(geom.contains(0.8, 0.5, 0.5));

    assert!(!geom.contains(1.0, 0.5, 0.5));
    assert!(!geom.contains(0.0, 0.5, 0.5));
}

#[test]
fn test_contains_cylindrical() {
    let geom = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);

    assert!(geom.contains(0.0, 0.0, 0.0));
    assert!(geom.contains(0.5, 0.0, 0.0));
    assert!(geom.contains(0.0, 0.5, 0.0));

    assert!(!geom.contains(1.0, 0.0, 0.0));
    assert!(!geom.contains(0.0, 0.0, 1.5));
    assert!(!geom.contains(0.0, 0.0, -1.5));
    assert!(!geom.contains(0.2, 0.2, 2.0));
}
