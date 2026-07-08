//! Geometry and boundary condition type tests.

use super::super::*;

#[test]
fn test_boundary_condition_types() {
    let dirichlet = BoundaryCondition3D::Dirichlet;
    let neumann = BoundaryCondition3D::Neumann;
    let absorbing = BoundaryCondition3D::Absorbing;
    let periodic = BoundaryCondition3D::Periodic;

    assert!(matches!(dirichlet, BoundaryCondition3D::Dirichlet));
    assert!(matches!(neumann, BoundaryCondition3D::Neumann));
    assert!(matches!(absorbing, BoundaryCondition3D::Absorbing));
    assert!(matches!(periodic, BoundaryCondition3D::Periodic));
}

#[test]
fn test_geometry_bounding_box_variants() {
    let rect = Geometry3D::rectangular(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = rect.bounding_box();
    assert_eq!(
        (x_min, x_max, y_min, y_max, z_min, z_max),
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    );

    let sphere = Geometry3D::spherical(1.0, 2.0, 3.0, 0.5);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = sphere.bounding_box();
    assert!((x_min - 0.5).abs() < 1e-6);
    assert!((x_max - 1.5).abs() < 1e-6);
    assert!((y_min - 1.5).abs() < 1e-6);
    assert!((y_max - 2.5).abs() < 1e-6);
    assert!((z_min - 2.5).abs() < 1e-6);
    assert!((z_max - 3.5).abs() < 1e-6);

    let cylinder = Geometry3D::cylindrical(1.0, 2.0, 0.0, 5.0, 0.5);
    let (x_min, x_max, y_min, y_max, z_min, z_max) = cylinder.bounding_box();
    assert!((x_min - 0.5).abs() < 1e-6);
    assert!((x_max - 1.5).abs() < 1e-6);
    assert!((y_min - 1.5).abs() < 1e-6);
    assert!((y_max - 2.5).abs() < 1e-6);
    assert!((z_min - 0.0).abs() < 1e-6);
    assert!((z_max - 5.0).abs() < 1e-6);
}

#[test]
fn test_geometry_contains_variants() {
    let rect = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    assert!(rect.contains(0.5, 0.5, 0.5));
    assert!(!rect.contains(1.5, 0.5, 0.5));

    let sphere = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    assert!(sphere.contains(0.5, 0.5, 0.5));
    assert!(!sphere.contains(1.0, 0.5, 0.5));

    let cylinder = Geometry3D::cylindrical(0.5, 0.5, 0.0, 1.0, 0.3);
    assert!(cylinder.contains(0.5, 0.5, 0.5));
    assert!(!cylinder.contains(1.0, 0.5, 0.5));
    assert!(!cylinder.contains(0.5, 0.5, 1.5));
}
