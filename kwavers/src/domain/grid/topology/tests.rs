use super::{CartesianTopology, CylindricalTopology, GridTopology, TopologyDimension};

#[test]
fn test_cartesian_creation() {
    let topo = CartesianTopology::new([10, 10, 10], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]).unwrap();
    assert_eq!(topo.size(), 1000);
    assert_eq!(topo.dimensionality(), TopologyDimension::Three);
}

#[test]
fn test_cartesian_coordinates() {
    let topo = CartesianTopology::new([10, 10, 10], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]).unwrap();

    let coords = topo.indices_to_coordinates([5, 5, 5]);
    assert!((coords[0] - 0.5).abs() < 1e-10);
    assert!((coords[1] - 0.5).abs() < 1e-10);
    assert!((coords[2] - 0.5).abs() < 1e-10);

    let indices = topo.coordinates_to_indices([0.5, 0.5, 0.5]).unwrap();
    assert_eq!(indices, [5, 5, 5]);
}

#[test]
fn test_cartesian_metric() {
    let topo = CartesianTopology::new([10, 10, 10], [1e-3, 2e-3, 3e-3], [0.0, 0.0, 0.0]).unwrap();

    let metric = topo.metric_coefficient([5, 5, 5]);
    assert!((metric - 6e-9).abs() < 1e-15);
}

#[test]
fn test_cylindrical_creation() {
    let topo = CylindricalTopology::new(64, 32, 1e-4, 1e-4).unwrap();
    assert_eq!(topo.nz, 64);
    assert_eq!(topo.nr, 32);
    assert_eq!(topo.size(), 64 * 32);
    assert_eq!(topo.dimensionality(), TopologyDimension::Two);
}

#[test]
fn test_cylindrical_coordinates() {
    let topo = CylindricalTopology::new(10, 10, 0.1, 0.1).unwrap();

    let coords = topo.indices_to_coordinates([5, 3, 0]);
    assert!((coords[0] - 0.5).abs() < 1e-10);
    assert!((coords[1] - 0.3).abs() < 1e-10);

    let indices = topo.coordinates_to_indices([0.5, 0.3, 0.0]).unwrap();
    assert_eq!(indices[0], 5);
    assert_eq!(indices[1], 3);
}

#[test]
fn test_cylindrical_metric() {
    let topo = CylindricalTopology::new(10, 10, 0.1, 0.1).unwrap();

    let metric_0 = topo.metric_coefficient([5, 0, 0]);
    assert!((metric_0 - 0.5 * 0.1 * 0.1 * 0.1).abs() < 1e-10);

    let metric_3 = topo.metric_coefficient([5, 3, 0]);
    assert!((metric_3 - 0.3 * 0.1 * 0.1).abs() < 1e-10);
}

#[test]
fn test_cylindrical_wavenumbers() {
    let topo = CylindricalTopology::new(64, 32, 1e-4, 1e-4).unwrap();

    assert_eq!(topo.kz_wavenumbers()[0], 0.0);
    assert_eq!(topo.kr_wavenumbers()[0], 0.0);

    assert!(topo.kz_wavenumbers()[1] > 0.0);
    assert!(topo.kr_wavenumbers()[1] > 0.0);
}

#[test]
fn test_cylindrical_coordinate_accessors() {
    let topo = CylindricalTopology::new(10, 10, 0.1, 0.1).unwrap();

    assert_eq!(topo.z_at(0), 0.0);
    assert!((topo.z_at(5) - 0.5).abs() < 1e-10);

    assert_eq!(topo.r_at(0), 0.0);
    assert!((topo.r_at(3) - 0.3).abs() < 1e-10);

    assert_eq!(topo.iz_for(0.5), 5);
    assert_eq!(topo.ir_for(0.3), 3);

    assert!((topo.z_max() - 0.9).abs() < 1e-10);
    assert!((topo.r_max() - 0.9).abs() < 1e-10);
}

#[test]
fn test_cylindrical_meshgrid() {
    let topo = CylindricalTopology::new(4, 3, 1.0, 1.0).unwrap();

    let z_mesh = topo.z_mesh();
    let r_mesh = topo.r_mesh();

    assert_eq!(z_mesh.dim(), (4, 3));
    assert_eq!(r_mesh.dim(), (4, 3));

    assert_eq!(z_mesh[[2, 0]], 2.0);
    assert_eq!(r_mesh[[0, 2]], 2.0);
}

#[test]
fn test_topology_field_creation() {
    let cart = CartesianTopology::new([10, 10, 10], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]).unwrap();
    let field = cart.create_field();
    assert_eq!(field.dim(), (10, 10, 10));

    let cyl = CylindricalTopology::new(64, 32, 1e-4, 1e-4).unwrap();
    let cyl_field = cyl.create_field();
    assert_eq!(cyl_field.dim(), (64, 32, 1));
}

#[test]
fn test_invalid_dimensions() {
    let result = CartesianTopology::new([0, 10, 10], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]);
    assert!(result.is_err());

    let result = CylindricalTopology::new(0, 10, 1e-3, 1e-3);
    assert!(result.is_err());
}

#[test]
fn test_invalid_spacing() {
    let result = CartesianTopology::new([10, 10, 10], [0.0, 1e-3, 1e-3], [0.0, 0.0, 0.0]);
    assert!(result.is_err());

    let result = CartesianTopology::new([10, 10, 10], [-1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]);
    assert!(result.is_err());

    let result = CylindricalTopology::new(10, 10, f64::INFINITY, 1e-3);
    assert!(result.is_err());
}
