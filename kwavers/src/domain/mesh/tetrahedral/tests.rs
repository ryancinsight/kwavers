use super::mesh::TetrahedralMesh;
use super::types::BoundaryType;

#[test]
fn tetrahedron_volume_unit_is_one_sixth() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    let e0 = mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    let vol = mesh.elements[e0].volume;
    assert!((vol - 1.0 / 6.0).abs() < 1e-12);
}

#[test]
fn point_location_barycentric_is_correct() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    let inside = mesh.locate_point([0.1, 0.1, 0.1]);
    assert_eq!(inside, vec![0]);

    let on_face = mesh.locate_point([0.3, 0.3, 0.0]);
    assert_eq!(on_face, vec![0]);

    let outside = mesh.locate_point([1.0, 1.0, 1.0]);
    assert!(outside.is_empty());
}

#[test]
fn adjacency_is_symmetric_for_shared_face() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    let n4 = mesh.add_node([0.0, 0.0, -1.0], BoundaryType::Interior);
    let e0 = mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    let e1 = mesh.add_element([n0, n1, n2, n4], 0).unwrap();

    assert!(mesh.adjacency[e0].contains(&e1));
    assert!(mesh.adjacency[e1].contains(&e0));
}

#[test]
fn non_manifold_face_is_rejected() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    let n4 = mesh.add_node([0.0, 0.0, -1.0], BoundaryType::Interior);
    let n5 = mesh.add_node([0.0, 0.0, 2.0], BoundaryType::Interior);

    mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    mesh.add_element([n0, n1, n2, n4], 0).unwrap();
    let err = mesh.add_element([n0, n1, n2, n5], 0).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("Non-manifold face encountered"));
}
