use super::mesh::TetrahedralMesh;
use super::types::MeshBoundaryType;
use crate::domain::grid::Grid;

#[test]
fn tetrahedron_volume_unit_is_one_sixth() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
    let e0 = mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    let vol = mesh.elements[e0].volume;
    assert!((vol - 1.0 / 6.0).abs() < 1e-12);
}

#[test]
fn point_location_barycentric_is_correct() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
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
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
    let n4 = mesh.add_node([0.0, 0.0, -1.0], MeshBoundaryType::Interior);
    let e0 = mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    let e1 = mesh.add_element([n0, n1, n2, n4], 0).unwrap();

    assert!(mesh.adjacency[e0].contains(&e1));
    assert!(mesh.adjacency[e1].contains(&e0));
}

#[test]
fn non_manifold_face_is_rejected() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
    let n4 = mesh.add_node([0.0, 0.0, -1.0], MeshBoundaryType::Interior);
    let n5 = mesh.add_node([0.0, 0.0, 2.0], MeshBoundaryType::Interior);

    mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    mesh.add_element([n0, n1, n2, n4], 0).unwrap();
    let err = mesh.add_element([n0, n1, n2, n5], 0).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("Non-manifold face encountered"));
}

#[test]
fn structured_grid_tetrahedralization_preserves_volume_and_counts() {
    let grid = Grid::new(3, 2, 2, 0.5, 2.0, 3.0).unwrap();

    let mesh = TetrahedralMesh::from_grid_vertices(&grid).unwrap();
    let stats = mesh.statistics();

    assert_eq!(stats.num_nodes, 12);
    assert_eq!(stats.num_elements, 12);
    assert_eq!(stats.num_boundary_faces, 20);
    assert!((stats.total_volume - 6.0).abs() < 1e-12);
    for element in &mesh.elements {
        assert!((element.volume - 0.5).abs() < 1e-12);
        assert!(element.quality > 0.0);
    }
    assert_eq!(mesh.nodes[0].coordinates, [0.0, 0.0, 0.0]);
    assert_eq!(mesh.nodes[11].coordinates, [1.0, 2.0, 3.0]);
}

#[test]
fn structured_grid_tetrahedralization_rejects_non_volume_grid() {
    let grid = Grid::new(2, 2, 1, 1.0, 1.0, 1.0).unwrap();

    let error = TetrahedralMesh::from_grid_vertices(&grid).unwrap_err();

    assert!(format!("{error}").contains("at least 2 vertices per axis"));
}
