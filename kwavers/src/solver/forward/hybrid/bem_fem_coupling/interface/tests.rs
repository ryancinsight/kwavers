use super::*;
use crate::domain::mesh::tetrahedral::MeshBoundaryType;

#[test]
fn test_bem_fem_interface_creation() {
    // Create simple tetrahedral mesh
    let mut fem_mesh = TetrahedralMesh::new();

    // Add some nodes
    fem_mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    fem_mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    fem_mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    fem_mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);

    // Define BEM boundary elements (placeholder)
    let bem_boundary = vec![0, 1, 2];

    let interface = BemFemInterface::new(&fem_mesh, &bem_boundary);

    // Interface creation should succeed even with simplified geometry
    let interface = interface.unwrap();
    assert_eq!(interface.fem_interface_nodes.len(), 3);
    // Verify node 3 is NOT in interface
    assert!(!interface.fem_interface_nodes.contains(&3));
}

#[test]
fn test_bem_fem_interface_geometric_match() {
    let mut fem_mesh = TetrahedralMesh::new();
    // Node 0: on boundary
    let n0 = fem_mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    // Node 1: duplicate of n0, but different index. Should be detected via geometric check.
    let n1 = fem_mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);

    // BEM boundary uses only n0
    let bem_boundary = vec![n0];

    let interface = BemFemInterface::new(&fem_mesh, &bem_boundary).unwrap();

    // Both n0 (index match) and n1 (geometric match) should be in interface
    assert!(interface.fem_interface_nodes.contains(&n0));
    assert!(interface.fem_interface_nodes.contains(&n1));
    assert_eq!(interface.fem_interface_nodes.len(), 2);
}

#[test]
fn test_find_corresponding_bem_element() {
    let mut fem_mesh = TetrahedralMesh::new();

    // Node 0: Origin (Query node)
    let n0 = fem_mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);

    // Node 1: (1, 0, 0)
    let n1 = fem_mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);

    // Node 2: (2, 0, 0)
    let n2 = fem_mesh.add_node([2.0, 0.0, 0.0], MeshBoundaryType::Interior);

    // Node 3: (0.5, 0, 0) - Closest
    let n3 = fem_mesh.add_node([0.5, 0.0, 0.0], MeshBoundaryType::Interior);

    // BEM boundary candidates: n1, n2, n3
    let bem_boundary = vec![n1, n2, n3];

    let query_node = fem_mesh.nodes[n0];

    // Access the private function via the associated function on the struct
    // Since we are in a child module, we can access private items of parent
    let closest_idx =
        BemFemInterface::find_corresponding_bem_element(&query_node, &bem_boundary, &fem_mesh);

    assert_eq!(
        closest_idx.unwrap(),
        n3,
        "Should find the node at distance 0.5"
    );
}

#[test]
fn test_compute_interface_normals_calculation() {
    let mut fem_mesh = TetrahedralMesh::new();

    // Create a single tetrahedron
    // n0=(0,0,0), n1=(1,0,0), n2=(0,1,0), n3=(0,0,1)
    // Orientation: right-hand rule
    let n0 = fem_mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = fem_mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = fem_mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = fem_mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);

    // Add element. This will compute adjacency and boundary faces.
    fem_mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    // We want to test normals for the boundary nodes.
    // All 4 nodes are on the boundary of this single tetrahedron.
    let nodes = vec![n0, n1, n2, n3];

    let normals = BemFemInterface::compute_interface_normals(&nodes, &fem_mesh);

    assert_eq!(normals.len(), 4);

    // Check n0 (origin).
    // Shared by z=0, y=0, x=0 faces.
    // Normals: (0,0,-1), (0,-1,0), (-1,0,0). Area weighted (0.5 each).
    // Sum direction should be (-1, -1, -1).
    let normal_n0 = normals[0];
    let val = -1.0 / 3.0_f64.sqrt();
    assert!(
        (normal_n0.0 - val).abs() < 1e-6,
        "n0 x failed: expected {}, got {}",
        val,
        normal_n0.0
    );
    assert!(
        (normal_n0.1 - val).abs() < 1e-6,
        "n0 y failed: expected {}, got {}",
        val,
        normal_n0.1
    );
    assert!(
        (normal_n0.2 - val).abs() < 1e-6,
        "n0 z failed: expected {}, got {}",
        val,
        normal_n0.2
    );

    // Check n3 (0,0,1).
    // Shared by faces with normals (-1,0,0), (0,-1,0), (1,1,1).
    // Weighted sum is (0,0,1).
    let normal_n3 = normals[3];
    assert!((normal_n3.0 - 0.0).abs() < 1e-6, "n3 x failed");
    assert!((normal_n3.1 - 0.0).abs() < 1e-6, "n3 y failed");
    assert!((normal_n3.2 - 1.0).abs() < 1e-6, "n3 z failed");

    // Similarly for n1 (1,0,0) -> (1, 0, 0)
    let normal_n1 = normals[1];
    assert!((normal_n1.0 - 1.0).abs() < 1e-6, "n1 x failed");
    assert!((normal_n1.1 - 0.0).abs() < 1e-6, "n1 y failed");
    assert!((normal_n1.2 - 0.0).abs() < 1e-6, "n1 z failed");

    // Similarly for n2 (0,1,0) -> (0, 1, 0)
    let normal_n2 = normals[2];
    assert!((normal_n2.0 - 0.0).abs() < 1e-6, "n2 x failed");
    assert!((normal_n2.1 - 1.0).abs() < 1e-6, "n2 y failed");
    assert!((normal_n2.2 - 0.0).abs() < 1e-6, "n2 z failed");
}
