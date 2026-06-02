use super::*;
use kwavers_domain::grid::Grid;
use kwavers_domain::mesh::TetrahedralMesh;

#[test]
fn test_pstd_sem_coupling_creation() {
    let pstd_grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();

    let mut sem_mesh = TetrahedralMesh::new();
    for i in 8..16 {
        for j in 8..16 {
            for k in 8..16 {
                let (x, y, z) = pstd_grid.indices_to_coordinates(i, j, k);
                sem_mesh.add_node(
                    [x, y, z],
                    kwavers_domain::mesh::tetrahedral::MeshBoundaryType::Interior,
                );
            }
        }
    }

    let config = PstdSemCouplingConfig::default();
    let solver = PstdSemSolver::new(config, pstd_grid, sem_mesh).unwrap();
    // Fresh solver: PSTD field is all-zero (16×16×16 grid), no convergence history.
    assert_eq!(
        solver.pstd_field().dim(),
        (16, 16, 16),
        "pstd_field must match 16×16×16 grid"
    );
    assert!(
        solver.pstd_field().iter().all(|&v| v == 0.0),
        "initial pstd_field must be all-zero"
    );
}

#[test]
fn test_spectral_coupling_convergence() {
    let pstd_grid = Grid::new(12, 12, 12, 0.001, 0.001, 0.001).unwrap();

    let mut sem_mesh = TetrahedralMesh::new();
    for i in 6..12 {
        for j in 6..12 {
            for k in 6..12 {
                let (x, y, z) = pstd_grid.indices_to_coordinates(i, j, k);
                sem_mesh.add_node(
                    [x, y, z],
                    kwavers_domain::mesh::tetrahedral::MeshBoundaryType::Interior,
                );
            }
        }
    }

    let config = PstdSemCouplingConfig {
        projection_tolerance: 1e-10,
        ..Default::default()
    };

    let mut solver = PstdSemSolver::new(config, pstd_grid, sem_mesh).unwrap();
    let residual = solver.step().unwrap();

    assert!(residual >= 0.0, "Residual should be non-negative");
    assert!(
        !solver.convergence_history().is_empty(),
        "Should have convergence history"
    );
}

#[test]
fn test_interface_detection() {
    let pstd_grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

    let mut sem_mesh = TetrahedralMesh::new();
    for i in 5..10 {
        for j in 5..10 {
            for k in 5..10 {
                let (x, y, z) = pstd_grid.indices_to_coordinates(i, j, k);
                sem_mesh.add_node(
                    [x, y, z],
                    kwavers_domain::mesh::tetrahedral::MeshBoundaryType::Interior,
                );
            }
        }
    }

    let config = PstdSemCouplingConfig::default();
    let interface = SpectralCouplingInterface::new(&pstd_grid, &sem_mesh, &config);

    let interface = interface.unwrap();
    assert!(
        !interface.pstd_interface_points.is_empty(),
        "Should find interface points"
    );
    assert!(
        !interface.sem_interface_nodes.is_empty(),
        "Should find interface nodes"
    );
}
