use super::config::FdtdFemCouplingConfig;
use super::solver::FdtdFemSolver;
use kwavers_grid::Grid;
use kwavers_domain::mesh::tetrahedral::TetrahedralMesh;
use kwavers_domain::mesh::MeshBoundaryType;

#[test]
fn test_fdtd_fem_coupling_creation() {
    let fdtd_grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

    let mut fem_mesh = TetrahedralMesh::new();
    for i in 5..10 {
        for j in 5..10 {
            for k in 5..10 {
                let (x, y, z) = fdtd_grid.indices_to_coordinates(i, j, k);
                fem_mesh.add_node([x, y, z], MeshBoundaryType::Interior);
            }
        }
    }

    let config = FdtdFemCouplingConfig::default();
    let solver = FdtdFemSolver::new(config, fdtd_grid, fem_mesh).unwrap();
    // Fresh solver: FDTD field is all-zero (ndarray zeros), FEM field is empty history.
    assert_eq!(
        solver.fdtd_field().dim(),
        (10, 10, 10),
        "fdtd_field must match 10×10×10 grid"
    );
    assert!(
        solver.fdtd_field().iter().all(|&v| v == 0.0),
        "initial fdtd_field must be all-zero"
    );
}

#[test]
fn test_schwarz_iteration_convergence() {
    let fdtd_grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();

    let mut fem_mesh = TetrahedralMesh::new();
    let (x0, y0, z0) = fdtd_grid.indices_to_coordinates(4, 4, 4);
    let (x1, y1, z1) = fdtd_grid.indices_to_coordinates(5, 4, 4);
    let (x2, y2, z2) = fdtd_grid.indices_to_coordinates(4, 5, 4);
    let (x3, y3, z3) = fdtd_grid.indices_to_coordinates(4, 4, 5);
    let n0 = fem_mesh.add_node([x0, y0, z0], MeshBoundaryType::Interior);
    let n1 = fem_mesh.add_node([x1, y1, z1], MeshBoundaryType::Interior);
    let n2 = fem_mesh.add_node([x2, y2, z2], MeshBoundaryType::Interior);
    let n3 = fem_mesh.add_node([x3, y3, z3], MeshBoundaryType::Interior);
    fem_mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    let config = FdtdFemCouplingConfig {
        max_iterations: 5,
        tolerance: 1e-8,
        ..Default::default()
    };

    let mut solver = FdtdFemSolver::new(config, fdtd_grid, fem_mesh).unwrap();
    solver.step().expect("Time step should succeed");

    let history = solver.convergence_history();
    assert!(!history.is_empty(), "Should have convergence history");

    let final_residual = *history.last().unwrap();
    assert!(final_residual >= 0.0, "Residual should be non-negative");
}
