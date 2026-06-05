use super::*;
use crate::forward::bem::field::BemSolution;
use kwavers_mesh::tetrahedral::{MeshBoundaryType, TetrahedralMesh};
use ndarray::Array1;
use num_complex::Complex64;

fn create_test_mesh() -> TetrahedralMesh {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
    mesh.add_element([n0, n1, n2, n3], 0).unwrap();
    mesh
}

#[test]
fn test_bem_solver_creation() {
    let config = BemConfig::default();
    let mesh = create_test_mesh();
    let solver = BemSolver::from_mesh(config, &mesh).unwrap();

    assert_eq!(solver.vertices.len(), 4);
    assert!(solver.boundary_manager_ref().is_empty());
    assert!(solver.h_matrix.is_none());
    assert!(solver.g_matrix.is_none());
}

#[test]
fn test_bem_system_assembly() {
    let config = BemConfig::default();
    let mesh = create_test_mesh();
    let mut solver = BemSolver::from_mesh(config, &mesh).unwrap();

    solver.assemble_system().unwrap();

    let h = solver.h_matrix.unwrap();
    let g = solver.g_matrix.unwrap();

    for i in 0..4 {
        let diag = h.get_diagonal(i);
        assert!((diag.re - 0.5).abs() < 1e-6, "Diagonal H should be 0.5");
    }

    for i in 0..4 {
        let diag = g.get_diagonal(i);
        assert!(diag.norm() > 1e-6, "Diagonal G should be non-zero");
    }
}

#[test]
fn test_bem_boundary_conditions() {
    let config = BemConfig::default();
    let mesh = create_test_mesh();
    let mut solver = BemSolver::from_mesh(config, &mesh).unwrap();

    {
        let bc_manager = solver.boundary_manager();
        bc_manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);
        bc_manager.add_radiation(vec![1, 2, 3]);
    }

    assert_eq!(solver.boundary_manager_ref().len(), 2);

    solver.assemble_system().unwrap();
    let solution = solver.solve(1.0, None).unwrap();

    assert_eq!(solution.boundary_pressure.len(), 4);
    assert_eq!(solution.boundary_velocity.len(), 4);
    assert_eq!(solution.wavenumber, 1.0);
}

#[test]
fn test_compute_scattered_field() {
    let config = BemConfig::default();
    let mesh = create_test_mesh();
    let solver = BemSolver::from_mesh(config, &mesh).unwrap();

    let n = solver.vertices.len();
    let boundary_pressure = Array1::from_elem(n, Complex64::new(1.0, 0.0));
    let boundary_velocity = Array1::from_elem(n, Complex64::new(0.0, 0.0));

    let solution = BemSolution {
        boundary_pressure,
        boundary_velocity,
        wavenumber: 1.0,
    };

    let points = Array1::from_vec(vec![[2.0, 2.0, 2.0]]);
    let field = solver.compute_scattered_field(&points, &solution).unwrap();

    assert_eq!(field.len(), 1);
    assert!(field[0].norm() > 1e-10, "Field should be non-zero");
}
