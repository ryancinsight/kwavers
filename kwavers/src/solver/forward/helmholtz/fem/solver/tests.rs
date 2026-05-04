use super::config::{FemHelmholtzConfig, PreconditionerType};
use super::core::FemHelmholtzSolver;
use crate::core::error::KwaversError;
use crate::domain::grid::Grid;
use crate::domain::mesh::{BoundaryType, TetrahedralMesh};
use approx::assert_relative_eq;
use ndarray::arr2;
use num_complex::Complex64;

fn unit_tet() -> (TetrahedralMesh, [usize; 4]) {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    mesh.add_element([n0, n1, n2, n3], 0)
        .expect("Failed to add element");
    (mesh, [n0, n1, n2, n3])
}

fn homogeneous_medium() -> (crate::domain::medium::HomogeneousMedium, ()) {
    let grid = crate::domain::grid::Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
    (
        crate::domain::medium::HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid),
        (),
    )
}

/// Assemble K on a single-element mesh (k=0, Laplace).
///
/// Expected stiffness values from P1 P1 formulas (Ihlenburg 1998, §2.1):
/// - K_00 = 0.5  (gradient of node 0 dotted with itself × volume 1/6)
/// - K_11 = 1/6
/// - Row sum = 0 (partition of unity)
#[test]
fn test_assembly_one_element() {
    let (mesh, _) = unit_tet();
    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        ..Default::default()
    };
    let mut solver = FemHelmholtzSolver::new(config, mesh);
    let (medium, _) = homogeneous_medium();
    solver.assemble_system(&medium).expect("Assembly failed");

    let mat = &solver.system_matrix;
    assert_eq!(mat.rows, 4);
    assert_eq!(mat.cols, 4);

    let k00 = mat.get_diagonal(0).re;
    assert_relative_eq!(k00, 0.5, epsilon = 1e-10);

    let k11 = mat.get_diagonal(1).re;
    assert_relative_eq!(k11, 1.0 / 6.0, epsilon = 1e-10);

    let (vals, _) = mat.get_row(0);
    let sum_real: f64 = vals.iter().map(|c| c.re).sum();
    assert_relative_eq!(sum_real, 0.0, epsilon = 1e-10);
}

/// Barycentric interpolation of u = x + 2y + 3z on a unit tetrahedron.
///
/// At centroid (0.25, 0.25, 0.25): u = 0.25(1+2+3) = 1.5.
/// At vertex (1,0,0): u = 1.0.
/// Outside mesh: u = 0.0.
#[test]
fn test_interpolate_solution_basic() {
    let (mesh, [n0, n1, n2, n3]) = unit_tet();
    let config = FemHelmholtzConfig::default();
    let mut solver = FemHelmholtzSolver::new(config, mesh);

    solver.solution[n0] = Complex64::new(0.0, 0.0);
    solver.solution[n1] = Complex64::new(1.0, 0.0);
    solver.solution[n2] = Complex64::new(2.0, 0.0);
    solver.solution[n3] = Complex64::new(3.0, 0.0);

    let query_points = arr2(&[[0.25, 0.25, 0.25], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0]]);
    let result = solver
        .interpolate_solution(query_points.view())
        .expect("Interpolation failed");

    assert_eq!(result.len(), 3);
    assert_relative_eq!(result[0].re, 1.5, epsilon = 1e-10);
    assert_relative_eq!(result[0].im, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result[1].re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(result[2].re, 0.0, epsilon = 1e-10);
}

/// Dirichlet BCs on nodes 0/1/2; solve Laplace; node 3 must satisfy free-node equation.
///
/// Boundary: u(n0)=0, u(n1)=1, u(n2)=0. Free node n3 enforces row sum = 0 → u(n3)=0.
#[test]
fn test_solve_system_one_element_dirichlet() {
    let (mesh, [n0, n1, n2, _n3]) = unit_tet();
    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        tolerance: 1e-10,
        ..Default::default()
    };
    let mut solver = FemHelmholtzSolver::new(config, mesh);

    solver.boundary_manager().add_dirichlet(vec![
        (n0, Complex64::new(0.0, 0.0)),
        (n1, Complex64::new(1.0, 0.0)),
        (n2, Complex64::new(0.0, 0.0)),
    ]);

    let (medium, _) = homogeneous_medium();
    solver.assemble_system(&medium).expect("Assembly failed");
    solver.solve_system().expect("Solve failed");

    // n1 should remain at prescribed value
    let u1 = solver.solution()[n1];
    assert_relative_eq!(u1.re, 1.0, epsilon = 1e-6);

    // n3 (free): Laplace + BCs → solution must satisfy the linear system
    let u3 = solver.solution()[_n3];
    assert_relative_eq!(u3.re, 0.0, epsilon = 1e-6);
    assert_relative_eq!(u3.im, 0.0, epsilon = 1e-6);
}

#[test]
fn test_from_grid_structured_mesh_assembly() {
    let grid = Grid::new(2, 2, 2, 1.0, 2.0, 3.0).unwrap();
    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        ..Default::default()
    };
    let mut solver = FemHelmholtzSolver::from_grid(config, &grid).unwrap();
    let (medium, _) = homogeneous_medium();

    solver.assemble_system(&medium).unwrap();

    assert_eq!(solver.system_matrix.rows, 8);
    assert_eq!(solver.system_matrix.cols, 8);
    let diagonal_sum: f64 = (0..8)
        .map(|row| solver.system_matrix.get_diagonal(row).re)
        .sum();
    assert!(diagonal_sum > 0.0);
}

#[test]
fn test_ilu_and_amg_preconditioners_fail_explicitly() {
    for preconditioner in [PreconditionerType::ILU, PreconditionerType::AMG] {
        let (mesh, _) = unit_tet();
        let config = FemHelmholtzConfig {
            preconditioner,
            ..Default::default()
        };
        let mut solver = FemHelmholtzSolver::new(config, mesh);

        let error = solver.solve_system().unwrap_err();

        assert!(matches!(error, KwaversError::FeatureNotAvailable(_)));
    }
}

#[test]
fn test_exact_nodal_load_updates_rhs_after_assembly() {
    let (mesh, [_n0, n1, _n2, _n3]) = unit_tet();
    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        ..Default::default()
    };
    let mut solver = FemHelmholtzSolver::new(config, mesh);
    let (medium, _) = homogeneous_medium();
    solver.assemble_system(&medium).unwrap();

    solver
        .add_nodal_load(n1, Complex64::new(2.0, -0.5))
        .unwrap();

    assert_eq!(solver.rhs()[n1], Complex64::new(2.0, -0.5));
    assert!(solver.add_nodal_load(99, Complex64::new(1.0, 0.0)).is_err());
    assert!(solver
        .add_nodal_load(n1, Complex64::new(f64::NAN, 0.0))
        .is_err());
}

#[test]
fn test_boundary_type_dirichlet_helper_applies_to_tagged_nodes() {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Dirichlet);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Dirichlet);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
    mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        ..Default::default()
    };
    let mut solver = FemHelmholtzSolver::new(config, mesh);
    let count = solver
        .add_dirichlet_on_boundary_type(BoundaryType::Dirichlet, Complex64::new(3.0, 0.0))
        .unwrap();
    let (medium, _) = homogeneous_medium();
    solver.assemble_system(&medium).unwrap();

    assert_eq!(count, 2);
    assert_eq!(solver.rhs()[n0], Complex64::new(3.0, 0.0));
    assert_eq!(solver.rhs()[n2], Complex64::new(3.0, 0.0));
    assert_eq!(
        solver.system_matrix.get_diagonal(n0),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(
        solver.system_matrix.get_diagonal(n2),
        Complex64::new(1.0, 0.0)
    );
}
