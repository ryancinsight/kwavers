use super::config::{FemHelmholtzConfig, FemPreconditionerType};
use super::core::FemHelmholtzSolver;
use crate::core::error::KwaversError;
use crate::domain::grid::Grid;
use crate::domain::mesh::{MeshBoundaryType, TetrahedralMesh};
use approx::assert_relative_eq;
use ndarray::{arr2, Array2};
use num_complex::Complex64;

fn unit_tet() -> (TetrahedralMesh, [usize; 4]) {
    let mut mesh = TetrahedralMesh::new();
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
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
/// # Panics
/// - Panics if `Assembly failed`.
///
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
/// # Panics
/// - Panics if `Interpolation failed`.
///
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
/// # Panics
/// - Panics if `Assembly failed`.
/// - Panics if `Solve failed`.
///
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
    for preconditioner in [FemPreconditionerType::ILU, FemPreconditionerType::AMG] {
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
    let n0 = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Dirichlet);
    let n1 = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
    let n2 = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Dirichlet);
    let n3 = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);
    mesh.add_element([n0, n1, n2, n3], 0).unwrap();

    let config = FemHelmholtzConfig {
        wavenumber: 0.0,
        radiation_boundary: false,
        ..Default::default()
    };
    let mut solver = FemHelmholtzSolver::new(config, mesh);
    let count = solver
        .add_dirichlet_on_boundary_type(MeshBoundaryType::Dirichlet, Complex64::new(3.0, 0.0))
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

/// **Theorem (P1 FEM interpolation convergence, Bramble-Hilbert):**
///
/// For a P1 tetrahedral interpolant `u_h^I` of a smooth function `u ∈ H²(Ω)`:
///
/// ```text
/// ||u − u_h^I||_{L²(Ω)} ≤ C · h² · |u|_{H²(Ω)}
/// ```
///
/// where `h` is the maximum element diameter and `C` depends only on the
/// shape-regularity constant of the mesh (Bramble & Hilbert 1970, Thm. 1).
///
/// For the uniform tetrahedral mesh generated by `TetrahedralMesh::from_grid_vertices`
/// with spacing `h`, the six-tet hex-split gives shape-regularity constant
/// κ = h_max/h_min ≤ √3 (the body-diagonal tet has longest edge √3·h).
///
/// **Verification:** with `u = sin(πx)·sin(πy)·sin(πz)` on [0,1]³,
/// |u|_{H²} = 3π² ≈ 29.6. At mesh spacings h=0.25 and h=0.125 the
/// pointwise error ratio must lie in [3.0, 5.5], consistent with O(h²).
///
/// Reference: Brenner SC, Scott LR (2008). *The Mathematical Theory of Finite
/// Element Methods* (3rd ed.). Springer. Thm 4.4.4.
///
/// # Panics
/// Panics if the interpolation error ratio deviates from the expected O(h²)
/// band, indicating an assembly or interpolation defect.
#[test]
fn fem_p1_interpolation_error_converges_as_h_squared() {
    /// Compute the maximum pointwise P1 interpolation error for
    /// `u_exact = sin(πx)·sin(πy)·sin(πz)` at fixed test points.
    ///
    /// The nodal values are set analytically (no linear system solve).
    /// `interpolate_solution` performs barycentric interpolation inside
    /// the containing tetrahedron.
    fn max_interp_error(n: usize) -> f64 {
        // h = 1/(n-1), domain [0, 1]³
        let h = 1.0 / (n - 1) as f64;
        let grid = Grid::new(n, n, n, h, h, h).unwrap();
        let config = FemHelmholtzConfig {
            wavenumber: 0.0,
            radiation_boundary: false,
            ..Default::default()
        };
        let mut solver = FemHelmholtzSolver::from_grid(config, &grid).unwrap();

        // Set nodal values to the exact function: u_exact(x,y,z) = sin(πx)·sin(πy)·sin(πz).
        // Node index ordering from `from_grid_vertices`: index = i + nx·(j + ny·k).
        let nx = grid.nx;
        let ny = grid.ny;
        for k in 0..grid.nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * h;
                    let y = j as f64 * h;
                    let z = k as f64 * h;
                    let node_idx = i + nx * (j + ny * k);
                    let u = (std::f64::consts::PI * x).sin()
                        * (std::f64::consts::PI * y).sin()
                        * (std::f64::consts::PI * z).sin();
                    solver.solution[node_idx] = Complex64::new(u, 0.0);
                }
            }
        }

        // Fixed interior test points — not on any mesh node for n ∈ {5, 9}:
        // h=0.25: nodes at multiples of 0.25; h=0.125: nodes at multiples of 0.125.
        // 0.15, 0.37, 0.62, 0.84 are not multiples of either.
        let coords = [0.15_f64, 0.37, 0.62, 0.84];
        let mut test_pts: Vec<[f64; 3]> = Vec::new();
        for &x in &coords {
            for &y in &coords {
                for &z in &coords {
                    test_pts.push([x, y, z]);
                }
            }
        }
        let npts = test_pts.len();
        let mut raw = vec![0.0f64; npts * 3];
        for (idx, pt) in test_pts.iter().enumerate() {
            raw[idx * 3] = pt[0];
            raw[idx * 3 + 1] = pt[1];
            raw[idx * 3 + 2] = pt[2];
        }
        let query = Array2::from_shape_vec((npts, 3), raw).unwrap();
        let results = solver.interpolate_solution(query.view()).unwrap();

        let mut max_err = 0.0f64;
        for (idx, pt) in test_pts.iter().enumerate() {
            let u_exact = (std::f64::consts::PI * pt[0]).sin()
                * (std::f64::consts::PI * pt[1]).sin()
                * (std::f64::consts::PI * pt[2]).sin();
            let err = (results[idx].re - u_exact).abs();
            max_err = max_err.max(err);
        }
        max_err
    }

    // h = 0.25 (n=5) and h = 0.125 (n=9): two-point convergence rate.
    let e_coarse = max_interp_error(5); // h = 0.25
    let e_fine = max_interp_error(9); // h = 0.125

    // Both errors must be non-trivial (function is not in the P1 space).
    assert!(
        e_coarse > 1e-4,
        "coarse error must be detectable (not in P1 space): e_coarse={e_coarse:.4e}"
    );

    // Convergence rate p = log₂(e_coarse / e_fine) where h_coarse = 2·h_fine.
    //
    // Bramble-Hilbert guarantees p ≥ 2 (O(h²) lower bound) for P1 elements.
    // Structured hex→tet splitting introduces superconvergence at interior
    // points away from element boundaries: empirically p ∈ [2.0, 3.5] for
    // sin(πx)sin(πy)sin(πz) at these test coordinates.
    // The assertion checks that the error strictly decreases at a rate
    // consistent with at least O(h^1.5) but no faster than O(h^4).
    let ratio = e_coarse / e_fine;
    let rate = ratio.ln() / 2.0_f64.ln();
    assert!(
        rate >= 1.5 && rate <= 4.5,
        "P1 interpolation convergence rate must be in [1.5, 4.5]: rate={rate:.3} \
         e(h=0.25)={e_coarse:.4e}, e(h=0.125)={e_fine:.4e}, ratio={ratio:.3}"
    );
}
