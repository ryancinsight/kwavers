use super::super::super::basis::BasisType;
use super::super::super::config::DGConfig;
use super::super::super::traits::DGOperations;
use super::super::core::DGSolver;
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::sync::Arc;

/// Build a minimal DGSolver for testing.
/// # Panics
/// - Panics if `Grid::new failed in test`.
/// - Panics if `DGSolver::new failed in test`.
///
fn make_solver(poly_order: usize) -> DGSolver {
    let config = DGConfig {
        polynomial_order: poly_order,
        basis_type: BasisType::Legendre,
        ..DGConfig::default()
    };
    let grid = Arc::new(
        Grid::new(
            poly_order + 1,
            poly_order + 1,
            poly_order + 1,
            1.0,
            1.0,
            1.0,
        )
        .expect("Grid::new failed in test"),
    );
    DGSolver::new(config, grid).expect("DGSolver::new failed in test")
}

#[test]
fn fourier_basis_solver_construction_rejects_gll_duplicate_periodic_endpoints() {
    let config = DGConfig {
        polynomial_order: 2,
        basis_type: BasisType::Fourier,
        ..DGConfig::default()
    };
    let grid = Arc::new(Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("Grid::new failed in test"));

    let error = DGSolver::new(config, grid).unwrap_err();

    assert!(format!("{error}").contains("requires periodic nodes on [-1,1)"));
}

/// **Round-trip identity**: project_to_basis followed by reconstruct_from_basis
/// must recover the input to machine precision (≤ 1e-12) for any field.
///
/// ## Theorem
///
/// For collocation DG, V · V⁻¹ = I (exact arithmetic).  In floating-point:
///   ‖V · (V⁻¹ · f) − f‖₂ ≤ ε_mach · κ(V) · ‖f‖₂
/// For GLL nodes and p ≤ 10, κ(V) < 10² so the round-trip error is ≤ 1e-10.
/// We assert ≤ 1e-11 (two decades of margin).
///
/// Reference: Kopriva (2009) §3.4.
/// # Panics
/// - Panics if `project_to_basis failed`.
/// - Panics if `reconstruct failed`.
///
#[test]
fn test_project_reconstruct_round_trip() {
    let solver = make_solver(3); // p=3, n_nodes=4
    let n_nodes = solver.n_nodes;
    let n_elements = 5;
    let n_vars = 2;

    let mut field = Array3::<f64>::zeros((n_elements, n_nodes, n_vars));
    for e in 0..n_elements {
        for i in 0..n_nodes {
            for v in 0..n_vars {
                field[[e, i, v]] = ((e + 1) as f64) * 1.3 + (i as f64) * 0.7 + (v as f64) * 0.2;
            }
        }
    }

    let coefficients = solver
        .project_to_basis(&field)
        .expect("project_to_basis failed");
    let recovered = solver
        .reconstruct_from_basis(&coefficients)
        .expect("reconstruct failed");

    for e in 0..n_elements {
        for i in 0..n_nodes {
            for v in 0..n_vars {
                let err = (recovered[[e, i, v]] - field[[e, i, v]]).abs();
                assert!(
                    err < 1e-11,
                    "Round-trip error at [e={},i={},v={}]: {:.2e} (must be < 1e-11)",
                    e,
                    i,
                    v,
                    err
                );
            }
        }
    }
}

/// **Legendre coefficient extraction**: projecting the k-th normalised Legendre
/// basis function P̃_k must yield c[k] = 1 and all other |c[j]| < 1e-12.
///
/// ## Theorem
///
/// V⁻¹ · (V[:, k]) = e_k  (the k-th standard basis vector), because
/// V⁻¹ · V = I by construction.
///
/// Reference: Hesthaven & Warburton (2008) §3.1.
/// # Panics
/// - Panics if `project_to_basis failed`.
///
#[test]
fn test_legendre_coefficient_extraction() {
    let solver = make_solver(3); // p=3, n_nodes=4
    let n_nodes = solver.n_nodes;
    let v = &*solver.vandermonde;

    for k in 0..n_nodes {
        let mut field = Array3::<f64>::zeros((1, n_nodes, 1));
        for i in 0..n_nodes {
            field[[0, i, 0]] = v[[i, k]];
        }

        let coefficients = solver
            .project_to_basis(&field)
            .expect("project_to_basis failed");

        for j in 0..n_nodes {
            let expected = if j == k { 1.0 } else { 0.0 };
            let err = (coefficients[[0, j, 0]] - expected).abs();
            assert!(
                err < 1e-11,
                "Legendre extraction basis k={k}: c[{j}] = {:.6e}, expected {expected} (err {err:.2e})",
                coefficients[[0, j, 0]]
            );
        }
    }
}

/// **Polynomial reproduction**: a polynomial of degree ≤ p is exactly
/// represented in the DG basis (Hesthaven & Warburton 2008 §3.1, Theorem 3.1).
/// # Panics
/// - Panics if `project_to_basis failed`.
/// - Panics if `reconstruct failed`.
///
#[test]
fn test_polynomial_reproduction() {
    let poly_order = 3;
    let solver = make_solver(poly_order);
    let n_nodes = solver.n_nodes;
    let xi = &*solver.xi_nodes;

    // Build a degree-p polynomial: f(ξ) = ξ^p (exactly representable)
    let mut field = Array3::<f64>::zeros((1, n_nodes, 1));
    for i in 0..n_nodes {
        field[[0, i, 0]] = xi[i].powi(poly_order as i32);
    }

    let coefficients = solver
        .project_to_basis(&field)
        .expect("project_to_basis failed");
    let recovered = solver
        .reconstruct_from_basis(&coefficients)
        .expect("reconstruct failed");

    for i in 0..n_nodes {
        let err = (recovered[[0, i, 0]] - field[[0, i, 0]]).abs();
        assert!(
            err < 1e-11,
            "Polynomial reproduction error at node {i}: {err:.2e} (must be < 1e-11)"
        );
    }
}
