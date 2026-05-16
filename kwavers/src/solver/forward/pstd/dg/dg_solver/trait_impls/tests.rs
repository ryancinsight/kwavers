use super::super::super::basis::BasisType;
use super::super::super::config::DGConfig;
use super::super::super::quadrature::fourier_periodic_nodes;
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

// ─── Fourier DG constructor tests ────────────────────────────────────────────

/// Build a minimal test grid.
fn make_grid(n: usize) -> Arc<Grid> {
    Arc::new(Grid::new(n, n, n, 1.0, 1.0, 1.0).expect("Grid::new failed in test"))
}

/// **Equispaced nodes**: `fourier_periodic_nodes(N)` must produce
/// `x_j = -1 + 2j/N` and `w_j = 2/N` for all `j < N`.
#[test]
fn fourier_periodic_nodes_generates_equispaced_nodes_and_uniform_weights() {
    let n = 8_usize;
    let (nodes, weights) = fourier_periodic_nodes(n).expect("fourier_periodic_nodes failed");

    assert_eq!(nodes.len(), n);
    assert_eq!(weights.len(), n);

    let expected_weight = 2.0 / n as f64;
    for j in 0..n {
        let expected_node = -1.0 + 2.0 * j as f64 / n as f64;
        assert!(
            (nodes[j] - expected_node).abs() < 1e-15,
            "node[{j}] = {}, expected {expected_node}",
            nodes[j]
        );
        assert!(
            (weights[j] - expected_weight).abs() < 1e-15,
            "weight[{j}] = {}, expected {expected_weight}",
            weights[j]
        );
    }
}

/// **Minimum-N rejection**: `fourier_periodic_nodes(1)` must return `Config` error.
#[test]
fn fourier_periodic_nodes_rejects_n_less_than_2() {
    let err = fourier_periodic_nodes(1).unwrap_err();
    assert!(format!("{err}").contains(">= 2") || format!("{err}").contains("fourier_nodes"));
}

/// **DGSolver::new_fourier constructs successfully**: basis type is Fourier,
/// `n_nodes = poly_order + 1`, and lift matrix is all-zero (periodic element).
#[test]
fn new_fourier_constructs_with_correct_metadata_and_zero_lift() {
    let poly_order = 7; // N = 8 equispaced nodes
    let config = DGConfig {
        polynomial_order: poly_order,
        ..DGConfig::default()
    };
    let grid = make_grid(poly_order + 1);

    let solver = DGSolver::new_fourier(config, grid).expect("new_fourier failed");

    assert_eq!(solver.config.basis_type, BasisType::Fourier);
    assert_eq!(solver.n_nodes, poly_order + 1);

    // Lift matrix must be all-zero (no net boundary flux in a periodic element).
    let lift = &*solver.lift_matrix;
    assert_eq!(lift.shape(), &[poly_order + 1, 2]);
    for &v in lift.iter() {
        assert_eq!(v, 0.0, "lift entry must be exactly 0.0");
    }
}

/// **Uniform mass matrix**: every diagonal entry of the mass matrix equals `2/N`.
#[test]
fn new_fourier_mass_matrix_has_uniform_diagonal_entries() {
    let n = 6_usize; // N = 6 equispaced nodes
    let config = DGConfig {
        polynomial_order: n - 1,
        ..DGConfig::default()
    };
    let grid = make_grid(n);
    let solver = DGSolver::new_fourier(config, grid).expect("new_fourier failed");

    let expected = 2.0 / n as f64;
    let m = &*solver.mass_matrix;
    for i in 0..n {
        assert!(
            (m[[i, i]] - expected).abs() < 1e-14,
            "M[{i},{i}] = {}, expected {expected}",
            m[[i, i]]
        );
        for j in 0..n {
            if i != j {
                assert!(
                    m[[i, j]].abs() < 1e-14,
                    "off-diagonal M[{i},{j}] = {} (must be 0)",
                    m[[i, j]]
                );
            }
        }
    }
}

/// **Spectral derivative exactness**: D · sin(π(x+1)) = π·cos(π(x+1)) at every node,
/// because `sin(π(x+1))` is exactly the first sine mode of the Fourier basis.
///
/// ## Proof
///
/// Let θ(x) = π(x+1). With `φ₁(x) = sin(θ)`, d/dx φ₁ = π cos(θ).
/// The N-node Fourier DG differentiates degree ≤ ⌊N/2⌋ trig polynomials exactly;
/// for N ≥ 4 the sine mode of wavenumber 1 is within range.
#[test]
fn new_fourier_differentiation_matrix_is_spectrally_exact_for_first_sine_mode() {
    let n = 8_usize;
    let config = DGConfig {
        polynomial_order: n - 1,
        ..DGConfig::default()
    };
    let grid = make_grid(n);
    let solver = DGSolver::new_fourier(config, grid).expect("new_fourier failed");

    let xi = &*solver.xi_nodes;
    let d = &*solver.diff_matrix;

    // f_j = sin(π(x_j + 1)),  df/dx = π cos(π(x_j + 1)).
    let f: Vec<f64> = (0..n)
        .map(|j| (std::f64::consts::PI * (xi[j] + 1.0)).sin())
        .collect();
    let df_exact: Vec<f64> = (0..n)
        .map(|j| std::f64::consts::PI * (std::f64::consts::PI * (xi[j] + 1.0)).cos())
        .collect();

    for i in 0..n {
        let df_computed: f64 = (0..n).map(|j| d[[i, j]] * f[j]).sum();
        let err = (df_computed - df_exact[i]).abs();
        assert!(
            err < 1e-11,
            "D·sin(π(x+1)) error at node {i}: computed={df_computed:.6e}, exact={:.6e}, err={err:.2e}",
            df_exact[i]
        );
    }
}

/// **GLL path still rejects Fourier**: `DGSolver::new` with `BasisType::Fourier` must
/// return an error because GLL nodes include both endpoints and violate periodicity.
#[test]
fn dg_solver_new_still_rejects_fourier_basis_type_via_gll_path() {
    let config = DGConfig {
        polynomial_order: 4,
        basis_type: BasisType::Fourier,
        ..DGConfig::default()
    };
    let grid = make_grid(5);
    let err = DGSolver::new(config, grid).unwrap_err();
    assert!(
        format!("{err}").contains("periodic nodes"),
        "expected 'periodic nodes' in error, got: {err}"
    );
}
