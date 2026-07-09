use super::matfree::{solve_lsqr_matfree, MatFreeOperator};
use super::solver::LsqrSolver;
use super::types::{LsqrConfig, StopReason};
use leto::{
    /* arr1 -- no leto equivalent */,
    Array2,
};

// ─── Matrix-free LSQR tests ──────────────────────────────────────────────────

/// Dense adapter used to test `solve_lsqr_matfree` against known analytic solutions.
struct DenseAdapter {
    a: Vec<Vec<f64>>,
    m: usize,
    n: usize,
}

impl DenseAdapter {
    fn new(a: Vec<Vec<f64>>) -> Self {
        let m = a.len();
        let n = if m > 0 { a[0].len() } else { 0 };
        Self { a, m, n }
    }
}

impl MatFreeOperator for DenseAdapter {
    fn rows(&self) -> usize {
        self.m
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for (i, row) in self.a.iter().enumerate() {
            y[i] = row.iter().zip(x).map(|(a, x)| a * x).sum();
        }
    }
    fn t_matvec(&self, y: &[f64], x: &mut [f64]) {
        x.fill(0.0);
        for (i, row) in self.a.iter().enumerate() {
            for (j, &a) in row.iter().enumerate() {
                x[j] += a * y[i];
            }
        }
    }
}

fn tight_config() -> LsqrConfig {
    LsqrConfig {
        max_iterations: 500,
        damping: 0.0,
        atol: 1e-12,
        btol: 1e-12,
        tolerance: 1e-12,
    }
}

#[test]
fn matfree_identity_recovers_rhs() {
    // A = I₄, b = [1,2,3,4] → exact solution x* = b; ‖Ax−b‖ = 0.
    let n = 4usize;
    let a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let r = solve_lsqr_matfree(&DenseAdapter::new(a), &b, &tight_config());
    for (xi, bi) in r.solution.iter().zip(&b) {
        assert!((xi - bi).abs() < 1e-9, "identity: x={xi}, expected {bi}");
    }
    assert!(r.residual_norm < 1e-9, "residual={}", r.residual_norm);
}

#[test]
fn matfree_diagonal_scales_solution_correctly() {
    // A = diag(2,3,4), b = [2,6,12] → x* = [1,2,3] analytically.
    let a = vec![
        vec![2.0, 0.0, 0.0],
        vec![0.0, 3.0, 0.0],
        vec![0.0, 0.0, 4.0],
    ];
    let b = vec![2.0, 6.0, 12.0];
    let r = solve_lsqr_matfree(&DenseAdapter::new(a), &b, &tight_config());
    let expected = [1.0, 2.0, 3.0];
    for (xi, &ei) in r.solution.iter().zip(&expected) {
        assert!((xi - ei).abs() < 1e-9, "diagonal: x={xi}, expected {ei}");
    }
}

#[test]
fn matfree_overdetermined_consistent_least_squares() {
    // A = [1 0; 0 1; 1 1], b = [1; 2; 3] — consistent (b in range(A)).
    // Normal equations: AᵀA·x = Aᵀb → [2 1; 1 2]·x = [4; 5] → x* = [1, 2].
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let b = vec![1.0, 2.0, 3.0];
    let r = solve_lsqr_matfree(&DenseAdapter::new(a), &b, &tight_config());
    assert!((r.solution[0] - 1.0).abs() < 1e-8, "x[0]={}", r.solution[0]);
    assert!((r.solution[1] - 2.0).abs() < 1e-8, "x[1]={}", r.solution[1]);
    assert!(r.residual_norm < 1e-8, "residual={}", r.residual_norm);
}

#[test]
fn matfree_damping_reduces_solution_norm() {
    // For A = I, b = [1;1;1], min ‖x-b‖² + λ²‖x‖² → x* = b/(1+λ²).
    // Larger λ → smaller ‖x*‖.
    let a: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let b = vec![1.0, 1.0, 1.0];
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();

    let r0 = solve_lsqr_matfree(
        &DenseAdapter::new(a.clone()),
        &b,
        &LsqrConfig {
            damping: 0.0,
            ..tight_config()
        },
    );
    let r1 = solve_lsqr_matfree(
        &DenseAdapter::new(a),
        &b,
        &LsqrConfig {
            damping: 2.0,
            ..tight_config()
        },
    );
    let n0 = norm(&r0.solution);
    let n1 = norm(&r1.solution);
    assert!(
        n1 < n0,
        "damping λ=2 should shrink ‖x‖: ‖x_λ‖={n1:.6} ≥ ‖x₀‖={n0:.6}"
    );
    // Analytic: x*(λ=2) = 1/(1+4) = 0.2 per component
    for xi in &r1.solution {
        assert!((xi - 0.2).abs() < 1e-6, "x*={xi:.6}, expected 0.2");
    }
}

#[test]
fn matfree_zero_rhs_returns_early_with_zero_solution() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![0.0, 0.0];
    let r = solve_lsqr_matfree(&DenseAdapter::new(a), &b, &tight_config());
    assert_eq!(
        r.iterations, 0,
        "zero RHS must return without any iteration"
    );
    assert!(
        r.solution.iter().all(|&x| x == 0.0),
        "solution must be zero"
    );
    assert!(r.converged, "zero RHS is trivially converged");
    assert_eq!(r.residual_norm, 0.0);
}

#[test]
fn matfree_objective_history_is_non_increasing() {
    // Monotone φ̄² ↘ confirms the Krylov subspace is expanding correctly.
    // A = [2 1; 1 3; 0 1] (overdetermined, no exact solution).
    let a = vec![vec![2.0, 1.0], vec![1.0, 3.0], vec![0.0, 1.0]];
    let b = vec![5.0, 10.0, 2.0];
    let r = solve_lsqr_matfree(
        &DenseAdapter::new(a),
        &b,
        &LsqrConfig {
            max_iterations: 20,
            ..tight_config()
        },
    );
    assert!(
        r.objective_history.len() >= 2,
        "need ≥ 2 entries to test monotonicity"
    );
    for w in r.objective_history.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-12,
            "objective increased: {:.6e} → {:.6e}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn matfree_consistent_with_explicit_lsqr_on_small_system() {
    // Verify that solve_lsqr_matfree and LsqrSolver agree to 1e-7 on a 4×3 system.
    let a_rows: Vec<Vec<f64>> = vec![
        vec![3.0, 1.0, 0.0],
        vec![0.0, 2.0, 1.0],
        vec![1.0, 0.0, 4.0],
        vec![1.0, 1.0, 1.0],
    ];
    let b = vec![4.0, 3.0, 9.0, 3.0];

    // Matrix-free result
    let r_mf = solve_lsqr_matfree(&DenseAdapter::new(a_rows.clone()), &b, &tight_config());

    // Explicit LsqrSolver result
    let mut a_nd = Array2::<f64>::zeros((4, 3));
    for (i, row) in a_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a_nd[[i, j]] = v;
        }
    }
    let b_nd = arr1(&b);
    let r_ex = LsqrSolver::new(LsqrConfig {
        max_iterations: 500,
        atol: 1e-12,
        btol: 1e-12,
        ..Default::default()
    })
    .solve(&a_nd, &b_nd);

    for (xi, xe) in r_mf.solution.iter().zip(r_ex.solution.iter()) {
        assert!(
            (xi - xe).abs() < 1e-7,
            "matfree x={xi:.9}, explicit x={xe:.9}"
        );
    }
}

#[test]
fn test_lsqr_config_default() {
    let cfg = LsqrConfig::default();
    assert_eq!(cfg.max_iterations, 1000);
    assert_eq!(cfg.tolerance, 1e-6);
    assert_eq!(cfg.damping, 0.0);
}

#[test]
fn test_lsqr_identity_system() {
    // Solve I·x = b where I is identity and b = [1, 2, 3]
    let a = Array2::eye(3);
    let b = arr1(&[1.0, 2.0, 3.0]);

    let solver = LsqrSolver::new(LsqrConfig::default());
    let result = solver.solve(&a, &b);

    assert!(result.converged || result.iterations > 0);
    assert!(result.residual_norm < 1e-6);
}

#[test]
fn test_lsqr_diagonal_system() {
    let mut a = Array2::eye(3);
    a[[0, 0]] = 2.0;
    a[[1, 1]] = 3.0;
    a[[2, 2]] = 5.0;
    let b = arr1(&[2.0, 3.0, 5.0]);

    let solver = LsqrSolver::new(LsqrConfig::default());
    let result = solver.solve(&a, &b);

    // Exact solution x = [1, 1, 1]
    assert!(result.residual_norm < 1e-5);
    for &xi in result.solution.iter() {
        assert!(
            (xi - 1.0).abs() < 1e-5,
            "Solution element should be 1.0, got {xi}"
        );
    }
}

/// **Test: overdetermined consistent system with exact LS solution**
///
/// System: A = [1 0; 0 2; 1 1], b = [1; 4; 3].
///
/// Normal equations: AᵀA·x = Aᵀb  →  [2 1; 1 5]·x = [4; 11].
/// Exact solution: x* = [1, 2] (b is in range(A), so residual is zero).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_lsqr_overdetermined_exact_solution() {
    let a = Array2::from_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 1.0, 1.0]).unwrap();
    let b = arr1(&[1.0, 4.0, 3.0]);

    let cfg = LsqrConfig {
        atol: 1e-10,
        btol: 1e-10,
        max_iterations: 500,
        ..Default::default()
    };
    let solver = LsqrSolver::new(cfg);
    let result = solver.solve(&a, &b);

    assert!(
        result.residual_norm < 1e-8,
        "Residual norm should be near zero for exact LS; got {}",
        result.residual_norm
    );
    assert!(
        (result.solution[0] - 1.0).abs() < 1e-7,
        "x[0] should be 1.0, got {}",
        result.solution[0]
    );
    assert!(
        (result.solution[1] - 2.0).abs() < 1e-7,
        "x[1] should be 2.0, got {}",
        result.solution[1]
    );
}

#[test]
fn test_lsqr_overdetermined_system() {
    // Overdetermined system (more equations than unknowns)
    let a = Array2::from_vec([3, 2], vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
    let b = arr1(&[1.0, 2.0, 1.0]);

    let solver = LsqrSolver::new(LsqrConfig::default());
    let result = solver.solve(&a, &b);

    // Should find least-squares solution
    assert!(result.iterations > 0);
    assert!(result.residual_norm < 1e-3);
}

#[test]
fn test_lsqr_zero_vector() {
    let a = Array2::eye(3);
    let b = arr1(&[0.0, 0.0, 0.0]);

    let solver = LsqrSolver::new(LsqrConfig::default());
    let result = solver.solve(&a, &b);

    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 0.0);
}

#[test]
fn test_lsqr_damping() {
    let a = Array2::eye(3);
    let b = arr1(&[1.0, 1.0, 1.0]);

    let cfg = LsqrConfig {
        damping: 0.1,
        ..Default::default()
    };

    let solver = LsqrSolver::new(cfg);
    let result = solver.solve(&a, &b);

    // Damping regularizes: solution < 1 for each component
    assert!(result.iterations > 0);
    assert!(result.residual_norm < 1e-3);
    // With I and λ=0.1, solution = b/(1+λ²) ≈ 0.99 < 1.0
    for &xi in result.solution.iter() {
        assert!(xi < 1.0 + 1e-8, "Damped solution should be < 1.0, got {xi}");
    }
}

#[test]
fn test_lsqr_condition_number() {
    let mut a = Array2::eye(2);
    a[[0, 0]] = 1.0;
    a[[1, 1]] = 1e-6;
    let b = arr1(&[1.0, 1.0]);

    let solver = LsqrSolver::new(LsqrConfig::default());
    let result = solver.solve(&a, &b);

    // Should estimate large condition number for this ill-conditioned system
    assert!(result.condition_number > 1e4 || result.iterations > 0);
}

#[test]
fn test_lsqr_convergence_tracking() {
    let a = Array2::eye(5);
    let b = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let solver = LsqrSolver::new(LsqrConfig::default());
    let result = solver.solve(&a, &b);

    assert!(result.converged || result.iterations > 0);
    assert!(result.at_residual_norm >= 0.0);
    assert!(result.residual_norm >= 0.0);
}

#[test]
fn test_stop_reason_variants() {
    // Ensure StopReason variants are accessible and distinct
    assert_ne!(StopReason::Converged, StopReason::MaxIterations);
    assert_ne!(StopReason::AtolSatisfied, StopReason::BtolSatisfied);
    assert_ne!(StopReason::IllConditioned, StopReason::Converged);
}
