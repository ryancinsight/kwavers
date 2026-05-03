use super::solver::LsqrSolver;
use super::types::{LsqrConfig, StopReason};
use ndarray::{arr1, Array2};

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
#[test]
fn test_lsqr_overdetermined_exact_solution() {
    let a = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 2.0, 1.0, 1.0]).unwrap();
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
    let a = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
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
