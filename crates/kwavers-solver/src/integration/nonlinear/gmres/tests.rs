//! Tests for the GMRES solver.

use super::config::GMRESConfig;
use super::solver::GMRESSolver;
use eunomia::assert_relative_eq;
use kwavers_core::error::KwaversResult;
use leto::Array3;

#[test]
fn test_gmres_identity_matrix() {
    let config = GMRESConfig {
        krylov_dim: 10,
        max_iterations: 10,
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-12,
        use_preconditioner: false,
    };

    let mut solver = GMRESSolver::new(config);

    let b = Array3::from_elem([2, 2, 2], 1.0);
    let mut x0 = Array3::zeros((2, 2, 2));

    let matvec = |v: &Array3<f64>| Ok(v.clone());

    let result = solver.solve(matvec, &b, &mut x0);

    match &result {
        Ok(info) => {
            println!(
                "Converged: {}, iterations: {}, residual: {}",
                info.converged, info.iterations, info.final_residual
            );
        }
        Err(e) => {
            println!("Error: {:?}", e);
            println!("Residual history len: {}", solver.residual_history().len());
            println!("Iteration count: {}", solver.iteration_count());
        }
    }

    let info = result.unwrap();

    assert!(info.converged);
    assert!(info.iterations <= 2);
    assert!(info.final_residual < 1e-10);

    for (&x_val, &b_val) in x0.iter().zip(b.iter()) {
        assert_relative_eq!(x_val, b_val, epsilon = 1e-10);
    }
}

#[test]
fn test_gmres_diagonal_matrix() {
    let config = GMRESConfig::default();
    let mut solver = GMRESSolver::new(config);

    let b = Array3::from_elem([4, 4, 4], 4.0);
    let mut x0 = Array3::zeros((4, 4, 4));

    let matvec = |v: &Array3<f64>| Ok(v * 2.0);

    let info = solver.solve(matvec, &b, &mut x0).unwrap();

    assert!(info.converged);
    assert!(info.final_residual < 1e-6);

    for &x_val in x0.iter() {
        assert_relative_eq!(x_val, 2.0, epsilon = 1e-6);
    }
}

#[test]
fn test_gmres_residual_decrease() {
    let config = GMRESConfig {
        krylov_dim: 10,
        max_iterations: 3,
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-10,
        use_preconditioner: false,
    };

    let mut solver = GMRESSolver::new(config);

    let b = Array3::from_elem([4, 4, 4], 1.0);
    let mut x0 = Array3::zeros((4, 4, 4));

    let matvec = |v: &Array3<f64>| Ok(v * 1.5);

    let _info = solver.solve(matvec, &b, &mut x0).unwrap();

    let history = solver.residual_history();
    for i in 1..(history.len()) {
        assert!(
            history[i] <= history[i - 1] * (1.0 + 1e-10),
            "Residual increased: {} -> {}",
            history[i - 1],
            history[i]
        );
    }
}

#[test]
fn test_givens_rotation() {
    let (c, s) = GMRESSolver::givens_rotation(3.0, 4.0);

    assert_relative_eq!(c * c + s * s, 1.0, epsilon = 1e-14);

    let eliminated = -s * 3.0 + c * 4.0;
    assert!(eliminated.abs() < 1e-14);
}

/// Diagonal operator over the flat element order, cycling `diagonal`.
fn diagonal_matvec(diagonal: &[f64]) -> impl Fn(&Array3<f64>) -> KwaversResult<Array3<f64>> + '_ {
    move |v: &Array3<f64>| {
        let mut out = v.clone();
        for (index, value) in out.iter_mut().enumerate() {
            *value *= diagonal[index % diagonal.len()];
        }
        Ok(out)
    }
}

#[test]
fn distinct_eigenvalue_count_bounds_the_iteration_count() {
    // GMRES terminates in at most d iterations for a diagonalisable operator
    // with d distinct eigenvalues (minimal-polynomial argument), so a two-value
    // diagonal must close the Krylov subspace by the second step.
    let mut solver = GMRESSolver::new(GMRESConfig {
        krylov_dim: 20,
        max_iterations: 10,
        relative_tolerance: 1e-12,
        absolute_tolerance: 1e-14,
        use_preconditioner: false,
    });
    let b = Array3::from_elem([4, 4, 4], 1.0);
    let mut x0 = Array3::zeros((4, 4, 4));

    let info = solver
        .solve(diagonal_matvec(&[2.0, 5.0]), &b, &mut x0)
        .unwrap();

    assert!(info.converged);
    assert!(
        info.iterations <= 2,
        "expected at most 2 iterations, took {}",
        info.iterations
    );
    for (index, &value) in x0.iter().enumerate() {
        let expected = if index % 2 == 0 { 0.5 } else { 0.2 };
        assert_relative_eq!(value, expected, epsilon = 1e-10);
    }
}

#[test]
fn non_finite_operator_output_is_reported() {
    // Regression: without a finiteness guard the recurrence propagated NaN,
    // never met the tolerance, and burned the whole iteration budget before
    // reporting a generic failure.
    let mut solver = GMRESSolver::new(GMRESConfig::default());
    let b = Array3::from_elem([2, 2, 2], 1.0);
    let mut x0 = Array3::zeros((2, 2, 2));

    let result = solver.solve(
        |v: &Array3<f64>| {
            let mut out = v.clone();
            for value in out.iter_mut() {
                *value = f64::NAN;
            }
            Ok(out)
        },
        &b,
        &mut x0,
    );

    assert!(
        result.is_err(),
        "a non-finite operator image must be reported"
    );
    assert!(
        solver.iteration_count() <= 1,
        "the guard must fire on the first step, not after the budget"
    );
}

#[test]
fn singular_operator_does_not_report_a_false_success() {
    // A ≡ 0 drives the least-squares estimate to zero on the first step while
    // ‖b − A·x‖ is untouched. Reporting convergence from the estimate alone
    // would return an unsolved system.
    let mut solver = GMRESSolver::new(GMRESConfig::default());
    let b = Array3::from_elem([2, 2, 2], 1.0);
    let mut x0 = Array3::zeros((2, 2, 2));

    let result = solver.solve(|v: &Array3<f64>| Ok(Array3::zeros(v.shape())), &b, &mut x0);

    assert!(
        result.is_err(),
        "a singular operator must not report success"
    );
}

#[test]
fn repeated_solves_reuse_the_workspace_and_agree() {
    // The workspace is retained across solves; a second solve must neither see
    // stale rotation state nor a stale basis.
    let mut solver = GMRESSolver::new(GMRESConfig::default());
    let b = Array3::from_elem([3, 3, 3], 2.0);

    let mut first = Array3::zeros((3, 3, 3));
    let info_first = solver
        .solve(diagonal_matvec(&[1.5, 4.0, 0.75]), &b, &mut first)
        .unwrap();

    let mut second = Array3::zeros((3, 3, 3));
    let info_second = solver
        .solve(diagonal_matvec(&[1.5, 4.0, 0.75]), &b, &mut second)
        .unwrap();

    assert_eq!(info_first.iterations, info_second.iterations);
    for (&a, &c) in first.iter().zip(second.iter()) {
        assert_relative_eq!(a, c, epsilon = 1e-14);
    }
    for (index, &value) in first.iter().enumerate() {
        let expected = 2.0 / [1.5, 4.0, 0.75][index % 3];
        assert_relative_eq!(value, expected, epsilon = 1e-8);
    }
}

#[test]
fn restarts_converge_on_an_ill_conditioned_diagonal() {
    // krylov_dim below the number of distinct eigenvalues forces several outer
    // cycles, exercising the restart path and the retained workspace. The
    // diagonal is positive, so the operator is SPD and restarted GMRES(m)
    // converges for any m; the spread is kept moderate because GMRES(m) has no
    // convergence guarantee once the condition number outruns the budget.
    let mut solver = GMRESSolver::new(GMRESConfig {
        krylov_dim: 2,
        max_iterations: 200,
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-14,
        use_preconditioner: false,
    });
    let diagonal = [1.0, 2.0, 4.0, 8.0, 3.0];
    let b = Array3::from_elem([3, 3, 3], 1.0);
    let mut x0 = Array3::zeros((3, 3, 3));

    let info = solver
        .solve(diagonal_matvec(&diagonal), &b, &mut x0)
        .unwrap();

    assert!(info.converged);
    for (index, &value) in x0.iter().enumerate() {
        let expected = 1.0 / diagonal[index % diagonal.len()];
        assert_relative_eq!(value, expected, epsilon = 1e-8);
    }
}
