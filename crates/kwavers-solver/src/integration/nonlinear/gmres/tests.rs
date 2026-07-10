//! Tests for the GMRES solver.

use super::config::GMRESConfig;
use super::solver::GMRESSolver;
use approx::assert_relative_eq;
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

    let b = Array3::from_elem((2, 2, 2), 1.0);
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

    let b = Array3::from_elem((4, 4, 4), 4.0);
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

    let b = Array3::from_elem((4, 4, 4), 1.0);
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
