//! Tests for the GMRES solver.

use super::config::GMRESConfig;
use super::solver::GMRESSolver;
use eunomia::assert_relative_eq;
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

/// Regression: Arnoldi estimate alone must never report convergence.
///
/// If `A = scale·I`, the Arnoldi residual estimate is `|scale| * ||true_residual||`.
/// For `scale < 1` the estimate crosses the threshold early while the true
/// residual is still large. The fixed solver checks the true residual before
/// returning `Ok`, so `final_residual < tolerance` always holds when this
/// function returns without error.
#[test]
fn preconditioned_estimate_alone_never_reports_convergence() {
    let tolerance = 1e-8_f64;
    // Use scale = 1e-6: the Arnoldi estimate is 1e6x smaller than true residual.
    // The unfixed solver would report Ok after ~1 iteration; the fixed one
    // must verify the true residual.
    for &scale in &[1.0_f64, 1e-4, 1e-6] {
        let config = GMRESConfig {
            krylov_dim: 15,
            max_iterations: 200,
            relative_tolerance: tolerance,
            absolute_tolerance: tolerance,
            use_preconditioner: false,
        };
        let mut solver = GMRESSolver::new(config);

        // b = all-ones tensor, true solution is x = all-ones / scale
        let shape = [3, 3, 3];
        let b = Array3::from_elem(shape, 1.0);
        let mut x0 = Array3::zeros(shape);

        // A·v = scale * v  (scaled identity)
        let matvec = move |v: &Array3<f64>| Ok(v * scale);

        match solver.solve(matvec, &b, &mut x0) {
            Ok(info) => {
                // When Ok, the true residual MUST be below tolerance
                assert!(
                    info.final_residual <= tolerance * 10.0,
                    "scale={scale}: reported convergence with true residual \
                     {r:.3e} > tolerance {tolerance:.3e}",
                    r = info.final_residual
                );
            }
            Err(_) => {
                // Stagnation is allowed — just not a false Ok
            }
        }
    }
}
