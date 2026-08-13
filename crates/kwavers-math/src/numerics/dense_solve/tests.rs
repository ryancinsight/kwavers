//! The two solves, against systems with known answers.

use super::*;

#[test]
fn square_solve_recovers_a_known_solution() {
    // [2 1; 1 3] x = [5; 10]  ->  x = [1; 3]
    let mut matrix = vec![2.0, 1.0, 1.0, 3.0];
    let mut rhs = vec![5.0, 10.0];
    solve_in_place(&mut matrix, &mut rhs, 2, "test system").expect("non-singular");
    assert!((rhs[0] - 1.0).abs() < 1e-12, "got {}", rhs[0]);
    assert!((rhs[1] - 3.0).abs() < 1e-12, "got {}", rhs[1]);
}

#[test]
fn square_solve_rejects_a_singular_system() {
    let mut matrix = vec![1.0, 2.0, 2.0, 4.0];
    let mut rhs = vec![1.0, 2.0];
    assert!(solve_in_place(&mut matrix, &mut rhs, 2, "test system").is_err());
}

/// An over-determined but *consistent* system — the shape the
/// summation-by-parts conditions take — is solved exactly, not approximately.
#[test]
fn least_squares_solves_a_consistent_overdetermined_system() {
    // Three equations, two unknowns, all satisfied by x = [2, -1].
    let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let b = vec![2.0, -1.0, 1.0];
    let x = solve_least_squares(&a, &b, 3, 2, 1e-12, "test system").expect("solvable");
    assert!((x[0] - 2.0).abs() < 1e-8, "got {x:?}");
    assert!((x[1] + 1.0).abs() < 1e-8, "got {x:?}");
    assert!(residual_norm(&a, &b, &x, 3, 2) < 1e-8);
}

/// An inconsistent system returns *something*, and the residual is what reveals
/// it. This is why the derivation checks the residual instead of trusting the
/// solve — a plausible vector satisfying none of the conditions is exactly the
/// failure mode a coefficient derivation must not ship.
#[test]
fn least_squares_residual_exposes_an_inconsistent_system() {
    let a = vec![1.0, 1.0, 1.0];
    let b = vec![1.0, 2.0, 3.0];
    let x = solve_least_squares(&a, &b, 3, 1, 1e-12, "test system").expect("solvable");
    assert!(
        residual_norm(&a, &b, &x, 3, 1) > 0.5,
        "an inconsistent system must not look solved"
    );
}

#[test]
fn least_squares_rejects_mismatched_shapes() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    assert!(solve_least_squares(&a, &b, 3, 2, 1e-12, "test system").is_err());
}
