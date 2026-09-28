//! Tests for kwavers' conjugate-gradient entry point and its slice adapters.

use super::super::{solve_cg, CpuBackend, SliceOperator, SlicePreconditioner};
use athena_core::{ConvergencePolicy, Identity, LinearOperator, Termination};
use eunomia::assert_relative_eq;
use leto::Array1;

/// `A = tridiag(−1, 2, −1)`: symmetric positive definite, so CG converges.
///
/// The system is the one-dimensional second-difference stencil, whose exact
/// solution for a unit right-hand side is `x_i = (i+1)(n−i)/2` scaled by the
/// Dirichlet solve. CG reaches it in at most `n` iterations in exact
/// arithmetic; the tolerance is `n · κ₂(A) · ε`.
fn tridiagonal(dimension: usize) -> impl Fn(&[f64], &mut [f64]) {
    move |input: &[f64], output: &mut [f64]| {
        for (row, image) in output.iter_mut().enumerate() {
            let below = if row == 0 { 0.0 } else { input[row - 1] };
            let above = if row + 1 == dimension {
                0.0
            } else {
                input[row + 1]
            };
            *image = 2.0f64.mul_add(input[row], -(below + above));
        }
    }
}

/// The slice adapters drive Athena's CG to the true solution of an SPD system.
#[test]
fn slice_adapters_solve_a_symmetric_positive_definite_system() {
    let dimension = 32;
    let operator = SliceOperator::new(dimension, tridiagonal(dimension));
    let preconditioner = SlicePreconditioner::new(|residual: &[f64], output: &mut [f64]| {
        for (scaled, &value) in output.iter_mut().zip(residual.iter()) {
            *scaled = value / 2.0;
        }
    });
    let right_hand_side = Array1::from_elem([dimension], 1.0);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let policy = ConvergencePolicy::new(0.0, 1e-12, dimension).expect("invariant: valid CG policy");

    let report = solve_cg(
        &operator,
        &preconditioner,
        &right_hand_side,
        &mut solution,
        policy,
    )
    .expect("invariant: SPD system solves");

    assert!(
        report.converged(),
        "termination {:?} after {} iterations",
        report.termination,
        report.iterations
    );
    // Residual of the reported solution, formed independently of the solver.
    let residual = solution
        .iter()
        .enumerate()
        .map(|(row, &x)| {
            let below = if row == 0 { 0.0 } else { solution[row - 1] };
            let above = if row + 1 == dimension {
                0.0
            } else {
                solution[row + 1]
            };
            1.0 - (2.0f64.mul_add(x, -(below + above)))
        })
        .fold(0.0_f64, |acc, r| acc.max(r.abs()));
    assert!(
        residual <= 1e-8,
        "max-abs residual of the reported solution is {residual:.3e}"
    );
}

/// The identity system is solved in one step and reported as converged.
#[test]
fn cg_with_the_identity_operator_returns_the_right_hand_side() {
    let dimension = 16;
    let operator = SliceOperator::new(dimension, |input: &[f64], output: &mut [f64]| {
        output.copy_from_slice(input);
    });
    let right_hand_side = Array1::from_elem([dimension], 3.5);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let policy = ConvergencePolicy::new(0.0, 1e-12, 4).expect("invariant: valid CG policy");

    let report = solve_cg(
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        policy,
    )
    .expect("invariant: identity system solves");

    assert_eq!(report.termination, Termination::Converged);
    for &value in solution.iter() {
        assert_relative_eq!(value, 3.5, epsilon = 1e-12);
    }
}

/// Non-positive curvature truncates CG at the iterate so far, and at the first
/// iteration that iterate is the initial guess.
///
/// This is Steihaug truncation, and it is the property
/// [`crate::inverse::fwi::frequency_domain::gauss_newton`] relies on: an
/// indefinite `⟨p, Ap⟩` must leave the zeros initial guess in place so the
/// Levenberg-Marquardt outer loop raises its damping rather than taking an
/// unscaled steepest-descent step.
#[test]
fn cg_truncates_on_non_positive_curvature_leaving_the_initial_guess() {
    let dimension = 8;
    let operator = SliceOperator::new(dimension, |input: &[f64], output: &mut [f64]| {
        for (image, &value) in output.iter_mut().zip(input.iter()) {
            *image = -value;
        }
    });
    let right_hand_side = Array1::from_elem([dimension], 1.0);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let policy = ConvergencePolicy::new(0.0, 1e-6, 8).expect("invariant: valid CG policy");

    let report = solve_cg(
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        policy,
    )
    .expect("invariant: an indefinite system is reported, not rejected");

    assert_eq!(
        report.termination,
        Termination::NonPositiveCurvature,
        "an indefinite operator must truncate rather than keep descending"
    );
    assert_eq!(report.iterations, 0, "no step may be taken");
    for &value in solution.iter() {
        assert_relative_eq!(value, 0.0, epsilon = 0.0);
    }
}

/// The `CpuBackend` alias names the Leto backend the `Backend` tests use.
#[test]
fn the_cpu_backend_alias_is_the_leto_backend() {
    let operator = SliceOperator::new(1, |input: &[f64], output: &mut [f64]| {
        output[0] = input[0];
    });
    let backend = CpuBackend::default();
    assert_eq!(
        <SliceOperator<_> as LinearOperator<CpuBackend>>::dimension(&operator),
        1
    );
    let input = Array1::from_elem([1], 2.0);
    let mut output = Array1::<f64>::zeros([1]);
    LinearOperator::<CpuBackend>::apply(&operator, &backend, input.view(), output.view_mut())
        .expect("invariant: shapes conform");
    assert_relative_eq!(output[0], 2.0, epsilon = 0.0);
}
