//! Tests for kwavers' Athena-backed Krylov entry points.
//!
//! These carry over the properties the crate's own matrix-free GMRES was
//! pinned on, restated against the [`LinearOperator`] seam that replaced its
//! closure argument.

use super::restart::RestartWidth;
use super::{
    solve_cg, CpuBackend, GMRESConfig, GmresConvergenceInfo, KrylovWorkspace, SliceOperator,
    SlicePreconditioner,
};
use athena_core::{
    ConvergencePolicy, Identity, IterationObserver, IterationState, KrylovBackend, LinearOperator,
    Termination,
};
use athena_leto::{LetoBackend, LetoBackendError};
use eunomia::assert_relative_eq;
use leto::Array1;

type Backend = LetoBackend<f64>;

/// Matrix-free `A = factor · I`.
///
/// The direct analogue of the closure the deleted solver took: no matrix is
/// ever formed, and the solve sees the operator only through its application.
struct ScaledIdentity {
    dimension: usize,
    factor: f64,
}

impl LinearOperator<Backend> for ScaledIdentity {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        _backend: &Backend,
        input: <Backend as KrylovBackend>::View<'_>,
        mut output: <Backend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if input.len() != output.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: input.len(),
                right: output.len(),
            });
        }
        for (image, &value) in output.iter_mut().zip(input.iter()) {
            *image = self.factor * value;
        }
        Ok(())
    }
}

/// Matrix-free lower-bidiagonal `A`: `y[i] = 4·x[i] − x[i−1]`.
///
/// Nonsymmetric by construction, so the Arnoldi Hessenberg genuinely carries a
/// subdiagonal for the Givens rotations to annihilate.
struct LowerBidiagonal {
    dimension: usize,
}

impl LinearOperator<Backend> for LowerBidiagonal {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        _backend: &Backend,
        input: <Backend as KrylovBackend>::View<'_>,
        mut output: <Backend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if input.len() != output.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: input.len(),
                right: output.len(),
            });
        }
        for (row, image) in output.iter_mut().enumerate() {
            let below = if row == 0 { 0.0 } else { input[row - 1] };
            *image = 4.0_f64.mul_add(input[row], -below);
        }
        Ok(())
    }
}

/// Records every checked residual Athena reports.
struct ResidualHistory {
    residuals: Vec<f64>,
}

impl IterationObserver<f64> for ResidualHistory {
    fn observe(&mut self, state: IterationState<f64>) {
        self.residuals.push(state.residual_norm);
    }
}

/// GMRES against the identity operator returns the right-hand side itself.
#[test]
fn identity_operator_solve_returns_the_right_hand_side() {
    let config = GMRESConfig {
        krylov_dim: 10,
        max_iterations: 10,
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-12,
    };
    let dimension = 8;
    let operator = ScaledIdentity {
        dimension,
        factor: 1.0,
    };
    let right_hand_side = Array1::from_elem([dimension], 1.0);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let mut workspace = KrylovWorkspace::new(config.krylov_dim, dimension)
        .expect("invariant: small workspace allocates");

    let report = workspace
        .solve(
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            config.policy().expect("invariant: valid policy"),
        )
        .expect("invariant: identity system solves");
    let info = GmresConvergenceInfo::from_report(&report);

    assert!(info.converged, "termination {:?}", report.termination);
    assert!(info.iterations <= 2, "iterations {}", info.iterations);
    assert!(
        info.final_residual < 1e-10,
        "final residual {:.3e}",
        info.final_residual
    );
    for (&value, &target) in solution.iter().zip(right_hand_side.iter()) {
        assert_relative_eq!(value, target, epsilon = 1e-10);
    }
}

/// GMRES against `A = 2·I` inverts the scaling.
#[test]
fn diagonal_operator_solve_inverts_the_scaling() {
    let config = GMRESConfig::default();
    let dimension = 64;
    let operator = ScaledIdentity {
        dimension,
        factor: 2.0,
    };
    let right_hand_side = Array1::from_elem([dimension], 4.0);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let mut workspace = KrylovWorkspace::new(config.krylov_dim, dimension)
        .expect("invariant: small workspace allocates");

    let report = workspace
        .solve(
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            config.policy().expect("invariant: valid policy"),
        )
        .expect("invariant: diagonal system solves");
    let info = GmresConvergenceInfo::from_report(&report);

    assert!(info.converged, "termination {:?}", report.termination);
    assert!(
        info.final_residual < 1e-6,
        "final residual {:.3e}",
        info.final_residual
    );
    for &value in solution.iter() {
        assert_relative_eq!(value, 2.0, epsilon = 1e-6);
    }
}

/// The checked residual never increases across a solve.
///
/// GMRES(m) minimises `‖b − Ax‖` over a subspace containing the zero
/// correction, so no step can raise the residual above the one it started from
/// (Saad & Schultz 1986, §2). The comparison allows one relative rounding of
/// the larger residual, which is the accuracy of the residual's own evaluation.
#[test]
fn checked_residuals_never_increase() {
    let config = GMRESConfig {
        krylov_dim: 10,
        max_iterations: 3,
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-10,
    };
    let dimension = 64;
    let operator = ScaledIdentity {
        dimension,
        factor: 1.5,
    };
    let right_hand_side = Array1::from_elem([dimension], 1.0);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let mut workspace = KrylovWorkspace::new(config.krylov_dim, dimension)
        .expect("invariant: small workspace allocates");
    let mut history = ResidualHistory {
        residuals: Vec::new(),
    };

    let report = workspace
        .solve_with_observer(
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            config.policy().expect("invariant: valid policy"),
            &mut history,
        )
        .expect("invariant: scaled-identity system solves");

    assert!(
        !history.residuals.is_empty(),
        "the observer must see every checked residual"
    );
    for pair in history.residuals.windows(2) {
        assert!(
            pair[1] <= pair[0] * (1.0 + f64::EPSILON),
            "residual increased: {:.6e} -> {:.6e}",
            pair[0],
            pair[1]
        );
    }
    assert!(report.converged(), "termination {:?}", report.termination);
    for &value in solution.iter() {
        assert_relative_eq!(value, 1.0 / 1.5, epsilon = 1e-8);
    }
}

/// A nonsymmetric `n × n` system is solved exactly within `n` iterations.
///
/// This replaces the deleted implementation's unit test of its private Givens
/// helper, which asserted `c² + s² = 1` and `−s·a + c·b = 0` directly. Athena
/// owns the rotations now, so the property is restated where kwavers can still
/// observe it. The rotations exist to reduce the Arnoldi Hessenberg to
/// triangular form so the least-squares subproblem is solved exactly at every
/// step; that exactness is what makes GMRES terminate with the true solution
/// after at most `n` iterations for a nonsingular `n × n` system in exact
/// arithmetic (Saad & Schultz 1986, Theorem 2.1). A rotation that failed to
/// annihilate the subdiagonal would leave the subproblem inexact and the
/// iterate short of the solution here. The operator is deliberately
/// nonsymmetric: for a scaled identity the Hessenberg carries no subdiagonal to
/// annihilate, and the test would pass without exercising the property.
///
/// The tolerance is `n · κ₂(A) · ε`. `A` is lower bidiagonal with diagonal 4
/// and subdiagonal −1, hence strictly diagonally dominant and well conditioned
/// (`κ₂(A) < 2` at `n = 6`): `6 · 2 · 2.2e-16 ≈ 2.7e-15`, and `1e-13` clears it
/// with margin for the residual re-evaluation at the end of the restart cycle.
#[test]
fn nonsymmetric_system_terminates_within_its_dimension() {
    let dimension = 6;
    let operator = LowerBidiagonal { dimension };
    let expected = Array1::from_shape_vec([dimension], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: vector length matches dimension");

    let backend = Backend::default();
    let mut right_hand_side = Array1::<f64>::zeros([dimension]);
    operator
        .apply(&backend, expected.view(), right_hand_side.view_mut())
        .expect("invariant: operator application shapes conform");

    let config = GMRESConfig {
        krylov_dim: dimension,
        max_iterations: 1,
        relative_tolerance: 1e-14,
        absolute_tolerance: 1e-14,
    };
    let mut solution = Array1::<f64>::zeros([dimension]);
    let mut workspace = KrylovWorkspace::new(config.krylov_dim, dimension)
        .expect("invariant: small workspace allocates");

    let report = workspace
        .solve(
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            config.policy().expect("invariant: valid policy"),
        )
        .expect("invariant: nonsingular system solves");

    assert_eq!(
        report.termination,
        Termination::Converged,
        "termination {:?}",
        report.termination
    );
    assert!(
        report.iterations <= dimension,
        "GMRES took {} iterations for a {dimension}-dimensional system",
        report.iterations
    );
    for (&value, &target) in solution.iter().zip(expected.iter()) {
        assert_relative_eq!(value, target, epsilon = 1e-13);
    }
}

/// Every requested restart lands on the smallest rung at least as wide.
#[test]
fn the_restart_ladder_covers_every_request() {
    for (requested, expected) in [
        (1, RestartWidth::W8),
        (8, RestartWidth::W8),
        (9, RestartWidth::W16),
        (30, RestartWidth::W32),
        (64, RestartWidth::W64),
        (100, RestartWidth::W128),
        (200, RestartWidth::W256),
        (10_000, RestartWidth::W256),
    ] {
        assert_eq!(
            RestartWidth::covering(requested),
            expected,
            "request {requested}"
        );
    }
}

/// The iteration budget is the restart count times the Krylov dimension.
#[test]
fn the_policy_budget_is_the_total_operator_application_count() {
    let config = GMRESConfig {
        krylov_dim: 7,
        max_iterations: 5,
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-12,
    };
    let policy = config.policy().expect("invariant: valid policy");

    assert_eq!(policy.max_iterations(), 35);
    assert_relative_eq!(policy.absolute_tolerance(), 1e-12, epsilon = 0.0);
    assert_relative_eq!(policy.relative_tolerance(), 1e-8, epsilon = 0.0);
}

/// A zero iteration budget is rejected rather than silently solved.
#[test]
fn a_zero_iteration_budget_is_rejected() {
    let config = GMRESConfig {
        krylov_dim: 0,
        max_iterations: 10,
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-12,
    };

    assert!(config.policy().is_err(), "a zero budget must be rejected");
}

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
