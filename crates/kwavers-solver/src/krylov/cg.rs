//! Athena's preconditioned conjugate gradient, in kwavers' vocabulary.
//!
//! The recurrence — `r → z = M⁻¹r → p → α = ⟨r,z⟩/⟨p,Ap⟩ → x += αp →
//! r -= αAp → β = ⟨r,z⟩_new/⟨r,z⟩_old` — is Athena's (Atlas ADR 0033). This
//! module holds no recurrence. It carries only what is genuinely kwavers':
//!
//! - [`solve_cg`], the entry point that binds an operator and a preconditioner
//!   to the Leto CPU backend and converts Athena's failure vocabulary into
//!   kwavers' typed numerical errors;
//! - [`SliceOperator`] and [`SlicePreconditioner`], the adapters that let a
//!   call site keep speaking in plain `&[f64]` slices while Athena sees its
//!   [`LinearOperator`] and [`Preconditioner`] seams.
//!
//! A site whose recurrence is *not* textbook preconditioned CG — a line search
//! on the step, a per-iteration objective history, a stopping norm that is not
//! `max(absolute, relative · ‖b‖₂)` — must keep its own loop or be shown to map
//! exactly. [`crate::inverse::fwi::frequency_domain::gauss_newton`] is the
//! shape that maps: Steihaug truncation is exactly Athena's non-positive-
//! curvature termination, and its `‖r‖² ≤ 1e-12‖r₀‖²` rule is a relative
//! tolerance of `1e-6`.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, KrylovBackend, LinearOperator, Preconditioner, SolveReport,
};
use athena_leto::{LetoBackend, LetoBackendError};
use kwavers_core::error::KwaversResult;
use leto::Array1;

use super::restart::{backend_failure, solve_failure};

/// The CPU backend every kwavers Krylov solve runs on.
pub type CpuBackend = LetoBackend<f64>;

/// Solve `A·x = b` with Athena's preconditioned conjugate gradient.
///
/// `solution` carries the initial iterate in and the final iterate out; every
/// kwavers call site starts from the zero iterate. A solve that stalls,
/// stagnates, or hits non-positive curvature is reported value-semantically in
/// the returned [`SolveReport`] rather than as an error: the last iterate is
/// still present, and whether it is usable is the caller's judgement. Only a
/// dimension mismatch or a backend failure is an error.
///
/// # Errors
///
/// Returns [`NumericalError::MatrixDimension`](kwavers_core::error::NumericalError::MatrixDimension)
/// when the operator, the vectors, and the workspace disagree on the system
/// dimension, and
/// [`NumericalError::SolverFailed`](kwavers_core::error::NumericalError::SolverFailed)
/// when the Krylov workspace cannot be allocated or the backend fails.
pub fn solve_cg<O, P>(
    operator: &O,
    preconditioner: &P,
    right_hand_side: &Array1<f64>,
    solution: &mut Array1<f64>,
    policy: ConvergencePolicy<f64>,
) -> KwaversResult<SolveReport<f64>>
where
    O: LinearOperator<CpuBackend>,
    P: Preconditioner<CpuBackend>,
{
    let backend = CpuBackend::default();
    let mut workspace = CgWorkspace::new(&backend, operator.dimension())
        .map_err(|error| backend_failure("CG workspace allocation", &error))?;
    Cg::<CpuBackend>::solve_into(
        &backend,
        operator,
        preconditioner,
        right_hand_side,
        solution,
        &mut workspace,
        policy,
    )
    .map_err(|error| solve_failure(&error))
}

/// A matrix-free square operator defined by a slice-to-slice application.
///
/// The body writes `output = A · input`; it must fill every element of
/// `output`, which Athena hands over holding the previous image rather than
/// zeros.
pub struct SliceOperator<F> {
    dimension: usize,
    apply: F,
}

impl<F> SliceOperator<F>
where
    F: Fn(&[f64], &mut [f64]),
{
    /// Borrow `dimension`-square operator application.
    pub const fn new(dimension: usize, apply: F) -> Self {
        Self { dimension, apply }
    }
}

impl<F> LinearOperator<CpuBackend> for SliceOperator<F>
where
    F: Fn(&[f64], &mut [f64]),
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        _backend: &CpuBackend,
        input: <CpuBackend as KrylovBackend>::View<'_>,
        mut output: <CpuBackend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        (self.apply)(input, output);
        Ok(())
    }
}

/// A preconditioner defined by a slice-to-slice application.
///
/// The body writes `output = M⁻¹·residual`; unlike [`SliceOperator`], Athena
/// hands the output over holding the previous image, and the body is free to
/// leave the residual unscaled on rows whose diagonal has no usable
/// reciprocal.
pub struct SlicePreconditioner<F> {
    apply: F,
}

impl<F> SlicePreconditioner<F>
where
    F: Fn(&[f64], &mut [f64]),
{
    /// Borrow the preconditioner's application.
    pub const fn new(apply: F) -> Self {
        Self { apply }
    }
}

impl<F> Preconditioner<CpuBackend> for SlicePreconditioner<F>
where
    F: Fn(&[f64], &mut [f64]),
{
    fn apply(
        &self,
        _backend: &CpuBackend,
        residual: <CpuBackend as KrylovBackend>::View<'_>,
        mut output: <CpuBackend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let residual = residual
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        (self.apply)(residual, output);
        Ok(())
    }
}
