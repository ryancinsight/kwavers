//! LSQR path for dense speed-of-sound shift reconstruction.
//!
//! Wraps [`SoundSpeedShiftOperator`] as a [`MatFreeOperator`] and drives
//! [`solve_lsqr_matfree`] with the Tikhonov damping extracted from
//! [`ShiftPrior::Lsqr`].

use kwavers_math::linear_algebra::iterative::lsqr::{
    matfree::{MatFreeOperator, MatFreeResult},
    solve_lsqr_matfree,
    types::LsqrConfig,
};

use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{ShiftPrior, SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};

/// Newtype so we can implement the foreign `MatFreeOperator` trait for the
/// crate-internal `SoundSpeedShiftOperator`.
struct ShiftOperatorAdapter<'a>(&'a SoundSpeedShiftOperator);

impl MatFreeOperator for ShiftOperatorAdapter<'_> {
    fn rows(&self) -> usize {
        self.0.rows()
    }
    fn cols(&self) -> usize {
        self.0.cols()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.0.matvec(x, y);
    }
    fn t_matvec(&self, y: &[f64], x: &mut [f64]) {
        self.0.t_matvec(y, x);
    }
}

/// Solve one frame using damped LSQR.
///
/// The `data` vector is `−c₀² · Δt` (RHS after the sign/scale convention of
/// the ray-integral forward model).  The damping λ is taken from
/// `config.prior` (must be [`ShiftPrior::Lsqr`]); panics in debug builds if
/// called with any other prior variant.
pub(super) fn solve_shift_lsqr(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    let damping = match config.prior {
        ShiftPrior::Lsqr { damping } => damping,
        _ => {
            debug_assert!(false, "solve_shift_lsqr called with non-Lsqr prior");
            0.0
        }
    };

    workspace.prepare(operator.rows(), operator.cols());

    // Scale tolerances relative to ‖b‖ so stopping criteria are data-adaptive
    // rather than absolute.  For a sound-speed RHS of order c₀²·Δt ~ O(1-10),
    // absolute 1e-8 would never trigger on under-determined geometries and would
    // trigger immediately on very small systems.
    let b_norm = data
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::EPSILON);
    let lsqr_config = LsqrConfig {
        max_iterations: config.iterations,
        damping,
        atol: 1e-8 * b_norm,
        btol: 1e-8 * b_norm,
        tolerance: 1e-6,
    };

    let adapter = ShiftOperatorAdapter(operator);
    let MatFreeResult {
        solution,
        objective_history,
        ..
    } = solve_lsqr_matfree(&adapter, data, &lsqr_config);

    workspace.solution.copy_from_slice(&solution);
    workspace.objective_history = objective_history;
}
