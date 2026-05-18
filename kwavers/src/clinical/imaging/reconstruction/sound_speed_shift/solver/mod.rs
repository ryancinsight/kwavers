//! Inverse solvers for speed-of-sound shift imaging.

mod dense;
mod linear_algebra;
mod lsqr;
mod metrics;
mod normal;
mod sparse;
mod workspace;

use super::operator::SoundSpeedShiftOperator;
use super::types::{ShiftPrior, SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};
pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) use metrics::{
    compute_solver_metrics, SoundSpeedShiftSolverMetrics,
};

pub(super) fn solve_shift(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    match config.prior {
        ShiftPrior::Dense => dense::solve_dense_pcg(operator, data, config, workspace),
        ShiftPrior::Sparse => sparse::solve_sparse_ista(operator, data, config, workspace),
        ShiftPrior::Lsqr { .. } => lsqr::solve_shift_lsqr(operator, data, config, workspace),
    }
}

pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) fn solve_shift_with_metrics(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
    metrics: &SoundSpeedShiftSolverMetrics,
) {
    match config.prior {
        ShiftPrior::Dense => dense::solve_dense_pcg_with_diagonal(
            operator,
            data,
            config,
            workspace,
            &metrics.normal_diagonal,
        ),
        ShiftPrior::Sparse => sparse::solve_sparse_ista_with_lipschitz(
            operator,
            data,
            config,
            workspace,
            metrics.sparse_lipschitz.unwrap_or(f64::EPSILON),
        ),
        // LSQR does not use precomputed diagonal or Lipschitz constant.
        ShiftPrior::Lsqr { .. } => lsqr::solve_shift_lsqr(operator, data, config, workspace),
    }
}
