//! Inverse solvers for speed-of-sound shift imaging.

mod dense;
mod linear_algebra;
mod normal;
mod sparse;
mod workspace;

use super::operator::SoundSpeedShiftOperator;
use super::types::{ShiftPrior, SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};

pub(super) fn solve_shift(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    match config.prior {
        ShiftPrior::Dense => dense::solve_dense_pcg(operator, data, config, workspace),
        ShiftPrior::Sparse => sparse::solve_sparse_ista(operator, data, config, workspace),
    }
}
