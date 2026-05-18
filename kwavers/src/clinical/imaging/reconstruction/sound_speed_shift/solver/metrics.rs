//! Frame-invariant solver metrics for fixed acquisition plans.

use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{ShiftPrior, SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};
use super::normal::estimate_lipschitz;

/// Metrics that depend on the operator and regularization, not frame data.
#[derive(Clone, Debug)]
pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) struct SoundSpeedShiftSolverMetrics
{
    pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) normal_diagonal: Vec<f64>,
    pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) sparse_lipschitz:
        Option<f64>,
}

pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) fn compute_solver_metrics(
    operator: &SoundSpeedShiftOperator,
    config: SoundSpeedShiftConfig,
) -> SoundSpeedShiftSolverMetrics {
    let mut normal_diagonal = vec![0.0; operator.cols()];
    operator.normal_diag_into(&mut normal_diagonal);
    let sparse_lipschitz = match config.prior {
        ShiftPrior::Dense | ShiftPrior::Lsqr { .. } => None,
        ShiftPrior::Sparse => {
            let mut workspace = SoundSpeedShiftWorkspace::new();
            workspace.prepare(operator.rows(), operator.cols());
            Some(estimate_lipschitz(operator, config, &mut workspace).max(f64::EPSILON))
        }
    };

    SoundSpeedShiftSolverMetrics {
        normal_diagonal,
        sparse_lipschitz,
    }
}
