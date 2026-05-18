//! Normal-equation application and objective evaluation.

use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{ShiftPrior, SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};
use super::linear_algebra::dot;

pub(super) fn normal_apply(
    operator: &SoundSpeedShiftOperator,
    x: &[f64],
    config: SoundSpeedShiftConfig,
    out: &mut [f64],
    row_workspace: &mut [f64],
    lap_workspace: &mut [f64],
) {
    operator.matvec(x, row_workspace);
    operator.t_matvec(row_workspace, out);
    operator.graph_laplacian_into(x, lap_workspace);
    for ((dst, xv), lv) in out.iter_mut().zip(x.iter()).zip(lap_workspace.iter()) {
        *dst += config.tikhonov_weight * *xv + config.smoothness_weight * *lv;
    }
}

pub(super) fn objective(
    operator: &SoundSpeedShiftOperator,
    x: &[f64],
    data: &[f64],
    config: SoundSpeedShiftConfig,
    prediction: &mut [f64],
    lap_workspace: &mut [f64],
) -> f64 {
    operator.matvec(x, prediction);
    let residual = prediction
        .iter()
        .zip(data.iter())
        .map(|(p, d)| (*p - *d).powi(2))
        .sum::<f64>();
    let norm = x.iter().map(|value| value.powi(2)).sum::<f64>();
    operator.graph_laplacian_into(x, lap_workspace);
    let smooth = x
        .iter()
        .zip(lap_workspace.iter())
        .map(|(value, lap)| *value * *lap)
        .sum::<f64>();
    let sparse = match config.prior {
        ShiftPrior::Dense | ShiftPrior::Lsqr { .. } => 0.0,
        ShiftPrior::Sparse => x.iter().map(|value| value.abs()).sum::<f64>(),
    };

    0.5 * residual
        + 0.5 * config.tikhonov_weight * norm
        + 0.5 * config.smoothness_weight * smooth
        + config.sparsity_weight * sparse
}

pub(super) fn estimate_lipschitz(
    operator: &SoundSpeedShiftOperator,
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) -> f64 {
    let scale = (operator.cols() as f64).sqrt().recip();
    workspace.power_vector.fill(scale);
    let mut eigenvalue = f64::EPSILON;

    for _ in 0..32 {
        normal_apply(
            operator,
            &workspace.power_vector,
            config,
            &mut workspace.power_normal,
            &mut workspace.row,
            &mut workspace.laplacian,
        );
        eigenvalue = dot(&workspace.power_vector, &workspace.power_normal).abs();
        let norm = workspace
            .power_normal
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            break;
        }
        for (vv, hvv) in workspace
            .power_vector
            .iter_mut()
            .zip(workspace.power_normal.iter())
        {
            *vv = *hvv / norm;
        }
    }

    eigenvalue
}
