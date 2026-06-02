//! Sparse L1 solve via proximal gradient iterations.

use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};
use super::linear_algebra::soft_threshold;
use super::normal::{estimate_lipschitz, normal_apply, objective};

pub(super) fn solve_sparse_ista(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    workspace.prepare(operator.rows(), operator.cols());
    let lipschitz = estimate_lipschitz(operator, config, workspace).max(f64::EPSILON);
    solve_sparse_ista_with_lipschitz(operator, data, config, workspace, lipschitz);
}

pub(super) fn solve_sparse_ista_with_lipschitz(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
    lipschitz: f64,
) {
    workspace.prepare(operator.rows(), operator.cols());
    operator.t_matvec(data, &mut workspace.rhs);
    let step = 1.0 / lipschitz.max(f64::EPSILON);
    let threshold = config.sparsity_weight * step;
    workspace.solution.fill(0.0);
    workspace.objective_history.clear();
    workspace.objective_history.push(objective(
        operator,
        &workspace.solution,
        data,
        config,
        &mut workspace.prediction,
        &mut workspace.laplacian,
    ));

    for _ in 0..config.iterations {
        normal_apply(
            operator,
            &workspace.solution,
            config,
            &mut workspace.normal_solution,
            &mut workspace.row,
            &mut workspace.laplacian,
        );
        workspace
            .previous_solution
            .copy_from_slice(&workspace.solution);
        for ((xv, hv), rv) in workspace
            .solution
            .iter_mut()
            .zip(workspace.normal_solution.iter())
            .zip(workspace.rhs.iter())
        {
            *xv = soft_threshold(*xv - step * (*hv - *rv), threshold);
        }
        workspace.objective_history.push(objective(
            operator,
            &workspace.solution,
            data,
            config,
            &mut workspace.prediction,
            &mut workspace.laplacian,
        ));
        let change = workspace
            .solution
            .iter()
            .zip(workspace.previous_solution.iter())
            .map(|(a, b)| (*a - *b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = workspace
            .solution
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if change <= 1.0e-5 * norm.max(1.0) {
            break;
        }
    }
}
