//! Dense Tikhonov/H1 solve via preconditioned conjugate gradients.

use super::super::operator::SoundSpeedShiftOperator;
use super::super::types::{SoundSpeedShiftConfig, SoundSpeedShiftWorkspace};
use super::linear_algebra::{axpy, dot};
use super::normal::{normal_apply, objective};

pub(super) fn solve_dense_pcg(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    let mut normal_diagonal = vec![0.0; operator.cols()];
    operator.normal_diag_into(&mut normal_diagonal);
    solve_dense_pcg_with_diagonal(operator, data, config, workspace, &normal_diagonal);
}

pub(super) fn solve_dense_pcg_with_diagonal(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
    normal_diagonal: &[f64],
) {
    workspace.prepare(operator.rows(), operator.cols());
    debug_assert_eq!(normal_diagonal.len(), operator.cols());
    operator.t_matvec(data, &mut workspace.rhs);
    let rhs_norm = dot(&workspace.rhs, &workspace.rhs).sqrt().max(f64::EPSILON);
    workspace.diagonal.copy_from_slice(normal_diagonal);
    for value in &mut workspace.diagonal {
        *value = (*value + config.tikhonov_weight).max(f64::EPSILON);
    }

    workspace.solution.fill(0.0);
    normal_apply(
        operator,
        &workspace.solution,
        config,
        &mut workspace.normal_solution,
        &mut workspace.row,
        &mut workspace.laplacian,
    );
    for ((rv, rhs), normal) in workspace
        .residual
        .iter_mut()
        .zip(workspace.rhs.iter())
        .zip(workspace.normal_solution.iter())
    {
        *rv = *rhs - *normal;
    }
    for ((zv, rv), diagonal) in workspace
        .preconditioned
        .iter_mut()
        .zip(workspace.residual.iter())
        .zip(workspace.diagonal.iter())
    {
        *zv = *rv / *diagonal;
    }
    workspace
        .direction
        .copy_from_slice(&workspace.preconditioned);
    let mut rz_old = dot(&workspace.residual, &workspace.preconditioned);
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
            &workspace.direction,
            config,
            &mut workspace.normal_direction,
            &mut workspace.row,
            &mut workspace.laplacian,
        );
        let denom = dot(&workspace.direction, &workspace.normal_direction);
        if denom <= f64::EPSILON {
            break;
        }
        let alpha = rz_old / denom;
        axpy(alpha, &workspace.direction, &mut workspace.solution);
        axpy(-alpha, &workspace.normal_direction, &mut workspace.residual);
        for ((zv, rv), diagonal) in workspace
            .preconditioned
            .iter_mut()
            .zip(workspace.residual.iter())
            .zip(workspace.diagonal.iter())
        {
            *zv = *rv / *diagonal;
        }
        let rz_new = dot(&workspace.residual, &workspace.preconditioned);
        workspace.objective_history.push(objective(
            operator,
            &workspace.solution,
            data,
            config,
            &mut workspace.prediction,
            &mut workspace.laplacian,
        ));
        if rz_new.sqrt() <= 1.0e-5 * rhs_norm {
            break;
        }
        let beta = rz_new / rz_old.max(f64::EPSILON);
        for (direction, preconditioned) in workspace
            .direction
            .iter_mut()
            .zip(workspace.preconditioned.iter())
        {
            *direction = *preconditioned + beta * *direction;
        }
        rz_old = rz_new;
    }
}
