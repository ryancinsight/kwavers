//! Armijo-style line-search over coupled and single-parameter blocks.

use super::candidate::{fill_candidate_beta, fill_candidate_speed, max_body_abs};
use super::objective::line_search_objective;
use super::types::{AcceptedBlock, CandidateAcceptance, LineSearchInput, LineSearchOutcome};

const LINE_SEARCH_SCALES: [f64; 9] = [
    1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
];

pub(in crate::therapy::theranostic_guidance::nonlinear3d) fn apply_line_search(
    input: LineSearchInput<'_>,
) -> Option<LineSearchOutcome> {
    let speed_scale = max_body_abs(input.grad_speed, input.body);
    let beta_scale = max_body_abs(input.grad_beta, input.body);
    if (speed_scale <= 0.0 || !speed_scale.is_finite())
        && (beta_scale <= 0.0 || !beta_scale.is_finite())
    {
        return None;
    }
    let base_speed_step = if speed_scale > 0.0 {
        0.45 * input.config.lesion_delta_c_m_s.abs() / speed_scale
    } else {
        0.0
    };
    let base_beta_step = if beta_scale > 0.0 {
        0.45 * input.config.lesion_delta_beta.abs() / beta_scale
    } else {
        0.0
    };
    input.workspace.resize_for(input.current_speed.len());
    for scale in LINE_SEARCH_SCALES {
        fill_candidate_speed(
            &mut input.workspace.candidate_speed,
            &*input.current_speed,
            input.background_speed,
            input.body,
            input.grad_speed,
            scale,
            base_speed_step,
        );
        fill_candidate_beta(
            &mut input.workspace.candidate_beta,
            &*input.current_beta,
            input.body,
            input.grad_beta,
            scale,
            base_beta_step,
        );
        let coupled_objective = line_search_objective(
            &input.workspace.candidate_speed,
            &input.workspace.candidate_beta,
            &input,
        );
        if coupled_objective < input.objective {
            input
                .current_speed
                .copy_from_slice(&input.workspace.candidate_speed);
            input
                .current_beta
                .copy_from_slice(&input.workspace.candidate_beta);
            return Some(LineSearchOutcome {
                accepted_block: AcceptedBlock::Coupled,
                scale,
                objective: coupled_objective,
            });
        }
    }
    let scale = LINE_SEARCH_SCALES[LINE_SEARCH_SCALES.len() - 1];
    if let Some(acceptance) = first_single_parameter_descent_block(
        input.objective,
        &*input.current_speed,
        &*input.current_beta,
        &input.workspace.candidate_speed,
        &input.workspace.candidate_beta,
        |speed, beta| line_search_objective(speed, beta, &input),
    ) {
        match acceptance.block {
            AcceptedBlock::Coupled => {
                unreachable!("single-parameter fallback cannot accept coupled")
            }
            AcceptedBlock::SpeedOnly => {
                input
                    .current_speed
                    .copy_from_slice(&input.workspace.candidate_speed);
            }
            AcceptedBlock::BetaOnly => {
                input
                    .current_beta
                    .copy_from_slice(&input.workspace.candidate_beta);
            }
        }
        return Some(LineSearchOutcome {
            accepted_block: acceptance.block,
            scale,
            objective: acceptance.objective,
        });
    }
    None
}

/// Tests require access to the block-selection predicates. Only compiled for test builds.
#[cfg(test)]
pub(super) fn first_descent_block<F>(
    objective: f64,
    current_speed: &[f64],
    current_beta: &[f64],
    candidate_speed: &[f64],
    candidate_beta: &[f64],
    mut objective_for_model: F,
) -> Option<CandidateAcceptance>
where
    F: FnMut(&[f64], &[f64]) -> f64,
{
    let coupled_objective = objective_for_model(candidate_speed, candidate_beta);
    if coupled_objective < objective {
        return Some(CandidateAcceptance {
            block: AcceptedBlock::Coupled,
            objective: coupled_objective,
        });
    }
    let speed_only_objective = objective_for_model(candidate_speed, current_beta);
    if speed_only_objective < objective {
        return Some(CandidateAcceptance {
            block: AcceptedBlock::SpeedOnly,
            objective: speed_only_objective,
        });
    }
    let beta_only_objective = objective_for_model(current_speed, candidate_beta);
    if beta_only_objective < objective {
        return Some(CandidateAcceptance {
            block: AcceptedBlock::BetaOnly,
            objective: beta_only_objective,
        });
    }
    None
}

/// Single-parameter-only fallback (excludes coupled candidate). Exposed for testing.
#[cfg(test)]
pub(super) fn first_single_parameter_descent_block_test<F>(
    objective: f64,
    current_speed: &[f64],
    current_beta: &[f64],
    candidate_speed: &[f64],
    candidate_beta: &[f64],
    objective_for_model: F,
) -> Option<CandidateAcceptance>
where
    F: FnMut(&[f64], &[f64]) -> f64,
{
    first_single_parameter_descent_block(
        objective,
        current_speed,
        current_beta,
        candidate_speed,
        candidate_beta,
        objective_for_model,
    )
}

fn first_single_parameter_descent_block<F>(
    objective: f64,
    current_speed: &[f64],
    current_beta: &[f64],
    candidate_speed: &[f64],
    candidate_beta: &[f64],
    mut objective_for_model: F,
) -> Option<CandidateAcceptance>
where
    F: FnMut(&[f64], &[f64]) -> f64,
{
    let speed_only_objective = objective_for_model(candidate_speed, current_beta);
    if speed_only_objective < objective {
        return Some(CandidateAcceptance {
            block: AcceptedBlock::SpeedOnly,
            objective: speed_only_objective,
        });
    }
    let beta_only_objective = objective_for_model(current_speed, candidate_beta);
    if beta_only_objective < objective {
        return Some(CandidateAcceptance {
            block: AcceptedBlock::BetaOnly,
            objective: beta_only_objective,
        });
    }
    None
}
