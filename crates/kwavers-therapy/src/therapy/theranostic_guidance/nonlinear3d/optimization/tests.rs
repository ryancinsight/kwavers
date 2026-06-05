use super::candidate::{fill_candidate_beta, fill_candidate_speed};
use super::search::{first_descent_block, first_single_parameter_descent_block_test};
use super::types::{AcceptedBlock, LineSearchWorkspace};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

#[test]
fn line_search_selects_single_parameter_descent_when_coupled_step_increases_objective() {
    let current_speed = [0.0, 0.0];
    let current_beta = [0.0, 0.0];
    let speed_descent = [1.0, 0.0];
    let beta_regression = [8.0, 0.0];
    let speed_regression = [8.0, 0.0];
    let beta_descent = [1.0, 0.0];

    let speed_block = first_descent_block(
        1.0,
        &current_speed,
        &current_beta,
        &speed_descent,
        &beta_regression,
        |speed, beta| (speed[0] - 1.0).powi(2) + beta[0].powi(2),
    );
    let speed_block = speed_block.expect("speed-only candidate should descend");
    assert_eq!(speed_block.block, AcceptedBlock::SpeedOnly);
    assert_eq!(speed_block.objective, 0.0);

    let beta_block = first_descent_block(
        1.0,
        &current_speed,
        &current_beta,
        &speed_regression,
        &beta_descent,
        |speed, beta| speed[0].powi(2) + (beta[0] - 1.0).powi(2),
    );
    let beta_block = beta_block.expect("beta-only candidate should descend");
    assert_eq!(beta_block.block, AcceptedBlock::BetaOnly);
    assert_eq!(beta_block.objective, 0.0);
}

#[test]
fn line_search_single_parameter_fallback_excludes_coupled_candidate() {
    let current_speed = [0.0, 0.0];
    let current_beta = [0.0, 0.0];
    let candidate_speed = [1.0, 0.0];
    let candidate_beta = [8.0, 0.0];

    let accepted = first_single_parameter_descent_block_test(
        1.0,
        &current_speed,
        &current_beta,
        &candidate_speed,
        &candidate_beta,
        |speed, beta| (speed[0] - 1.0).powi(2) + beta[0].powi(2),
    )
    .expect("single-parameter fallback should accept the speed block");

    assert_eq!(accepted.block, AcceptedBlock::SpeedOnly);
    assert_eq!(accepted.objective, 0.0);
}

#[test]
fn line_search_prefers_coupled_descent_when_both_parameters_reduce_objective() {
    let current_speed = [0.0, 0.0];
    let current_beta = [0.0, 0.0];
    let candidate_speed = [1.0, 0.0];
    let candidate_beta = [1.0, 0.0];

    let accepted = first_descent_block(
        2.0,
        &current_speed,
        &current_beta,
        &candidate_speed,
        &candidate_beta,
        |speed, beta| (speed[0] - 1.0).powi(2) + (beta[0] - 1.0).powi(2),
    );

    let accepted = accepted.expect("coupled candidate should descend");
    assert_eq!(accepted.block, AcceptedBlock::Coupled);
    assert_eq!(accepted.objective, 0.0);
}

#[test]
fn line_search_workspace_reuses_candidate_buffers_and_preserves_inactive_cells() {
    let mut workspace = LineSearchWorkspace::new(4);
    let initial_speed_capacity = workspace.candidate_speed.capacity();
    let initial_beta_capacity = workspace.candidate_beta.capacity();
    let current_speed = [SOUND_SPEED_WATER_SIM, 1520.0, 1540.0, 1560.0];
    let background_speed = [SOUND_SPEED_WATER_SIM, 1520.0, 1540.0, 1560.0];
    let current_beta = [4.0, 5.0, 6.0, 7.0];
    let body = [true, false, true, true];
    let grad_speed = [2.0, 99.0, -4.0, 1.0e6];
    let grad_beta = [0.5, 99.0, -0.25, -1.0e6];

    fill_candidate_speed(
        &mut workspace.candidate_speed,
        &current_speed,
        &background_speed,
        &body,
        &grad_speed,
        0.5,
        10.0,
    );
    fill_candidate_beta(
        &mut workspace.candidate_beta,
        &current_beta,
        &body,
        &grad_beta,
        0.5,
        2.0,
    );

    assert_eq!(workspace.candidate_speed[0], 1490.0);
    assert_eq!(workspace.candidate_speed[1], current_speed[1]);
    assert_eq!(workspace.candidate_speed[2], 1560.0);
    assert_eq!(workspace.candidate_speed[3], 1400.0);
    assert_eq!(workspace.candidate_beta[0], 3.5);
    assert_eq!(workspace.candidate_beta[1], current_beta[1]);
    assert_eq!(workspace.candidate_beta[2], 6.25);
    assert_eq!(workspace.candidate_beta[3], 12.0);

    workspace.resize_for(4);

    assert_eq!(workspace.candidate_speed.capacity(), initial_speed_capacity);
    assert_eq!(workspace.candidate_beta.capacity(), initial_beta_capacity);
}
