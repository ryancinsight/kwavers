//! Iterative elastic FWI loop with ordered backtracking line search.
//!
//! # Line search
//!
//! Each outer FWI iteration contains an inner line-search over up to
//! `ELASTIC_FWI_MAX_LINE_SEARCH_STEPS` candidate step sizes. Each candidate
//! requires a full elastic PSTD forward propagation, so candidates are
//! evaluated lazily in descending step-size order.
//!
//! # Theorem (descent correctness)
//!
//! The accepted candidate is the first `k` in `0..MAX_STEPS` for which
//! `f(model + step_k * gradient) < f(model)`. Lazy ordered evaluation yields
//! the same accepted index as eager evaluation because later candidates cannot
//! precede the first improving candidate in that total order.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::extensions::ElasticPstdVelocitySource;
use leto::{
    Array2,
    Array3,
};

use super::super::config::TheranosticInverseConfig;
use super::super::exposure::normalize_positive;
use super::super::geometry::Point2;
use super::super::medium::PreparedTheranosticSlice;
use super::medium::elastic_medium;
use super::propagation::propagate_traces;
use super::sampling::{migrate_residual, trace_energy};
use super::types::ElasticFwiInversion;

const ELASTIC_FWI_MAX_LINE_SEARCH_STEPS: usize = 4;
const ELASTIC_FWI_INITIAL_STEP: f64 = 0.75;

/// One line-search candidate: `(updated model, simulated traces,
/// data residual, objective value)`.
type ElasticCandidate = (Array2<f64>, Array2<f64>, Array2<f64>, f64);

#[allow(clippy::too_many_arguments)]
pub(super) fn run_iterative_elastic_fwi(
    prepared: &PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
    grid: &Grid,
    source: &ElasticPstdVelocitySource,
    receiver_mask: &Array3<bool>,
    observed_traces: &Array2<f64>,
    baseline_traces: &Array2<f64>,
    dt_s: f64,
    center_frequency_hz: f64,
    source_points: &[Point2],
    receiver_points: &[Point2],
    lesion_fraction: f64,
) -> KwaversResult<ElasticFwiInversion> {
    let mut model = Array2::<f64>::zeros(prepared.ct_hu.shape());
    let mut predicted_traces = baseline_traces.clone();
    let mut residual = observed_traces - &predicted_traces;
    let mut objective = fwi_objective(&residual);
    let mut objective_history = vec![objective];
    let mut accepted_step_count = 0usize;

    for _ in 0..config.elastic_fwi_iterations {
        let gradient = normalize_positive(
            &migrate_residual(
                prepared,
                &residual,
                dt_s,
                center_frequency_hz,
                source_points,
                receiver_points,
                config.elastic_shear_speed_m_s,
            ),
            &prepared.body_mask,
        );

        // Ordered backtracking line search: evaluate only the candidates that
        // can affect the first-improving descent decision.
        let mut accepted = None;
        for search_step in 0..ELASTIC_FWI_MAX_LINE_SEARCH_STEPS {
            let (candidate, traces, candidate_residual, candidate_objective) = evaluate_candidate(
                prepared,
                config,
                grid,
                source,
                receiver_mask,
                observed_traces,
                predicted_traces.shape()[1],
                dt_s,
                lesion_fraction,
                &model,
                &gradient,
                search_step,
            )?;
            if candidate_objective < objective {
                accepted = Some((candidate, traces, candidate_residual, candidate_objective));
                break;
            }
        }

        let Some((next_model, next_traces, next_residual, next_objective)) = accepted else {
            objective_history.push(objective);
            break;
        };
        model = next_model;
        predicted_traces = next_traces;
        residual = next_residual;
        objective = next_objective;
        accepted_step_count += 1;
        objective_history.push(objective);
    }

    Ok(ElasticFwiInversion {
        model,
        iteration_count: objective_history.len().saturating_sub(1),
        accepted_step_count,
        objective_history,
        final_residual_energy: trace_energy(&residual),
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_candidate(
    prepared: &PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
    grid: &Grid,
    source: &ElasticPstdVelocitySource,
    receiver_mask: &Array3<bool>,
    observed_traces: &Array2<f64>,
    time_steps: usize,
    dt_s: f64,
    lesion_fraction: f64,
    model: &Array2<f64>,
    gradient: &Array2<f64>,
    search_step: usize,
) -> KwaversResult<ElasticCandidate> {
    let scale = ELASTIC_FWI_INITIAL_STEP * 0.5_f64.powi(search_step as i32);
    let candidate = candidate_model(model, gradient, &prepared.body_mask, scale);
    let candidate_medium = elastic_medium(
        prepared,
        &candidate,
        config.elastic_shear_speed_m_s,
        lesion_fraction,
    );
    let candidate_traces = propagate_traces(
        grid,
        candidate_medium,
        dt_s,
        time_steps,
        source,
        receiver_mask,
    )?;
    let candidate_residual = observed_traces - &candidate_traces;
    let candidate_objective = fwi_objective(&candidate_residual);
    Ok((
        candidate,
        candidate_traces,
        candidate_residual,
        candidate_objective,
    ))
}

fn candidate_model(
    model: &Array2<f64>,
    gradient: &Array2<f64>,
    body_mask: &Array2<bool>,
    scale: f64,
) -> Array2<f64> {
    Array2::from_shape_fn(model.shape(), |idx| {
        if body_mask[idx] {
            (model[idx] + scale * gradient[idx]).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

fn fwi_objective(residual: &Array2<f64>) -> f64 {
    0.5 * trace_energy(residual)
}
