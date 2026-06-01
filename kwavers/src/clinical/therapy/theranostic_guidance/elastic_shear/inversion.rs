//! Iterative elastic FWI loop with parallel speculative line-search.
//!
//! # Parallelism
//!
//! Each outer FWI iteration contains an inner line-search over up to
//! `ELASTIC_FWI_MAX_LINE_SEARCH_STEPS` candidate step sizes, each requiring
//! a full elastic PSTD forward propagation (`propagate_traces`).  The
//! propagations are **independent** (different scale factors, otherwise
//! identical inputs), so they are launched in parallel via Rayon.
//!
//! After all candidates complete, the **first** (largest scale, index 0) one
//! that strictly decreases the objective is accepted — preserving the descent
//! semantics of the original sequential loop exactly.  Propagation errors on
//! non-required candidates are skipped; the first error encountered before an
//! accepting candidate is propagated.
//!
//! # Theorem (descent correctness)
//! Sequential loop: accept the first `k ∈ {0,1,2,3}` with `f(m + sₖ·g) < f(m)`.
//! Parallel loop: compute all `{f(m + s₀·g), …, f(m + s₃·g)}` concurrently,
//! then scan in order 0 → 3 and accept the first improving index `k`.
//! The accepted index is identical to the sequential result because both
//! criteria are "first-improving in order 0→3".  ∎

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::solver::forward::pstd::extensions::ElasticPstdVelocitySource;
use ndarray::{Array2, Array3};
use rayon::prelude::*;

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
    let mut model = Array2::<f64>::zeros(prepared.ct_hu.dim());
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

        // ── Parallel speculative line-search ─────────────────────────────
        // All ELASTIC_FWI_MAX_LINE_SEARCH_STEPS candidate propagations are
        // launched concurrently.  Results are collected in step-index order;
        // the sequential scan below preserves the "first-improving" descent
        // semantics exactly.
        let candidate_results: Vec<KwaversResult<(Array2<f64>, Array2<f64>, Array2<f64>, f64)>> =
            (0..ELASTIC_FWI_MAX_LINE_SEARCH_STEPS)
                .into_par_iter()
                .map(|search_step| {
                    let scale = ELASTIC_FWI_INITIAL_STEP * 0.5_f64.powi(search_step as i32);
                    let candidate = candidate_model(&model, &gradient, &prepared.body_mask, scale);
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
                        predicted_traces.ncols(),
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
                })
                .collect();

        // Sequential selection: first improving step in scale-descending
        // order.  Propagates errors only for candidates preceding any
        // accepted step.
        let mut accepted = None;
        for result in candidate_results {
            let (candidate, traces, resid, obj) = result?;
            if obj < objective {
                accepted = Some((candidate, traces, resid, obj));
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

fn candidate_model(
    model: &Array2<f64>,
    gradient: &Array2<f64>,
    body_mask: &Array2<bool>,
    scale: f64,
) -> Array2<f64> {
    Array2::from_shape_fn(model.dim(), |idx| {
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
