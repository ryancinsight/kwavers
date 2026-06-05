//! Reverse-mode discrete adjoint for the Westervelt FDTD recurrence.

use super::absorption::{AbsorptionBuilder, FractionalLaplacianAbsorption};
use super::checkpoint::ForwardHistory;
use super::checkpoint::HistoryReplayWorkspace;
use super::encoding::SourceEncoding;
use super::forward::{replay_history_segment_into, ReplayInput, TimeSchedule};
use super::stencil::{index, laplacian, sponge, westervelt_cell_terms, WesterveltCellTerms};
use super::types::{flat_index, Nonlinear3dAperture, Nonlinear3dConfig};

pub(super) struct GradientInput<'a> {
    pub(super) history: &'a ForwardHistory,
    pub(super) cells: usize,
    pub(super) residual: &'a [f64],
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    pub(super) attenuation_np_per_m_mhz: Option<&'a [f64]>,
    pub(super) attenuation_power_law_y: Option<&'a [f64]>,
    pub(super) body: &'a [bool],
    pub(super) source_body_mask: &'a [bool],
    pub(super) n: usize,
    pub(super) spacing_m: f64,
    pub(super) aperture: &'a Nonlinear3dAperture,
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) encoding: SourceEncoding,
    pub(super) source_scale: f64,
    pub(super) dt: f64,
    pub(super) observed_energy: f64,
}

#[derive(Clone, Debug)]
pub(super) struct ParameterGradient {
    pub sound_speed: Vec<f64>,
    pub beta: Vec<f64>,
}

pub(super) fn gradient(input: GradientInput<'_>) -> ParameterGradient {
    let cells = input.cells;
    debug_assert_eq!(cells, input.history.cells());
    let receivers = input
        .aperture
        .receivers
        .iter()
        .map(|idx| flat_index(*idx, input.n))
        .collect::<Vec<_>>();
    let steps = input.history.steps();
    let sponge_weights = sponge(input.n);
    let absorption = build_absorption_for_adjoint(&input);
    let mut grad = ParameterGradient {
        sound_speed: vec![0.0; cells],
        beta: vec![0.0; cells],
    };
    let mut adj_next = vec![0.0; cells];
    let mut adj_curr = vec![0.0; cells];
    let mut adj_prev = vec![0.0; cells];
    let mut adj_older = vec![0.0; cells];
    let mut replay_workspace = HistoryReplayWorkspace::new(cells, input.history.interval());
    add_receiver_residual(
        &mut adj_next,
        steps,
        &receivers,
        input.residual,
        input.observed_energy,
    );
    let mut next_unprocessed = steps;
    while next_unprocessed > 0 {
        replay_history_segment_into(
            ReplayInput {
                history: input.history,
                speed: input.speed,
                density: input.density,
                beta: input.beta,
                attenuation_np_per_m_mhz: input.attenuation_np_per_m_mhz,
                attenuation_power_law_y: input.attenuation_power_law_y,
                source_body_mask: Some(input.source_body_mask),
                n: input.n,
                spacing_m: input.spacing_m,
                aperture: input.aperture,
                config: input.config,
                schedule: input.schedule,
                encoding: input.encoding,
                source_scale: input.source_scale,
                sponge: &sponge_weights,
                step: next_unprocessed - 1,
            },
            &mut replay_workspace,
        );
        let segment = replay_workspace.segment();
        for step in (segment.start_step..segment.end_step).rev() {
            let curr = segment.state(step);
            let prev = if step >= 1 {
                segment.previous_for_step(step)
            } else {
                curr
            };
            if let Some(op) = absorption.as_ref() {
                op.apply_transpose(&adj_next, &mut adj_curr, &mut adj_prev);
            }
            accumulate_step(AccumulateInput {
                adj_curr: &mut adj_curr,
                adj_prev: &mut adj_prev,
                grad: &mut grad,
                adj_next: &adj_next,
                step,
                curr,
                prev,
                input: &input,
                sponge: &sponge_weights,
            });
            add_receiver_residual(
                &mut adj_curr,
                step,
                &receivers,
                input.residual,
                input.observed_energy,
            );
            std::mem::swap(&mut adj_next, &mut adj_curr);
            std::mem::swap(&mut adj_curr, &mut adj_prev);
            std::mem::swap(&mut adj_prev, &mut adj_older);
            adj_older.fill(0.0);
        }
        next_unprocessed = segment.start_step;
    }
    grad
}

fn build_absorption_for_adjoint(
    input: &GradientInput<'_>,
) -> Option<FractionalLaplacianAbsorption> {
    let alpha = input.attenuation_np_per_m_mhz?;
    let y_field = input.attenuation_power_law_y?;
    FractionalLaplacianAbsorption::maybe_new(AbsorptionBuilder {
        n: input.n,
        spacing_m: input.spacing_m,
        dt_s: input.schedule.dt_s,
        speed_m_s: input.speed,
        attenuation_np_per_m_mhz: alpha,
        attenuation_power_law_y: y_field,
    })
}

struct AccumulateInput<'a> {
    adj_curr: &'a mut [f64],
    adj_prev: &'a mut [f64],
    grad: &'a mut ParameterGradient,
    adj_next: &'a [f64],
    step: usize,
    curr: &'a [f64],
    prev: &'a [f64],
    input: &'a GradientInput<'a>,
    sponge: &'a [f64],
}

fn accumulate_step(acc: AccumulateInput<'_>) {
    let input = acc.input;
    let _ = acc.step;
    for x in 1..input.n - 1 {
        for y in 1..input.n - 1 {
            for z in 1..input.n - 1 {
                let i = index(x, y, z, input.n);
                let lambda = acc.adj_next[i] * acc.sponge[i];
                if lambda == 0.0 {
                    continue;
                }
                let c = input.speed[i];
                let lap = laplacian(acc.curr, x, y, z, input.n, input.spacing_m);
                let terms = westervelt_cell_terms(
                    acc.curr[i],
                    acc.prev[i],
                    lap,
                    c,
                    input.density[i],
                    input.beta[i],
                    input.dt,
                );
                acc.adj_curr[i] += 2.0 * lambda;
                acc.adj_prev[i] -= lambda;
                let inv_denom = terms.denominator.recip();
                let lap_scale =
                    (c * input.dt).powi(2) / (input.spacing_m * input.spacing_m) * inv_denom;
                add_laplacian_transpose(acc.adj_curr, i, input.n, lap_scale * lambda);
                add_finite_amplitude_transpose(FiniteAmplitudeTransposeInput {
                    adj_curr: acc.adj_curr,
                    adj_prev: acc.adj_prev,
                    i,
                    lambda,
                    terms,
                });
                if input.body[i] {
                    let inv_denom2 = inv_denom * inv_denom;
                    let dbeta = 1.0 / (input.density[i] * c * c).max(1.0e-18);
                    let d_n_dbeta = 2.0 * dbeta * terms.pressure_increment.powi(2);
                    let d_d_dbeta = -2.0 * dbeta * acc.curr[i];
                    let d_update_dbeta =
                        d_n_dbeta * inv_denom - terms.numerator * d_d_dbeta * inv_denom2;
                    let db_dc = -2.0 * terms.pressure_to_bulk_modulus / c.max(1.0e-18);
                    let d_n_dc = 2.0 * c * input.dt * input.dt * lap
                        + 2.0 * db_dc * terms.pressure_increment.powi(2);
                    let d_d_dc = -2.0 * db_dc * acc.curr[i];
                    let d_update_dc = d_n_dc * inv_denom - terms.numerator * d_d_dc * inv_denom2;
                    acc.grad.sound_speed[i] += lambda * d_update_dc;
                    acc.grad.beta[i] += lambda * d_update_dbeta;
                }
            }
        }
    }
}

fn add_receiver_residual(
    adjoint: &mut [f64],
    state_step: usize,
    receivers: &[usize],
    residual: &[f64],
    observed_energy: f64,
) {
    if state_step == 0 {
        return;
    }
    let trace_step = state_step - 1;
    let scale = observed_energy.recip();
    for (receiver, cell) in receivers.iter().copied().enumerate() {
        let trace = trace_step * receivers.len() + receiver;
        adjoint[cell] += residual[trace] * scale;
    }
}

fn add_laplacian_transpose(adj: &mut [f64], i: usize, n: usize, value: f64) {
    let n2 = n * n;
    adj[i] -= 6.0 * value;
    adj[i - n2] += value;
    adj[i + n2] += value;
    adj[i - n] += value;
    adj[i + n] += value;
    adj[i - 1] += value;
    adj[i + 1] += value;
}

struct FiniteAmplitudeTransposeInput<'a> {
    adj_curr: &'a mut [f64],
    adj_prev: &'a mut [f64],
    i: usize,
    lambda: f64,
    terms: WesterveltCellTerms,
}

fn add_finite_amplitude_transpose(input: FiniteAmplitudeTransposeInput<'_>) {
    let b = input.terms.pressure_to_bulk_modulus;
    if b == 0.0 {
        return;
    }
    let inv_denom = input.terms.denominator.recip();
    let inv_denom2 = inv_denom * inv_denom;
    input.adj_curr[input.i] += input.lambda
        * (4.0 * b * input.terms.pressure_increment * inv_denom
            + 2.0 * b * input.terms.numerator * inv_denom2);
    input.adj_prev[input.i] -= input.lambda * 4.0 * b * input.terms.pressure_increment * inv_denom;
}
