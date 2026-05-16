//! Reverse-mode discrete adjoint for the Westervelt FDTD recurrence.

use super::absorption::{AbsorptionBuilder, FractionalLaplacianAbsorption};
use super::checkpoint::ForwardHistory;
use super::checkpoint::HistoryReplayWorkspace;
use super::encoding::SourceEncoding;
use super::forward::{replay_history_segment_into, ReplayInput, TimeSchedule};
use super::stencil::{index, laplacian, nonlinear_term, sponge};
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
            let older = if step >= 2 {
                segment.older_for_step(step)
            } else {
                curr
            };
            if let Some(op) = absorption.as_ref() {
                op.apply_transpose(&adj_next, &mut adj_curr, &mut adj_prev);
            }
            accumulate_step(AccumulateInput {
                adj_curr: &mut adj_curr,
                adj_prev: &mut adj_prev,
                adj_older: &mut adj_older,
                grad: &mut grad,
                adj_next: &adj_next,
                step,
                curr,
                prev,
                older,
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
    adj_older: &'a mut [f64],
    grad: &'a mut ParameterGradient,
    adj_next: &'a [f64],
    step: usize,
    curr: &'a [f64],
    prev: &'a [f64],
    older: &'a [f64],
    input: &'a GradientInput<'a>,
    sponge: &'a [f64],
}

fn accumulate_step(acc: AccumulateInput<'_>) {
    let input = acc.input;
    for x in 1..input.n - 1 {
        for y in 1..input.n - 1 {
            for z in 1..input.n - 1 {
                let i = index(x, y, z, input.n);
                let lambda = acc.adj_next[i] * acc.sponge[i];
                if lambda == 0.0 {
                    continue;
                }
                acc.adj_curr[i] += 2.0 * lambda;
                if acc.step >= 1 {
                    acc.adj_prev[i] -= lambda;
                }
                let a = (input.speed[i] * input.dt).powi(2) / (input.spacing_m * input.spacing_m);
                add_laplacian_transpose(acc.adj_curr, i, input.n, a * lambda);
                let nl = nonlinear_term(acc.curr, acc.prev, acc.older, i, input.dt, acc.step);
                add_nonlinear_transpose(NonlinearTransposeInput {
                    adj_curr: acc.adj_curr,
                    adj_prev: acc.adj_prev,
                    adj_older: acc.adj_older,
                    step: acc.step,
                    i,
                    dt: input.dt,
                    q: input.beta[i] * input.dt * input.dt
                        / (input.density[i] * input.speed[i] * input.speed[i]).max(1.0e-18),
                    lambda,
                    p: acc.curr[i],
                    p_prev: acc.prev[i],
                    p_older: acc.older[i],
                });
                if input.body[i] {
                    let lap = laplacian(acc.curr, x, y, z, input.n, input.spacing_m);
                    let c = input.speed[i];
                    // d(p_{n+1})/dc = 2 c dt^2 \nabla^2 p + d(q)/dc * nl
                    // with the corrected forward `+q nl` and q = beta dt^2 / (rho c^2):
                    //   d(q)/dc = -2 beta dt^2 / (rho c^3)
                    // so d(+q nl)/dc = -(2 beta dt^2 / (rho c^3)) * nl. The leading
                    // 2 c dt^2 Laplacian sensitivity is unchanged.
                    let d_update_dc = 2.0 * c * input.dt * input.dt * lap
                        - 2.0 * input.beta[i] * input.dt * input.dt * nl
                            / (input.density[i] * c.powi(3)).max(1.0e-18);
                    acc.grad.sound_speed[i] += lambda * d_update_dc;
                    let d_update_dbeta =
                        input.dt * input.dt * nl / (input.density[i] * c * c).max(1.0e-18);
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

struct NonlinearTransposeInput<'a> {
    adj_curr: &'a mut [f64],
    adj_prev: &'a mut [f64],
    adj_older: &'a mut [f64],
    step: usize,
    i: usize,
    dt: f64,
    q: f64,
    lambda: f64,
    p: f64,
    p_prev: f64,
    p_older: f64,
}

fn add_nonlinear_transpose(input: NonlinearTransposeInput<'_>) {
    // For forward `p_{n+1} += q * nl`, the adjoint contributions are
    // `adj[step-k] += q * (dnl/dp_{n-k}) * lambda` (positive sign), matching
    // the chain-rule transpose of the corrected forward.
    let inv_dt2 = 1.0 / (input.dt * input.dt);
    if input.step >= 2 {
        let d_curr = (8.0 * input.p - 8.0 * input.p_prev + 2.0 * input.p_older) * inv_dt2;
        let d_prev = (-8.0 * input.p + 4.0 * input.p_prev) * inv_dt2;
        let d_older = 2.0 * input.p * inv_dt2;
        input.adj_curr[input.i] += input.q * d_curr * input.lambda;
        input.adj_prev[input.i] += input.q * d_prev * input.lambda;
        input.adj_older[input.i] += input.q * d_older * input.lambda;
    } else if input.step >= 1 {
        let d_curr = 4.0 * (input.p - input.p_prev) * inv_dt2;
        let d_prev = -d_curr;
        input.adj_curr[input.i] += input.q * d_curr * input.lambda;
        input.adj_prev[input.i] += input.q * d_prev * input.lambda;
    } else {
        input.adj_curr[input.i] += input.q * 4.0 * input.p * inv_dt2 * input.lambda;
    }
}
