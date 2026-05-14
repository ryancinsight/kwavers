//! 3-D Westervelt propagation and discrete-adjoint FWI.
//!
//! The forward recurrence is the same finite-difference Westervelt update
//! documented in `solver::forward::nonlinear::westervelt`, specialized here
//! to CT-derived arrays so the inverse path can expose every pressure history
//! to a reverse-mode discrete adjoint.
//!
//! # Theorem
//!
//! For recurrence
//! `p[n+1] = S * (2 p[n] - p[n-1] + c^2 dt^2 L p[n] + q dtt(p^2)[n] + s[n])`
//! with fixed sponge `S`, fixed density and nonlinearity coefficient
//! `q = beta dt^2 / (rho c^2)`, the reverse sweep in `gradient` computes the
//! exact derivative of the discrete trace least-squares objective with respect
//! to the nodal sound speed.
//!
//! # Proof sketch
//!
//! Each loop applies the transpose of the Jacobian of the scalar recurrence
//! to the adjoint variables and sums the local `∂p[n+1]/∂c` term;
//! reverse-mode accumulation over an acyclic time-unrolled graph is the chain
//! rule.
//!
//! # Performance contract
//!
//! - The forward pressure history is stored as exact sparse checkpoints.
//!   Each checkpoint contains `p[n-2]`, `p[n-1]`, and `p[n]`; the reverse
//!   sweep replays one bounded interval at a time with the same recurrence.
//!   This preserves dense-history gradients while reducing retained forward
//!   state from `O(steps * cells)` to `O((steps / interval + interval) *
//!   cells)`.
//! - The adjoint variables use four rolling `Vec<f64>` states for
//!   `lambda[n+1]`, `lambda[n]`, `lambda[n-1]`, and `lambda[n-2]`. This is
//!   algebraically equivalent to storing `(steps + 1)` adjoint states because
//!   the Westervelt recurrence has temporal stencil width three; the reverse
//!   sweep never reads an adjoint state after it shifts past this window.
//! - The forward cell update is rayon-parallel: each cell writes only to its
//!   own `next[i]`, so the outer 3-D loop dispatches through
//!   `par_iter_mut().enumerate()` without coloring, atomics, or locks.
//! - The four rotating buffers (older, previous, current, next) are
//!   `mem::swap`-rotated each step. No `vec![0.0; cells]` allocation occurs
//!   inside the time loop.

use ndarray::Array3;

use super::adjoint::{gradient, GradientInput, ParameterGradient};
use super::encoding::{EncodedTrace, SourceEncoding};
use super::forward::{forward_with_schedule, time_schedule, ForwardInput};
use super::metrics::metrics_from_score;
use super::optimization::{
    apply_line_search, objective_for_model, LineSearchInput, ObjectiveInput,
};
use super::regularization::{add_h1_gradient, h1_penalty, multiparameter_score, smooth_gradient};
use super::stencil::index;
use super::types::{
    Nonlinear3dAperture, Nonlinear3dConfig, Nonlinear3dVolume, VolumeReconstructionMetrics,
};

#[derive(Clone, Debug)]
pub(crate) struct WesterveltFwiResult {
    pub reconstructed_sound_speed_m_s: Array3<f64>,
    pub reconstructed_delta_c_m_s: Array3<f64>,
    pub reconstructed_beta: Array3<f64>,
    pub reconstructed_delta_beta: Array3<f64>,
    pub multiparameter_fwi_score: Array3<f64>,
    pub peak_pressure_pa: Array3<f64>,
    pub objective_history: Vec<f64>,
    pub metrics: VolumeReconstructionMetrics,
    pub dt_s: f64,
    pub time_steps: usize,
}

pub(crate) fn run_fwi(
    volume: &Nonlinear3dVolume,
    aperture: &Nonlinear3dAperture,
    config: &Nonlinear3dConfig,
) -> WesterveltFwiResult {
    let n = volume.body_mask.dim().0;
    let true_speed = flatten(&volume.true_sound_speed_m_s);
    let background = flatten(&volume.background_sound_speed_m_s);
    let density = flatten(&volume.density_kg_m3);
    let true_beta = flatten(&volume.true_beta);
    let background_beta = flatten(&volume.background_beta);
    let attenuation_alpha0 = flatten(&volume.attenuation_np_per_m_mhz);
    let attenuation_y = flatten(&volume.attenuation_power_law_y);
    let body = volume.body_mask.iter().copied().collect::<Vec<_>>();
    let inversion = volume.inversion_mask.iter().copied().collect::<Vec<_>>();
    let target = volume.target_mask.iter().copied().collect::<Vec<_>>();
    let schedule = time_schedule(&true_speed, n, volume.spacing_m, config);
    let encodings = SourceEncoding::all(config.source_encoding_count);
    let mut therapy_peak = vec![0.0; n * n * n];
    let observed = encodings
        .iter()
        .copied()
        .map(|encoding| {
            let forward = forward_with_schedule(ForwardInput {
                speed: &true_speed,
                density: &density,
                beta: &true_beta,
                attenuation_np_per_m_mhz: Some(&attenuation_alpha0),
                attenuation_power_law_y: Some(&attenuation_y),
                n,
                spacing_m: volume.spacing_m,
                aperture,
                config,
                schedule,
                encoding,
                retain_history: false,
            });
            if encoding.index == 0 {
                therapy_peak.copy_from_slice(&forward.peak_pressure);
            }
            EncodedTrace {
                encoding,
                traces: forward.traces,
            }
        })
        .collect::<Vec<_>>();
    let observed_energy = observed
        .iter()
        .flat_map(|shot| shot.traces.iter())
        .map(|value| value * value)
        .sum::<f64>()
        .max(1.0e-24);
    let mut current = background.clone();
    let mut current_beta = background_beta.clone();
    let mut objective_history = Vec::with_capacity(config.iterations + 1);
    let cells = n * n * n;
    for _ in 0..config.iterations {
        let mut data_objective = 0.0;
        let mut grad = ParameterGradient {
            sound_speed: vec![0.0; cells],
            beta: vec![0.0; cells],
        };
        for shot in &observed {
            let predicted = forward_with_schedule(ForwardInput {
                speed: &current,
                density: &density,
                beta: &current_beta,
                attenuation_np_per_m_mhz: Some(&attenuation_alpha0),
                attenuation_power_law_y: Some(&attenuation_y),
                n,
                spacing_m: volume.spacing_m,
                aperture,
                config,
                schedule,
                encoding: shot.encoding,
                retain_history: true,
            });
            let residual = predicted
                .traces
                .iter()
                .zip(shot.traces.iter())
                .map(|(p, o)| p - o)
                .collect::<Vec<_>>();
            data_objective += residual.iter().map(|value| value * value).sum::<f64>();
            let history = predicted
                .history
                .as_ref()
                .expect("FWI forward pass must retain pressure history");
            let shot_grad = gradient(GradientInput {
                history,
                cells,
                residual: &residual,
                speed: &current,
                density: &density,
                beta: &current_beta,
                attenuation_np_per_m_mhz: Some(&attenuation_alpha0),
                attenuation_power_law_y: Some(&attenuation_y),
                body: &inversion,
                n,
                spacing_m: volume.spacing_m,
                aperture,
                config,
                schedule,
                encoding: shot.encoding,
                dt: predicted.dt_s,
                observed_energy,
            });
            accumulate_gradient(&mut grad, &shot_grad);
        }
        let objective = 0.5 * data_objective / observed_energy
            + h1_penalty(
                &current,
                &background,
                &inversion,
                n,
                config.sound_speed_regularization,
                config.lesion_delta_c_m_s.abs(),
            )
            + h1_penalty(
                &current_beta,
                &background_beta,
                &inversion,
                n,
                config.nonlinearity_regularization,
                config.lesion_delta_beta.abs(),
            );
        objective_history.push(objective);
        add_h1_gradient(
            &mut grad.sound_speed,
            &current,
            &background,
            &inversion,
            n,
            config.sound_speed_regularization,
            config.lesion_delta_c_m_s.abs(),
        );
        add_h1_gradient(
            &mut grad.beta,
            &current_beta,
            &background_beta,
            &inversion,
            n,
            config.nonlinearity_regularization,
            config.lesion_delta_beta.abs(),
        );
        smooth_gradient(
            &mut grad.sound_speed,
            &inversion,
            n,
            config.gradient_smoothing_steps,
        );
        smooth_gradient(
            &mut grad.beta,
            &inversion,
            n,
            config.gradient_smoothing_steps,
        );
        if !apply_line_search(LineSearchInput {
            current_speed: &mut current,
            current_beta: &mut current_beta,
            background_speed: &background,
            background_beta: &background_beta,
            body: &inversion,
            grad_speed: &grad.sound_speed,
            grad_beta: &grad.beta,
            objective,
            observed: &observed,
            observed_energy,
            density: &density,
            attenuation_np_per_m_mhz: &attenuation_alpha0,
            attenuation_power_law_y: &attenuation_y,
            n,
            spacing_m: volume.spacing_m,
            aperture,
            config,
            schedule,
        }) {
            break;
        }
    }
    objective_history.push(objective_for_model(
        &current,
        &current_beta,
        ObjectiveInput {
            observed: &observed,
            observed_energy,
            density: &density,
            attenuation_np_per_m_mhz: &attenuation_alpha0,
            attenuation_power_law_y: &attenuation_y,
            background_speed: &background,
            background_beta: &background_beta,
            body: &inversion,
            n,
            spacing_m: volume.spacing_m,
            aperture,
            config,
            schedule,
        },
    ));
    let delta = current
        .iter()
        .zip(background.iter())
        .map(|(c, b)| c - b)
        .collect::<Vec<_>>();
    let delta_beta = current_beta
        .iter()
        .zip(background_beta.iter())
        .map(|(b, b0)| b - b0)
        .collect::<Vec<_>>();
    let score = multiparameter_score(&delta, &delta_beta, &inversion, config, n);
    let score_vec = score.iter().copied().collect::<Vec<_>>();
    WesterveltFwiResult {
        reconstructed_sound_speed_m_s: unflatten(&current, n),
        reconstructed_delta_c_m_s: unflatten(&delta, n),
        reconstructed_beta: unflatten(&current_beta, n),
        reconstructed_delta_beta: unflatten(&delta_beta, n),
        multiparameter_fwi_score: score,
        peak_pressure_pa: unflatten(&therapy_peak, n),
        objective_history,
        metrics: metrics_from_score(&score_vec, &target, &body),
        dt_s: schedule.dt_s,
        time_steps: schedule.time_steps,
    }
}

fn accumulate_gradient(total: &mut ParameterGradient, shot: &ParameterGradient) {
    for (dst, src) in total.sound_speed.iter_mut().zip(shot.sound_speed.iter()) {
        *dst += src;
    }
    for (dst, src) in total.beta.iter_mut().zip(shot.beta.iter()) {
        *dst += src;
    }
}

fn flatten(values: &Array3<f64>) -> Vec<f64> {
    values.iter().copied().collect()
}

fn unflatten(values: &[f64], n: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, n, n), |(x, y, z)| values[index(x, y, z, n)])
}
