//! Line-search update for source-encoded multiparameter nonlinear 3-D FWI.

use super::encoding::EncodedTrace;
use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
use super::regularization::h1_penalty;
use super::types::{Nonlinear3dAperture, Nonlinear3dConfig};

pub(super) struct LineSearchInput<'a> {
    pub current_speed: &'a mut [f64],
    pub current_beta: &'a mut [f64],
    pub background_speed: &'a [f64],
    pub background_beta: &'a [f64],
    pub body: &'a [bool],
    pub grad_speed: &'a [f64],
    pub grad_beta: &'a [f64],
    pub objective: f64,
    pub observed: &'a [EncodedTrace],
    pub observed_energy: f64,
    pub density: &'a [f64],
    pub attenuation_np_per_m_mhz: &'a [f64],
    pub attenuation_power_law_y: &'a [f64],
    pub n: usize,
    pub spacing_m: f64,
    pub aperture: &'a Nonlinear3dAperture,
    pub config: &'a Nonlinear3dConfig,
    pub schedule: TimeSchedule,
}

pub(super) fn apply_line_search(input: LineSearchInput<'_>) -> bool {
    let speed_scale = max_body_abs(input.grad_speed, input.body);
    let beta_scale = max_body_abs(input.grad_beta, input.body);
    if (speed_scale <= 0.0 || !speed_scale.is_finite())
        && (beta_scale <= 0.0 || !beta_scale.is_finite())
    {
        return false;
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
    for scale in [
        1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
    ] {
        let candidate_speed = candidate_speed(&input, scale, base_speed_step);
        let candidate_beta = candidate_beta(&input, scale, base_beta_step);
        let candidate_objective = objective_for_model(
            &candidate_speed,
            &candidate_beta,
            ObjectiveInput {
                observed: input.observed,
                observed_energy: input.observed_energy,
                density: input.density,
                attenuation_np_per_m_mhz: input.attenuation_np_per_m_mhz,
                attenuation_power_law_y: input.attenuation_power_law_y,
                background_speed: input.background_speed,
                background_beta: input.background_beta,
                body: input.body,
                n: input.n,
                spacing_m: input.spacing_m,
                aperture: input.aperture,
                config: input.config,
                schedule: input.schedule,
            },
        );
        if candidate_objective < input.objective {
            input.current_speed.copy_from_slice(&candidate_speed);
            input.current_beta.copy_from_slice(&candidate_beta);
            return true;
        }
    }
    false
}

pub(super) struct ObjectiveInput<'a> {
    pub observed: &'a [EncodedTrace],
    pub observed_energy: f64,
    pub density: &'a [f64],
    pub attenuation_np_per_m_mhz: &'a [f64],
    pub attenuation_power_law_y: &'a [f64],
    pub background_speed: &'a [f64],
    pub background_beta: &'a [f64],
    pub body: &'a [bool],
    pub n: usize,
    pub spacing_m: f64,
    pub aperture: &'a Nonlinear3dAperture,
    pub config: &'a Nonlinear3dConfig,
    pub schedule: TimeSchedule,
}

pub(super) fn objective_for_model(speed: &[f64], beta: &[f64], input: ObjectiveInput<'_>) -> f64 {
    let data = input
        .observed
        .iter()
        .map(|shot| {
            let predicted = forward_with_schedule(ForwardInput {
                speed,
                density: input.density,
                beta,
                attenuation_np_per_m_mhz: Some(input.attenuation_np_per_m_mhz),
                attenuation_power_law_y: Some(input.attenuation_power_law_y),
                n: input.n,
                spacing_m: input.spacing_m,
                aperture: input.aperture,
                config: input.config,
                schedule: input.schedule,
                encoding: shot.encoding,
                retain_history: false,
            });
            predicted
                .traces
                .iter()
                .zip(shot.traces.iter())
                .map(|(p, o)| (p - o).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    0.5 * data / input.observed_energy
        + h1_penalty(
            speed,
            input.background_speed,
            input.body,
            input.n,
            input.config.sound_speed_regularization,
            input.config.lesion_delta_c_m_s.abs(),
        )
        + h1_penalty(
            beta,
            input.background_beta,
            input.body,
            input.n,
            input.config.nonlinearity_regularization,
            input.config.lesion_delta_beta.abs(),
        )
}

fn candidate_speed(input: &LineSearchInput<'_>, scale: f64, base_step: f64) -> Vec<f64> {
    input
        .current_speed
        .iter()
        .zip(input.background_speed.iter())
        .zip(input.body.iter())
        .zip(input.grad_speed.iter())
        .map(|(((c, b), active), g)| {
            if *active {
                (c - scale * base_step * g).clamp((b - 160.0).max(343.0), b + 160.0)
            } else {
                *c
            }
        })
        .collect()
}

fn candidate_beta(input: &LineSearchInput<'_>, scale: f64, base_step: f64) -> Vec<f64> {
    input
        .current_beta
        .iter()
        .zip(input.body.iter())
        .zip(input.grad_beta.iter())
        .map(|((b, active), g)| {
            if *active {
                (b - scale * base_step * g).clamp(1.0, 12.0)
            } else {
                *b
            }
        })
        .collect()
}

fn max_body_abs(values: &[f64], body: &[bool]) -> f64 {
    values
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(value.abs()))
        .fold(0.0, f64::max)
}
