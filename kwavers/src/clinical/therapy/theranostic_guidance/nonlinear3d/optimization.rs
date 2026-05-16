//! Line-search update for source-encoded multiparameter nonlinear 3-D FWI.

use super::encoding::EncodedTrace;
use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
use super::regularization::h1_penalty;
use super::types::{Nonlinear3dAperture, Nonlinear3dConfig};

pub(super) struct LineSearchInput<'a> {
    pub current_speed: &'a mut [f64],
    pub current_beta: &'a mut [f64],
    pub workspace: &'a mut LineSearchWorkspace,
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
    pub source_scale: f64,
}

#[derive(Clone, Debug)]
pub(super) struct LineSearchWorkspace {
    candidate_speed: Vec<f64>,
    candidate_beta: Vec<f64>,
}

impl LineSearchWorkspace {
    #[must_use]
    pub(super) fn new(cells: usize) -> Self {
        Self {
            candidate_speed: vec![0.0; cells],
            candidate_beta: vec![0.0; cells],
        }
    }

    fn resize_for(&mut self, cells: usize) {
        self.candidate_speed.resize(cells, 0.0);
        self.candidate_beta.resize(cells, 0.0);
    }
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
    input.workspace.resize_for(input.current_speed.len());
    for scale in [
        1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
    ] {
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
        let candidate_objective = objective_for_model(
            &input.workspace.candidate_speed,
            &input.workspace.candidate_beta,
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
                source_scale: input.source_scale,
            },
        );
        if candidate_objective < input.objective {
            input
                .current_speed
                .copy_from_slice(&input.workspace.candidate_speed);
            input
                .current_beta
                .copy_from_slice(&input.workspace.candidate_beta);
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
    pub source_scale: f64,
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
                source_scale: input.source_scale,
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

fn fill_candidate_speed(
    out: &mut [f64],
    current_speed: &[f64],
    background_speed: &[f64],
    body: &[bool],
    grad_speed: &[f64],
    scale: f64,
    base_step: f64,
) {
    debug_assert_eq!(out.len(), current_speed.len());
    debug_assert_eq!(out.len(), background_speed.len());
    debug_assert_eq!(out.len(), body.len());
    debug_assert_eq!(out.len(), grad_speed.len());
    for ((((dst, c), b), active), g) in out
        .iter_mut()
        .zip(current_speed.iter())
        .zip(background_speed.iter())
        .zip(body.iter())
        .zip(grad_speed.iter())
    {
        *dst = if *active {
            (c - scale * base_step * g).clamp((b - 160.0).max(343.0), b + 160.0)
        } else {
            *c
        };
    }
}

fn fill_candidate_beta(
    out: &mut [f64],
    current_beta: &[f64],
    body: &[bool],
    grad_beta: &[f64],
    scale: f64,
    base_step: f64,
) {
    debug_assert_eq!(out.len(), current_beta.len());
    debug_assert_eq!(out.len(), body.len());
    debug_assert_eq!(out.len(), grad_beta.len());
    for (((dst, b), active), g) in out
        .iter_mut()
        .zip(current_beta.iter())
        .zip(body.iter())
        .zip(grad_beta.iter())
    {
        *dst = if *active {
            (b - scale * base_step * g).clamp(1.0, 12.0)
        } else {
            *b
        };
    }
}

fn max_body_abs(values: &[f64], body: &[bool]) -> f64 {
    values
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(value.abs()))
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_search_workspace_reuses_candidate_buffers_and_preserves_inactive_cells() {
        let mut workspace = LineSearchWorkspace::new(4);
        let initial_speed_capacity = workspace.candidate_speed.capacity();
        let initial_beta_capacity = workspace.candidate_beta.capacity();
        let current_speed = [1500.0, 1520.0, 1540.0, 1560.0];
        let background_speed = [1500.0, 1520.0, 1540.0, 1560.0];
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
}
