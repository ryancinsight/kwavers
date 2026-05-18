//! Source-encoded 3-D Westervelt forward propagation.

mod source;
mod stencil;

use super::absorption::{AbsorptionBuilder, FractionalLaplacianAbsorption};
#[cfg(test)]
use super::checkpoint::HistorySegment;
use super::checkpoint::{ForwardHistory, HistoryReplayWorkspace};
use super::encoding::SourceEncoding;
use super::stencil::sponge;
use super::types::{flat_index, Nonlinear3dAperture, Nonlinear3dConfig};

pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d) use source::source_plan_metrics;
use source::{build_source_plan, inject_sources, source_cells, DriveContext};
use stencil::{record_receivers, update_cells, update_peak, UpdateCells};

#[derive(Clone, Debug)]
pub(super) struct ForwardResult {
    pub traces: Vec<f64>,
    pub peak_pressure: Vec<f64>,
    pub history: Option<ForwardHistory>,
    pub dt_s: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TimeSchedule {
    pub(super) dt_s: f64,
    pub(super) time_steps: usize,
}

pub(super) struct ForwardInput<'a> {
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    /// Per-voxel `α₀` at 1 MHz in Np/m. When `None` (or identically zero
    /// after building) the forward propagates with no power-law absorption
    /// — the loss-free baseline.
    pub(super) attenuation_np_per_m_mhz: Option<&'a [f64]>,
    /// Per-voxel power-law exponent `y`. Required when
    /// `attenuation_np_per_m_mhz` is `Some` and not all zero.
    pub(super) attenuation_power_law_y: Option<&'a [f64]>,
    pub(super) source_body_mask: Option<&'a [bool]>,
    pub(super) n: usize,
    pub(super) spacing_m: f64,
    pub(super) aperture: &'a Nonlinear3dAperture,
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) encoding: SourceEncoding,
    pub(super) source_scale: f64,
    pub(super) retain_history: bool,
}

pub(super) struct ReplayInput<'a> {
    pub(super) history: &'a ForwardHistory,
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    pub(super) attenuation_np_per_m_mhz: Option<&'a [f64]>,
    pub(super) attenuation_power_law_y: Option<&'a [f64]>,
    pub(super) source_body_mask: Option<&'a [bool]>,
    pub(super) n: usize,
    pub(super) spacing_m: f64,
    pub(super) aperture: &'a Nonlinear3dAperture,
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) encoding: SourceEncoding,
    pub(super) source_scale: f64,
    pub(super) sponge: &'a [f64],
    pub(super) step: usize,
}

pub(super) fn time_schedule(
    speed: &[f64],
    n: usize,
    spacing_m: f64,
    config: &Nonlinear3dConfig,
) -> TimeSchedule {
    let c_max = speed.iter().copied().fold(0.0, f64::max).max(343.0);
    let c_min = speed
        .iter()
        .copied()
        .filter(|value| *value > 100.0)
        .fold(f64::INFINITY, f64::min)
        .max(343.0);
    let dt = config.cfl * spacing_m / (c_max * 3.0_f64.sqrt());
    let diagonal = 3.0_f64.sqrt() * (n - 1) as f64 * spacing_m;
    let duration = 2.0 * diagonal / c_min + config.cycles / config.frequency_hz;
    TimeSchedule {
        dt_s: dt,
        time_steps: (duration / dt).ceil().max(24.0) as usize,
    }
}

pub(super) fn forward_with_schedule(input: ForwardInput<'_>) -> ForwardResult {
    let cells = input.n * input.n * input.n;
    let dt = input.schedule.dt_s;
    let steps = input.schedule.time_steps;
    let source_plan = build_source_plan(
        input.speed,
        input.n,
        input.spacing_m,
        input.aperture,
        input.encoding,
        input.source_body_mask,
    );
    let mut source_mask = vec![false; cells];
    for cell in source_cells(&source_plan) {
        source_mask[cell] = true;
    }
    let receiver_cells = input
        .aperture
        .receivers
        .iter()
        .map(|idx| flat_index(*idx, input.n))
        .collect::<Vec<_>>();
    let sponge = sponge(input.n);
    let mut absorption = build_absorption(
        input.n,
        input.spacing_m,
        dt,
        input.speed,
        input.attenuation_np_per_m_mhz,
        input.attenuation_power_law_y,
    );
    let mut older = vec![0.0_f64; cells];
    let mut previous = vec![0.0_f64; cells];
    let mut current = vec![0.0_f64; cells];
    let mut next = vec![0.0_f64; cells];
    let mut peak = vec![0.0_f64; cells];
    let mut traces = vec![0.0; steps * receiver_cells.len()];
    let mut history = input.retain_history.then(|| {
        ForwardHistory::new(
            cells,
            steps,
            input.config.checkpoint_interval_steps,
            &older,
            &previous,
            &current,
        )
    });
    let drive = DriveContext {
        config: input.config,
        schedule: input.schedule,
        source_scale: input.source_scale,
    };
    let inv_dx2 = 1.0 / (input.spacing_m * input.spacing_m);
    for step in 0..steps {
        update_cells(
            UpdateCells {
                next: &mut next,
                current: &current,
                previous: &previous,
                speed: input.speed,
                density: input.density,
                beta: input.beta,
                sponge: &sponge,
            },
            input.n,
            dt,
            inv_dx2,
            step,
        );
        if let Some(op) = absorption.as_mut() {
            op.apply(&current, &previous, &mut next);
        }
        inject_sources(&mut next, &source_plan, &drive, step);
        record_receivers(&mut traces, &receiver_cells, &next, step);
        update_peak(&mut peak, &next, &source_mask);
        std::mem::swap(&mut older, &mut previous);
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        if let Some(stored) = history.as_mut() {
            stored.store_if_boundary(step + 1, &older, &previous, &current);
        }
    }
    ForwardResult {
        traces,
        peak_pressure: peak,
        history,
        dt_s: dt,
    }
}

/// Construct a fractional-Laplacian absorption operator from per-voxel
/// material arrays. Returns `None` when no attenuation array is provided
/// or when the requested attenuation is identically zero — both cases
/// short-circuit the forward and adjoint paths to the loss-free baseline.
fn build_absorption(
    n: usize,
    spacing_m: f64,
    dt_s: f64,
    speed_m_s: &[f64],
    attenuation_np_per_m_mhz: Option<&[f64]>,
    attenuation_power_law_y: Option<&[f64]>,
) -> Option<FractionalLaplacianAbsorption> {
    let alpha = attenuation_np_per_m_mhz?;
    let y_field = attenuation_power_law_y?;
    FractionalLaplacianAbsorption::maybe_new(AbsorptionBuilder {
        n,
        spacing_m,
        dt_s,
        speed_m_s,
        attenuation_np_per_m_mhz: alpha,
        attenuation_power_law_y: y_field,
    })
}

#[cfg(test)]
pub(super) fn forward_dense_history_for_test(input: ForwardInput<'_>) -> Vec<f64> {
    let cells = input.n * input.n * input.n;
    let steps = input.schedule.time_steps;
    let source_plan = build_source_plan(
        input.speed,
        input.n,
        input.spacing_m,
        input.aperture,
        input.encoding,
        input.source_body_mask,
    );
    let drive = DriveContext {
        config: input.config,
        schedule: input.schedule,
        source_scale: input.source_scale,
    };
    let sponge = sponge(input.n);
    let mut absorption = build_absorption(
        input.n,
        input.spacing_m,
        input.schedule.dt_s,
        input.speed,
        input.attenuation_np_per_m_mhz,
        input.attenuation_power_law_y,
    );
    let mut older = vec![0.0_f64; cells];
    let mut previous = vec![0.0_f64; cells];
    let mut current = vec![0.0_f64; cells];
    let mut next = vec![0.0_f64; cells];
    let mut history = vec![0.0_f64; (steps + 1) * cells];
    let inv_dx2 = 1.0 / (input.spacing_m * input.spacing_m);
    for step in 0..steps {
        update_cells(
            UpdateCells {
                next: &mut next,
                current: &current,
                previous: &previous,
                speed: input.speed,
                density: input.density,
                beta: input.beta,
                sponge: &sponge,
            },
            input.n,
            input.schedule.dt_s,
            inv_dx2,
            step,
        );
        if let Some(op) = absorption.as_mut() {
            op.apply(&current, &previous, &mut next);
        }
        inject_sources(&mut next, &source_plan, &drive, step);
        std::mem::swap(&mut older, &mut previous);
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        let offset = (step + 1) * cells;
        history[offset..offset + cells].copy_from_slice(&current);
    }
    history
}

#[cfg(test)]
pub(super) fn replay_history_segment(input: ReplayInput<'_>) -> HistorySegment {
    let mut workspace =
        HistoryReplayWorkspace::new(input.history.cells(), input.history.interval());
    replay_history_segment_into(input, &mut workspace);
    workspace.segment().clone()
}

pub(super) fn replay_history_segment_into(
    input: ReplayInput<'_>,
    workspace: &mut HistoryReplayWorkspace,
) {
    debug_assert_eq!(input.history.cells(), input.n * input.n * input.n);
    debug_assert!(input.step < input.history.steps());
    let (start_step, end_step) = input.history.segment_bounds_for_step(input.step);
    let checkpoint = input.history.checkpoint_at(start_step);
    workspace.reset_from_checkpoint(start_step, end_step, checkpoint);
    let source_plan = build_source_plan(
        input.speed,
        input.n,
        input.spacing_m,
        input.aperture,
        input.encoding,
        input.source_body_mask,
    );
    let drive = DriveContext {
        config: input.config,
        schedule: input.schedule,
        source_scale: input.source_scale,
    };
    // Reproduce the lossy forward exactly so the replayed pressure states
    // match the originals bit-for-bit when absorption is active. The
    // operator's `prev_l_y` cache is local to this replay; the forward's
    // separate cache is unaffected.
    let mut absorption = build_absorption(
        input.n,
        input.spacing_m,
        input.schedule.dt_s,
        input.speed,
        input.attenuation_np_per_m_mhz,
        input.attenuation_power_law_y,
    );
    let inv_dx2 = 1.0 / (input.spacing_m * input.spacing_m);
    for step in start_step..end_step {
        update_cells(
            UpdateCells {
                next: &mut workspace.next,
                current: &workspace.current,
                previous: &workspace.previous,
                speed: input.speed,
                density: input.density,
                beta: input.beta,
                sponge: input.sponge,
            },
            input.n,
            input.schedule.dt_s,
            inv_dx2,
            step,
        );
        if let Some(op) = absorption.as_mut() {
            op.apply(&workspace.current, &workspace.previous, &mut workspace.next);
        }
        inject_sources(&mut workspace.next, &source_plan, &drive, step);
        std::mem::swap(&mut workspace.older, &mut workspace.previous);
        std::mem::swap(&mut workspace.previous, &mut workspace.current);
        std::mem::swap(&mut workspace.current, &mut workspace.next);
        workspace.segment.set_state(step + 1, &workspace.current);
    }
}
