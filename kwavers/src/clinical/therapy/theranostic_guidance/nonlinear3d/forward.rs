//! Source-encoded 3-D Westervelt forward propagation.

use rayon::prelude::*;

use super::absorption::{AbsorptionBuilder, FractionalLaplacianAbsorption};
#[cfg(test)]
use super::checkpoint::HistorySegment;
use super::checkpoint::{ForwardHistory, HistoryReplayWorkspace};
use super::encoding::{focused_delay_s, max_source_focus_distance_m, SourceEncoding};
use super::stencil::sponge;
use super::types::{flat_index, Nonlinear3dAperture, Nonlinear3dConfig};

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
    pub(super) n: usize,
    pub(super) spacing_m: f64,
    pub(super) aperture: &'a Nonlinear3dAperture,
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) encoding: SourceEncoding,
    pub(super) retain_history: bool,
}

pub(super) struct ReplayInput<'a> {
    pub(super) history: &'a ForwardHistory,
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    pub(super) attenuation_np_per_m_mhz: Option<&'a [f64]>,
    pub(super) attenuation_power_law_y: Option<&'a [f64]>,
    pub(super) n: usize,
    pub(super) spacing_m: f64,
    pub(super) aperture: &'a Nonlinear3dAperture,
    pub(super) config: &'a Nonlinear3dConfig,
    pub(super) schedule: TimeSchedule,
    pub(super) encoding: SourceEncoding,
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
    );
    let mut source_mask = vec![false; cells];
    for cell in &source_plan.source_cells {
        source_mask[*cell] = true;
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
        aperture: input.aperture,
        config: input.config,
        schedule: input.schedule,
        encoding: input.encoding,
        spacing_m: input.spacing_m,
        max_focus_distance: source_plan.max_focus_distance,
        reference_speed: source_plan.reference_speed,
        weight_norm: source_plan.weight_norm,
    };
    let inv_dx2 = 1.0 / (input.spacing_m * input.spacing_m);
    for step in 0..steps {
        update_cells(
            UpdateCells {
                next: &mut next,
                current: &current,
                previous: &previous,
                older: &older,
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
        inject_sources(&mut next, &source_plan.source_cells, &drive, step);
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
    );
    let drive = DriveContext {
        aperture: input.aperture,
        config: input.config,
        schedule: input.schedule,
        encoding: input.encoding,
        spacing_m: input.spacing_m,
        max_focus_distance: source_plan.max_focus_distance,
        reference_speed: source_plan.reference_speed,
        weight_norm: source_plan.weight_norm,
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
                older: &older,
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
        inject_sources(&mut next, &source_plan.source_cells, &drive, step);
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
    );
    let drive = DriveContext {
        aperture: input.aperture,
        config: input.config,
        schedule: input.schedule,
        encoding: input.encoding,
        spacing_m: input.spacing_m,
        max_focus_distance: source_plan.max_focus_distance,
        reference_speed: source_plan.reference_speed,
        weight_norm: source_plan.weight_norm,
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
                older: &workspace.older,
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
        inject_sources(&mut workspace.next, &source_plan.source_cells, &drive, step);
        std::mem::swap(&mut workspace.older, &mut workspace.previous);
        std::mem::swap(&mut workspace.previous, &mut workspace.current);
        std::mem::swap(&mut workspace.current, &mut workspace.next);
        workspace.segment.set_state(step + 1, &workspace.current);
    }
}

struct SourcePlan {
    source_cells: Vec<usize>,
    max_focus_distance: f64,
    reference_speed: f64,
    weight_norm: f64,
}

fn build_source_plan(
    speed: &[f64],
    n: usize,
    spacing_m: f64,
    aperture: &Nonlinear3dAperture,
    encoding: SourceEncoding,
) -> SourcePlan {
    let source_cells = aperture
        .sources
        .iter()
        .map(|idx| flat_index(*idx, n))
        .collect::<Vec<_>>();
    let max_focus_distance =
        max_source_focus_distance_m(&aperture.sources, aperture.focus, spacing_m);
    let focus_cell = flat_index(aperture.focus, n);
    let reference_speed = speed[focus_cell].max(343.0);
    let weight_norm = aperture
        .sources
        .iter()
        .enumerate()
        .map(|(source, _)| encoding.source_weight(source, aperture.sources.len()).abs())
        .sum::<f64>()
        .max(1.0);
    SourcePlan {
        source_cells,
        max_focus_distance,
        reference_speed,
        weight_norm,
    }
}

struct UpdateCells<'a> {
    next: &'a mut [f64],
    current: &'a [f64],
    previous: &'a [f64],
    older: &'a [f64],
    speed: &'a [f64],
    density: &'a [f64],
    beta: &'a [f64],
    sponge: &'a [f64],
}

fn update_cells(buffers: UpdateCells<'_>, n: usize, dt: f64, inv_dx2: f64, step: usize) {
    let n2 = n * n;
    let dt2 = dt * dt;
    let inv_dt = 1.0 / dt;
    let inv_dt2 = 1.0 / dt2;
    buffers
        .next
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, dst)| {
            let z = i % n;
            let y = (i / n) % n;
            let x = i / n2;
            if x == 0 || y == 0 || z == 0 || x + 1 == n || y + 1 == n || z + 1 == n {
                *dst = 0.0;
                return;
            }
            let center = buffers.current[i];
            let prev = buffers.previous[i];
            let lap = (buffers.current[i - n2]
                + buffers.current[i + n2]
                + buffers.current[i - n]
                + buffers.current[i + n]
                + buffers.current[i - 1]
                + buffers.current[i + 1]
                - 6.0 * center)
                * inv_dx2;
            let nl = nonlinear_term_inline(center, prev, buffers.older[i], inv_dt, inv_dt2, step);
            let c = buffers.speed[i];
            let q = buffers.beta[i] * dt2 / (buffers.density[i] * c * c).max(1.0e-18);
            let raw = 2.0_f64.mul_add(center, -prev) + (c * dt).powi(2) * lap + q * nl;
            *dst = buffers.sponge[i] * raw;
        });
}

struct DriveContext<'a> {
    aperture: &'a Nonlinear3dAperture,
    config: &'a Nonlinear3dConfig,
    schedule: TimeSchedule,
    encoding: SourceEncoding,
    spacing_m: f64,
    max_focus_distance: f64,
    reference_speed: f64,
    weight_norm: f64,
}

fn inject_sources(next: &mut [f64], source_cells: &[usize], drive: &DriveContext<'_>, step: usize) {
    let time = step as f64 * drive.schedule.dt_s;
    for (source_idx, cell) in source_cells.iter().enumerate() {
        let delay = focused_delay_s(
            drive.aperture.sources[source_idx],
            drive.aperture.focus,
            drive.max_focus_distance,
            drive.spacing_m,
            drive.reference_speed,
        );
        let signal = source_signal(time - delay, drive.config)
            * drive
                .encoding
                .source_weight(source_idx, drive.aperture.sources.len())
            / drive.weight_norm;
        next[*cell] += signal;
    }
}

fn record_receivers(traces: &mut [f64], receiver_cells: &[usize], next: &[f64], step: usize) {
    for (receiver, cell) in receiver_cells.iter().copied().enumerate() {
        traces[step * receiver_cells.len() + receiver] = next[cell];
    }
}

fn update_peak(peak: &mut [f64], next: &[f64], source_mask: &[bool]) {
    for ((dst, value), is_source) in peak.iter_mut().zip(next.iter()).zip(source_mask.iter()) {
        if !*is_source {
            *dst = (*dst).max(value.abs());
        }
    }
}

#[inline(always)]
fn nonlinear_term_inline(
    center: f64,
    prev: f64,
    older: f64,
    inv_dt: f64,
    inv_dt2: f64,
    step: usize,
) -> f64 {
    let dp_dt = (center - prev) * inv_dt;
    if step >= 2 {
        let d2 = (center - 2.0 * prev + older) * inv_dt2;
        2.0 * center * d2 + 2.0 * dp_dt * dp_dt
    } else {
        2.0 * dp_dt * dp_dt
    }
}

fn source_signal(t: f64, config: &Nonlinear3dConfig) -> f64 {
    let duration = config.cycles / config.frequency_hz;
    if t < 0.0 || t >= duration {
        return 0.0;
    }
    let phase = 2.0 * std::f64::consts::PI * config.frequency_hz * t;
    let window = (std::f64::consts::PI * t / duration).sin().powi(2);
    config.source_pressure_pa * phase.sin() * window
}
