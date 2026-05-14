//! Source-encoded time-domain receive simulation for same-device guidance.
//!
//! The solver in this module is deliberately separate from the reduced
//! finite-frequency normal-equation inverse. It integrates the two-dimensional
//! scalar acoustic wave equation on the CT-derived slice, records receiver
//! traces on the treatment/imaging aperture, and backpropagates the trace
//! residual to produce an adjoint RTM image. It is not nonlinear wave
//! propagation and it is not an iterative full-waveform inversion.

use ndarray::Array2;

use super::config::TheranosticInverseConfig;
use super::exposure::normalize_positive;
use super::geometry::{DeviceLayout, Point2};
use super::medium::PreparedTheranosticSlice;
use super::misfit::evaluate_trace_residual;

pub const THERANOSTIC_WAVEFORM_MODEL: &str = "source_encoded_time_domain_acoustic_adjoint_rtm";

#[derive(Clone, Debug)]
pub struct WaveformSimulationResult {
    pub reconstruction: Array2<f64>,
    pub residual_energy: f64,
    pub observed_energy: f64,
    pub receiver_count: usize,
    pub time_steps: usize,
    pub dt_s: f64,
    pub model_name: &'static str,
    pub misfit_name: &'static str,
    pub misfit_scale: f32,
    pub objective_value: f64,
}

#[derive(Clone, Debug)]
struct WavefieldRun {
    traces: Vec<f32>,
    history: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct AcousticGrid {
    nx: usize,
    ny: usize,
    dx_m: f64,
    dt_s: f64,
    time_steps: usize,
    source_cells: Vec<usize>,
    receiver_cells: Vec<usize>,
    source_delays_s: Vec<f64>,
    sponge: Vec<f32>,
}

/// Run source-encoded forward modeling and adjoint residual backpropagation.
///
/// # Mathematical contract
///
/// The forward model is the heterogeneous scalar acoustic equation
/// `p_tt = c(x)^2 Δp + s(x,t)` with a deterministic absorbing sponge. The
/// observed trace set is generated from `c_true = c0 + Δc lesion(x)`, the
/// predicted trace set is generated from `c0`, and the reconstruction is the
/// source-state/adjoint-state correlation over the CT body support.
#[must_use]
pub fn simulate_waveform_adjoint_rtm(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
    lesion: &Array2<f64>,
) -> WaveformSimulationResult {
    let true_speed = lesion_speed(prepared, config, lesion);
    let grid = acoustic_grid(
        prepared,
        layout,
        config,
        &prepared.sound_speed_m_s,
        &true_speed,
    );
    let observed = propagate(&grid, &true_speed, config, false);
    let predicted = propagate(&grid, &prepared.sound_speed_m_s, config, true);
    let residual = evaluate_trace_residual(
        &observed.traces,
        &predicted.traces,
        config.waveform_misfit,
        config.waveform_misfit_scale_fraction,
    );
    let residual_energy = energy(&residual.adjoint_source);
    let observed_energy = energy(&observed.traces);
    let forward_history = predicted
        .history
        .as_ref()
        .expect("forward history is requested for adjoint imaging");
    let image = adjoint_image(
        &grid,
        &prepared.sound_speed_m_s,
        &residual.adjoint_source,
        forward_history,
    );
    let reconstruction = normalize_positive(&image, &prepared.body_mask);
    WaveformSimulationResult {
        reconstruction,
        residual_energy,
        observed_energy,
        receiver_count: grid.receiver_cells.len(),
        time_steps: grid.time_steps,
        dt_s: grid.dt_s,
        model_name: THERANOSTIC_WAVEFORM_MODEL,
        misfit_name: residual.misfit.label(),
        misfit_scale: residual.scale,
        objective_value: residual.objective_value,
    }
}

fn acoustic_grid(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
    baseline_speed: &Array2<f64>,
    true_speed: &Array2<f64>,
) -> AcousticGrid {
    let (nx, ny) = prepared.sound_speed_m_s.dim();
    let (cmin, cmax) = speed_bounds(baseline_speed, true_speed);
    let dt_s = 0.35 * prepared.spacing_m / (std::f64::consts::SQRT_2 * cmax);
    let frequency_hz = config.frequencies_hz[0];
    let aperture_extent = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point.x_m.hypot(point.y_m))
        .fold(0.0, f64::max);
    let domain_extent = 0.5 * prepared.spacing_m * nx.max(ny) as f64;
    let travel_time_s = 2.0 * (aperture_extent + domain_extent) / cmin;
    let pulse_time_s = 5.0 / frequency_hz;
    let time_steps = (((travel_time_s + pulse_time_s) / dt_s).ceil() as usize).max(96);
    let delay_speed_m_s = reference_speed(prepared, baseline_speed);
    let focus = layout.focus_m;
    let source_distances = layout
        .therapy_elements
        .iter()
        .map(|source| distance(*source, focus))
        .collect::<Vec<_>>();
    let max_distance = source_distances
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let source_delays_s = source_distances
        .into_iter()
        .map(|distance_m| (max_distance - distance_m) / delay_speed_m_s)
        .collect();
    let source_cells = layout
        .therapy_elements
        .iter()
        .map(|point| point_to_cell(*point, nx, ny, prepared.spacing_m))
        .collect();
    let receiver_cells = layout
        .therapy_elements
        .iter()
        .chain(layout.imaging_receivers.iter())
        .map(|point| point_to_cell(*point, nx, ny, prepared.spacing_m))
        .collect();
    let sponge = sponge(nx, ny);
    AcousticGrid {
        nx,
        ny,
        dx_m: prepared.spacing_m,
        dt_s,
        time_steps,
        source_cells,
        receiver_cells,
        source_delays_s,
        sponge,
    }
}

fn propagate(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    config: &TheranosticInverseConfig,
    keep_history: bool,
) -> WavefieldRun {
    let n = grid.nx * grid.ny;
    let mut previous = vec![0.0_f32; n];
    let mut current = vec![0.0_f32; n];
    let mut next = vec![0.0_f32; n];
    let mut traces = vec![0.0_f32; grid.time_steps * grid.receiver_cells.len()];
    let mut history = keep_history.then(|| vec![0.0_f32; grid.time_steps * n]);
    let source_scale =
        config.source_pressure_pa as f32 / (grid.source_cells.len().max(1) as f32).sqrt();
    for step in 0..grid.time_steps {
        step_wavefield(grid, speed_m_s, &previous, &current, &mut next);
        inject_sources(
            grid,
            config.frequencies_hz[0],
            source_scale,
            step,
            &mut next,
        );
        apply_sponge(grid, &mut next);
        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            traces[step * grid.receiver_cells.len() + receiver] = current[cell];
        }
        if let Some(stored) = history.as_mut() {
            let range = step * n..(step + 1) * n;
            stored[range].copy_from_slice(&current);
        }
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        next.fill(0.0);
    }
    WavefieldRun { traces, history }
}

fn step_wavefield(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    previous: &[f32],
    current: &[f32],
    next: &mut [f32],
) {
    let coeff = (grid.dt_s / grid.dx_m).powi(2);
    for ix in 1..grid.nx - 1 {
        for iy in 1..grid.ny - 1 {
            let idx = linear(ix, iy, grid.ny);
            let lap = current[linear(ix + 1, iy, grid.ny)]
                + current[linear(ix - 1, iy, grid.ny)]
                + current[linear(ix, iy + 1, grid.ny)]
                + current[linear(ix, iy - 1, grid.ny)]
                - 4.0 * current[idx];
            let c2 = speed_m_s[[ix, iy]].powi(2) as f32;
            next[idx] = 2.0 * current[idx] - previous[idx] + (coeff as f32) * c2 * lap;
        }
    }
}

fn inject_sources(
    grid: &AcousticGrid,
    frequency_hz: f64,
    source_scale: f32,
    step: usize,
    pressure: &mut [f32],
) {
    let t = step as f64 * grid.dt_s;
    for (cell, delay_s) in grid.source_cells.iter().zip(grid.source_delays_s.iter()) {
        let tau = t - *delay_s;
        pressure[*cell] += source_scale * ricker(frequency_hz, tau) as f32;
    }
}

fn adjoint_image(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    residual: &[f32],
    forward_history: &[f32],
) -> Array2<f64> {
    let n = grid.nx * grid.ny;
    let mut previous = vec![0.0_f32; n];
    let mut current = vec![0.0_f32; n];
    let mut next = vec![0.0_f32; n];
    let mut image = vec![0.0_f64; n];
    let receiver_count = grid.receiver_cells.len();
    for reverse in 0..grid.time_steps {
        let step = grid.time_steps - 1 - reverse;
        step_wavefield(grid, speed_m_s, &previous, &current, &mut next);
        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            next[cell] += residual[step * receiver_count + receiver];
        }
        apply_sponge(grid, &mut next);
        let forward = &forward_history[step * n..(step + 1) * n];
        for (idx, value) in image.iter_mut().enumerate() {
            *value += forward[idx] as f64 * current[idx] as f64;
        }
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        next.fill(0.0);
    }
    Array2::from_shape_fn((grid.nx, grid.ny), |(ix, iy)| {
        image[linear(ix, iy, grid.ny)].abs()
    })
}

fn lesion_speed(
    prepared: &PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
    lesion: &Array2<f64>,
) -> Array2<f64> {
    Array2::from_shape_fn(prepared.sound_speed_m_s.dim(), |idx| {
        if prepared.body_mask[idx] {
            (prepared.sound_speed_m_s[idx] + config.lesion_delta_c_m_s * lesion[idx])
                .clamp(1000.0, 3500.0)
        } else {
            prepared.sound_speed_m_s[idx]
        }
    })
}

fn speed_bounds(baseline_speed: &Array2<f64>, true_speed: &Array2<f64>) -> (f64, f64) {
    let (cmin, cmax) = baseline_speed
        .iter()
        .chain(true_speed.iter())
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold((f64::INFINITY, 0.0_f64), |(cmin, cmax), value| {
            (cmin.min(value), cmax.max(value))
        });
    let cmin = if cmin.is_finite() { cmin.max(1.0) } else { 1.0 };
    (cmin, cmax.max(1.0))
}

fn reference_speed(prepared: &PreparedTheranosticSlice, baseline_speed: &Array2<f64>) -> f64 {
    let (sum, count) = baseline_speed
        .indexed_iter()
        .filter(|(idx, value)| prepared.body_mask[*idx] && value.is_finite() && **value > 0.0)
        .fold((0.0, 0usize), |(sum, count), (_, value)| {
            (sum + *value, count + 1)
        });
    if count == 0 {
        let (sum, count) = baseline_speed
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .fold((0.0, 0usize), |(sum, count), value| {
                (sum + value, count + 1)
            });
        return (sum / count.max(1) as f64).max(1.0);
    }
    (sum / count as f64).max(1.0)
}

fn apply_sponge(grid: &AcousticGrid, pressure: &mut [f32]) {
    for (value, damping) in pressure.iter_mut().zip(grid.sponge.iter()) {
        *value *= 1.0 - *damping;
    }
}

fn sponge(nx: usize, ny: usize) -> Vec<f32> {
    let width = nx.min(ny).clamp(8, 24) / 2;
    let mut out = vec![0.0_f32; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            let edge = ix.min(iy).min(nx - 1 - ix).min(ny - 1 - iy);
            if edge < width {
                let x = (width - edge) as f32 / width as f32;
                out[linear(ix, iy, ny)] = 0.18 * x * x;
            }
        }
    }
    out
}

fn point_to_cell(point: Point2, nx: usize, ny: usize, spacing_m: f64) -> usize {
    let cx = 0.5 * (nx - 1) as f64;
    let cy = 0.5 * (ny - 1) as f64;
    let ix = (point.x_m / spacing_m + cx)
        .round()
        .clamp(1.0, (nx - 2) as f64) as usize;
    let iy = (point.y_m / spacing_m + cy)
        .round()
        .clamp(1.0, (ny - 2) as f64) as usize;
    linear(ix, iy, ny)
}

fn ricker(frequency_hz: f64, tau_s: f64) -> f64 {
    let x = std::f64::consts::PI * frequency_hz * (tau_s - 2.0 / frequency_hz);
    (1.0 - 2.0 * x * x) * (-x * x).exp()
}

fn energy(values: &[f32]) -> f64 {
    values.iter().map(|value| (*value as f64).powi(2)).sum()
}

fn distance(a: Point2, b: Point2) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}

const fn linear(ix: usize, iy: usize, ny: usize) -> usize {
    ix * ny + iy
}
