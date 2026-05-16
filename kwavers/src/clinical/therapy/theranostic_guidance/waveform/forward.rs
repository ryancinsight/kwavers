//! Forward propagation for the 2-D acoustic waveform simulation.

use ndarray::Array2;
use rayon::prelude::*;

use super::super::config::TheranosticInverseConfig;
use super::types::{AcousticGrid, CheckpointSchedule, WavefieldRun};
use super::utils::{linear, ricker};

pub(super) fn propagate(
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
    let mut psi_x = vec![0.0_f32; n];
    let mut psi_y = vec![0.0_f32; n];

    let schedule = CheckpointSchedule::new(grid.time_steps);
    let mut checkpoints: Option<Vec<f32>> =
        keep_history.then(|| vec![0.0_f32; schedule.slot_count() * n]);
    let source_scale =
        config.source_pressure_pa as f32 / (grid.source_cells.len().max(1) as f32).sqrt();

    for step in 0..grid.time_steps {
        if keep_history && schedule.is_checkpoint(step) {
            if let Some(ref mut buf) = checkpoints {
                let slot = schedule.slot_for(step);
                let range = slot * n..(slot + 1) * n;
                buf[range].copy_from_slice(&current);
            }
        }
        step_wavefield_cpml(
            grid, speed_m_s, &previous, &current, &mut next, &mut psi_x, &mut psi_y,
        );
        inject_sources(
            grid,
            config.frequencies_hz[0],
            source_scale,
            step,
            &mut next,
        );
        apply_attenuation(grid, &mut next);

        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            traces[step * grid.receiver_cells.len() + receiver] = current[cell];
        }
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        next.fill(0.0);
    }

    WavefieldRun {
        traces,
        checkpoints,
        checkpoint_interval: schedule.interval,
    }
}

/// Advance the pressure field by one time step using the 4th-order FD stencil
/// with CPML absorbing boundaries.
///
/// # Stencil
///
/// Fornberg (1988), Table 2, derivative order 2, stencil half-width 2:
/// `L₄[u]ᵢ = (-u[i-2] + 16u[i-1] - 30u[i] + 16u[i+1] - u[i+2]) / (12h²)`
///
/// # CPML update
///
/// Modified update equation (Komatitsch & Martin 2007, Eq. 14):
/// `p^{n+1} = 2p^n - p^{n-1} + dt²·c²·(Lx + Ly + ψx_ix + ψy_iy)`
pub(super) fn step_wavefield_cpml(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    previous: &[f32],
    current: &[f32],
    next: &mut [f32],
    psi_x: &mut [f32],
    psi_y: &mut [f32],
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx_m as f32;
    let dt = grid.dt_s as f32;
    let inv12dx2 = 1.0_f32 / (12.0 * dx * dx);
    let dt2 = dt * dt;

    for ix in 1..nx - 1 {
        let ax = grid.cpml.a_x[ix];
        let bx = grid.cpml.b_x[ix];
        if ax == 0.0 && bx == 1.0 {
            continue;
        }
        for iy in 0..ny {
            let idx = linear(ix, iy, ny);
            let dpx =
                (current[linear(ix + 1, iy, ny)] - current[linear(ix - 1, iy, ny)]) / (2.0 * dx);
            psi_x[idx] = bx * psi_x[idx] + ax * dpx;
        }
    }

    for ix in 0..nx {
        for iy in 1..ny - 1 {
            let ay = grid.cpml.a_y[iy];
            let by_ = grid.cpml.b_y[iy];
            if ay == 0.0 && by_ == 1.0 {
                continue;
            }
            let idx = linear(ix, iy, ny);
            let dpy =
                (current[linear(ix, iy + 1, ny)] - current[linear(ix, iy - 1, ny)]) / (2.0 * dx);
            psi_y[idx] = by_ * psi_y[idx] + ay * dpy;
        }
    }

    next.par_chunks_mut(ny)
        .enumerate()
        .skip(2)
        .take(nx.saturating_sub(4))
        .for_each(|(ix, row)| {
            for iy in 2..ny - 2 {
                let idx = linear(ix, iy, ny);
                let lx = (-current[linear(ix - 2, iy, ny)]
                    + 16.0 * current[linear(ix - 1, iy, ny)]
                    - 30.0 * current[idx]
                    + 16.0 * current[linear(ix + 1, iy, ny)]
                    - current[linear(ix + 2, iy, ny)])
                    * inv12dx2;
                let ly = (-current[linear(ix, iy - 2, ny)]
                    + 16.0 * current[linear(ix, iy - 1, ny)]
                    - 30.0 * current[idx]
                    + 16.0 * current[linear(ix, iy + 1, ny)]
                    - current[linear(ix, iy + 2, ny)])
                    * inv12dx2;
                let cpml_correction = psi_x[idx] + psi_y[idx];
                let c2 = speed_m_s[[ix, iy]].powi(2) as f32;
                row[iy] =
                    2.0 * current[idx] - previous[idx] + dt2 * c2 * (lx + ly + cpml_correction);
            }
        });
}

/// Apply per-cell fractional amplitude decay (Treeby & Cox 2010, §II.A).
///
/// `p^{n+1}[i,j] *= (1 - α_cell[i,j])`
#[inline]
pub(super) fn apply_attenuation(grid: &AcousticGrid, pressure: &mut [f32]) {
    for (p, &alpha) in pressure.iter_mut().zip(grid.alpha_np_per_step.iter()) {
        *p *= 1.0 - alpha;
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
