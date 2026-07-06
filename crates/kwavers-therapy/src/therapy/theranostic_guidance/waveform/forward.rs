//! Forward propagation for the 2-D acoustic waveform simulation.

use moirai_parallel::{
    enumerate_mut_with, for_each_chunk_mut_enumerated_with,
    for_each_chunk_pair_mut_enumerated_with, Adaptive,
};
use ndarray::Array2;

use super::types::{AcousticGrid, CheckpointSchedule, WavefieldRun};
use super::utils::{linear, ricker};

const WAVEFORM_FIELD_CHUNK: usize = 1024;

pub(super) fn propagate(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    keep_history: bool,
) -> WavefieldRun {
    let n = grid.nx * grid.ny;
    let mut previous = vec![0.0_f32; n];
    let mut current = vec![0.0_f32; n];
    let mut next = vec![0.0_f32; n];
    let mut traces = vec![0.0_f32; grid.time_steps * grid.receiver_cells.len()];
    let mut psi_x = vec![0.0_f32; n];
    let mut psi_y = vec![0.0_f32; n];

    let schedule = CheckpointSchedule::new(grid.time_steps, n);
    // Each slot stores a consecutive pair (previous, current) so the adjoint
    // replay can initialise the second-order scheme without the O(|p|)
    // zero-velocity error that arises when both are set to the same snapshot.
    // Buffer layout: [prev₀ | curr₀ | prev₁ | curr₁ | …], size = 2·slots·N.
    let mut checkpoints: Option<Vec<f32>> =
        keep_history.then(|| vec![0.0_f32; schedule.slot_count() * 2 * n]);

    let c2dt2 = c2dt2_field(grid, speed_m_s);

    for step in 0..grid.time_steps {
        if keep_history && schedule.is_checkpoint(step) {
            if let Some(ref mut buf) = checkpoints {
                let slot = schedule.slot_for(step);
                let base = slot * 2 * n;
                // Save the pair (prev, curr) at this checkpoint time.
                buf[base..base + n].copy_from_slice(&previous);
                buf[base + n..base + 2 * n].copy_from_slice(&current);
            }
        }
        step_wavefield_cpml(
            grid, &c2dt2, &previous, &current, &mut next, &mut psi_x, &mut psi_y,
        );
        inject_sources(grid, step, &mut next);
        apply_attenuation(grid, &mut next);

        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            traces[step * grid.receiver_cells.len() + receiver] = current[cell];
        }
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        clear_fd_halo(&mut next, grid.nx, grid.ny);
    }

    WavefieldRun {
        traces,
        checkpoints,
        checkpoint_interval: schedule.interval,
    }
}

pub(super) fn propagate_peak_pressure(grid: &AcousticGrid, speed_m_s: &Array2<f64>) -> Vec<f32> {
    let n = grid.nx * grid.ny;
    let mut previous = vec![0.0_f32; n];
    let mut current = vec![0.0_f32; n];
    let mut next = vec![0.0_f32; n];
    let mut psi_x = vec![0.0_f32; n];
    let mut psi_y = vec![0.0_f32; n];
    let mut peak = vec![0.0_f32; n];

    let c2dt2 = c2dt2_field(grid, speed_m_s);

    for step in 0..grid.time_steps {
        step_wavefield_cpml(
            grid, &c2dt2, &previous, &current, &mut next, &mut psi_x, &mut psi_y,
        );
        inject_sources(grid, step, &mut next);
        apply_attenuation_and_update_peak(grid, &mut next, &mut peak);
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        clear_fd_halo(&mut next, grid.nx, grid.ny);
    }

    peak
}

#[inline]
pub(super) const fn peak_pressure_workspace_values(nx: usize, ny: usize) -> usize {
    6 * nx * ny
}

/// Precompute the loop-invariant stencil coefficient `c²·dt²` (f32) in
/// `linear(ix, iy, ny)` index order.
///
/// The speed field is constant across the entire time loop, so `c²·dt²`
/// is loop-invariant. Hoisting it out of [`step_wavefield_cpml`] removes,
/// per cell per timestep, an `f64` `powi(2)`, an `f64 → f32` cast, and the
/// strided `Array2<f64>` element access, replacing them with a single
/// contiguous `f32` slice load (half the memory traffic of the `f64`
/// field). The per-cell value is computed identically to the previous
/// inline form (`(dt as f32)² * (c.powi(2) as f32)`), so the stencil
/// result is bit-for-bit unchanged.
pub(super) fn c2dt2_field(grid: &AcousticGrid, speed_m_s: &Array2<f64>) -> Vec<f32> {
    let nx = grid.nx;
    let ny = grid.ny;
    let dt = grid.dt_s as f32;
    let dt2 = dt * dt;
    let mut field = vec![0.0_f32; nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            let c2 = speed_m_s[[ix, iy]].powi(2) as f32;
            field[linear(ix, iy, ny)] = dt2 * c2;
        }
    }
    field
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
    c2dt2: &[f32],
    previous: &[f32],
    current: &[f32],
    next: &mut [f32],
    psi_x: &mut [f32],
    psi_y: &mut [f32],
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx_m as f32;
    let inv12dx2 = 1.0_f32 / (12.0 * dx * dx);

    // CPML auxiliary memory-variable updates. Each cell's `psi` update reads
    // only its own prior `psi` value and the read-only `current` neighbours, so
    // rows are independent and the per-cell arithmetic is identical to the
    // sequential form — parallelising over rows is bit-exact.
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(psi_x, ny, |ix, psi_row| {
        if ix == 0 || ix + 1 >= nx {
            return;
        }
        let ax = grid.cpml.a_x[ix];
        let bx = grid.cpml.b_x[ix];
        if ax == 0.0 && bx == 1.0 {
            return;
        }
        for iy in 0..ny {
            let dpx =
                (current[linear(ix + 1, iy, ny)] - current[linear(ix - 1, iy, ny)]) / (2.0 * dx);
            psi_row[iy] = bx * psi_row[iy] + ax * dpx;
        }
    });

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(psi_y, ny, |ix, psi_row| {
        for iy in 1..ny - 1 {
            let ay = grid.cpml.a_y[iy];
            let by_ = grid.cpml.b_y[iy];
            if ay == 0.0 && by_ == 1.0 {
                continue;
            }
            let dpy =
                (current[linear(ix, iy + 1, ny)] - current[linear(ix, iy - 1, ny)]) / (2.0 * dx);
            psi_row[iy] = by_ * psi_row[iy] + ay * dpy;
        }
    });

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(next, ny, |ix, row| {
        if ix < 2 || ix + 2 >= nx {
            return;
        }
        for iy in 2..ny - 2 {
            let idx = linear(ix, iy, ny);
            let lx = (-current[linear(ix - 2, iy, ny)] + 16.0 * current[linear(ix - 1, iy, ny)]
                - 30.0 * current[idx]
                + 16.0 * current[linear(ix + 1, iy, ny)]
                - current[linear(ix + 2, iy, ny)])
                * inv12dx2;
            let ly = (-current[linear(ix, iy - 2, ny)] + 16.0 * current[linear(ix, iy - 1, ny)]
                - 30.0 * current[idx]
                + 16.0 * current[linear(ix, iy + 1, ny)]
                - current[linear(ix, iy + 2, ny)])
                * inv12dx2;
            let cpml_correction = psi_x[idx] + psi_y[idx];
            row[iy] = 2.0 * current[idx] - previous[idx] + c2dt2[idx] * (lx + ly + cpml_correction);
        }
    });
}

/// Apply per-cell fractional amplitude decay (Treeby & Cox 2010, §II.A).
///
/// `p^{n+1}[i,j] *= (1 - α_cell[i,j])`
#[inline]
pub(super) fn apply_attenuation(grid: &AcousticGrid, pressure: &mut [f32]) {
    enumerate_mut_with::<Adaptive, _, _>(pressure, |idx, p| {
        *p *= 1.0 - grid.alpha_np_per_step[idx];
    });
}

#[inline]
pub(super) fn apply_attenuation_and_update_peak(
    grid: &AcousticGrid,
    pressure: &mut [f32],
    peak: &mut [f32],
) {
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        pressure,
        peak,
        WAVEFORM_FIELD_CHUNK,
        |chunk_idx, pressure_chunk, peak_chunk| {
            let offset = chunk_idx * WAVEFORM_FIELD_CHUNK;
            for (local_idx, (p, peak_value)) in pressure_chunk
                .iter_mut()
                .zip(peak_chunk.iter_mut())
                .enumerate()
            {
                let alpha = grid.alpha_np_per_step[offset + local_idx];
                *p *= 1.0 - alpha;
                *peak_value = (*peak_value).max((*p).abs());
            }
        },
    );
}

/// Inject the Ricker-wavelet pressure source at all source cells.
///
/// Uses `grid.source_frequency_hz` and `grid.source_scale` so that the
/// adjoint replay path can call the same function without extra parameters.
///
/// # Correctness contract
///
/// This function must be called in the adjoint replay loop with the same
/// `step` index as in the forward pass. Any discrepancy introduces a
/// systematic phase error in the imaging condition.
pub(super) fn inject_sources(grid: &AcousticGrid, step: usize, pressure: &mut [f32]) {
    // Broadband passive-emission path: inject a precomputed waveform
    // simultaneously (zero delay) at every source cell.
    if let Some(waveform) = grid.source_waveform.as_ref() {
        let amplitude = grid.source_scale * waveform.get(step).copied().unwrap_or(0.0);
        for &cell in &grid.source_cells {
            pressure[cell] += amplitude;
        }
        return;
    }
    // Default focused-transmit path: per-cell Ricker wavelet with focal-law delays.
    let t = step as f64 * grid.dt_s;
    for (cell, delay_s) in grid.source_cells.iter().zip(grid.source_delays_s.iter()) {
        let tau = t - *delay_s;
        pressure[*cell] += grid.source_scale * ricker(grid.source_frequency_hz, tau) as f32;
    }
}

#[inline]
fn clear_fd_halo(field: &mut [f32], nx: usize, ny: usize) {
    if nx == 0 || ny == 0 {
        return;
    }
    for ix in 0..nx.min(2) {
        let start = ix * ny;
        field[start..start + ny].fill(0.0);
    }
    for ix in nx.saturating_sub(2)..nx {
        let start = ix * ny;
        field[start..start + ny].fill(0.0);
    }
    let interior_start = nx.min(2);
    let interior_end = nx.saturating_sub(2);
    if ny <= 4 || interior_start >= interior_end {
        return;
    }
    for ix in interior_start..interior_end {
        let row = ix * ny;
        field[row] = 0.0;
        field[row + 1] = 0.0;
        field[row + ny - 2] = 0.0;
        field[row + ny - 1] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::clear_fd_halo;
    use crate::therapy::theranostic_guidance::waveform::utils::linear;

    #[test]
    fn clear_fd_halo_preserves_interior_cells_only() {
        let nx = 6;
        let ny = 7;
        let mut field = vec![1.0_f32; nx * ny];

        clear_fd_halo(&mut field, nx, ny);

        for ix in 0..nx {
            for iy in 0..ny {
                let expected = if (2..nx - 2).contains(&ix) && (2..ny - 2).contains(&iy) {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(field[linear(ix, iy, ny)], expected);
            }
        }
    }
}
