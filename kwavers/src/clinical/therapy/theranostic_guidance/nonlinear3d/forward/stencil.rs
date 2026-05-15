//! 3-D Westervelt finite-difference stencil and output recording.

use rayon::prelude::*;

pub(super) struct UpdateCells<'a> {
    pub(super) next: &'a mut [f64],
    pub(super) current: &'a [f64],
    pub(super) previous: &'a [f64],
    pub(super) older: &'a [f64],
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    pub(super) sponge: &'a [f64],
}

pub(super) fn update_cells(
    buffers: UpdateCells<'_>,
    n: usize,
    dt: f64,
    inv_dx2: f64,
    step: usize,
) {
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
            let nl =
                nonlinear_term_inline(center, prev, buffers.older[i], inv_dt, inv_dt2, step);
            let c = buffers.speed[i];
            let q = buffers.beta[i] * dt2 / (buffers.density[i] * c * c).max(1.0e-18);
            let raw = 2.0_f64.mul_add(center, -prev) + (c * dt).powi(2) * lap + q * nl;
            *dst = buffers.sponge[i] * raw;
        });
}

pub(super) fn record_receivers(
    traces: &mut [f64],
    receiver_cells: &[usize],
    next: &[f64],
    step: usize,
) {
    for (receiver, cell) in receiver_cells.iter().copied().enumerate() {
        traces[step * receiver_cells.len() + receiver] = next[cell];
    }
}

pub(super) fn update_peak(peak: &mut [f64], next: &[f64], source_mask: &[bool]) {
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
