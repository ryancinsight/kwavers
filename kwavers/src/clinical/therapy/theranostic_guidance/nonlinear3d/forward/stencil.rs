//! 3-D Westervelt finite-difference stencil and output recording.

use rayon::prelude::*;

use super::super::stencil::westervelt_cell_terms;

pub(super) struct UpdateCells<'a> {
    pub(super) next: &'a mut [f64],
    pub(super) current: &'a [f64],
    pub(super) previous: &'a [f64],
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    pub(super) sponge: &'a [f64],
}

pub(super) fn update_cells(buffers: UpdateCells<'_>, n: usize, dt: f64, inv_dx2: f64, step: usize) {
    let n2 = n * n;
    let _ = step;
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
            let c = buffers.speed[i];
            let terms = westervelt_cell_terms(
                center,
                prev,
                lap,
                c,
                buffers.density[i],
                buffers.beta[i],
                dt,
            );
            let raw = 2.0_f64.mul_add(center, -prev) + terms.numerator / terms.denominator;
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
