//! Shared Moirai-backed traversal for finite-difference output buffers.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use leto::Array3;

const DIFFERENTIAL_CHUNK_LEN: usize = 4096;

pub(super) fn try_fill_standard_layout<F>(dst: &mut Array3<f64>, value_at: F) -> bool
where
    F: Fn(usize, usize, usize) -> f64 + Send + Sync + Copy,
{
    if dst.as_slice_memory_order_mut().is_none() {
        return false;
    }

    let [_nx, ny, nz] = dst.shape();
    let Some(values) = dst.as_slice_mut() else {
        return false;
    };

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        values,
        DIFFERENTIAL_CHUNK_LEN,
        move |chunk_index, chunk| {
            let base = chunk_index * DIFFERENTIAL_CHUNK_LEN;
            for (offset, value) in chunk.iter_mut().enumerate() {
                let linear = base + offset;
                let plane = ny * nz;
                let i = linear / plane;
                let rem = linear % plane;
                let j = rem / nz;
                let k = rem % nz;
                *value = value_at(i, j, k);
            }
        },
    );
    true
}

pub(super) const fn row_major_index(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    (i * ny + j) * nz + k
}
