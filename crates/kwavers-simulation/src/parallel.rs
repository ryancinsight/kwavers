//! Provider-owned traversal adapters for simulation fields.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{ArrayView3, ArrayViewMut3, Zip};

const FIELD_CHUNK_SIZE: usize = 4096;

pub(crate) fn zip_indexed_mut_ref3<T, U, F>(
    mut out: ArrayViewMut3<'_, T>,
    input: ArrayView3<'_, U>,
    f: F,
) where
    T: Send,
    U: Sync,
    F: Fn((usize, usize, usize), &mut T, &U) + Send + Sync,
{
    assert_eq!(
        out.dim(),
        input.dim(),
        "invariant: indexed simulation traversal shapes must match"
    );

    let (_nx, ny, nz) = out.dim();
    match (out.as_slice_mut(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        let index = base + lane;
                        let i = index / (ny * nz);
                        let remainder = index % (ny * nz);
                        let j = remainder / nz;
                        let k = remainder % nz;
                        f_ref((i, j, k), value, &input[index]);
                    }
                },
            );
        }
        _ => Zip::indexed(out).and(input).for_each(f),
    }
}
