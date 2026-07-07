//! Provider-owned traversal adapters for source masks.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{ArrayView, ArrayViewMut, Dimension, Zip};

const MASK_CHUNK_SIZE: usize = 4096;

pub(crate) fn zip_mut_ref<T, U, D, F>(
    mut out: ArrayViewMut<'_, T, D>,
    input: ArrayView<'_, U, D>,
    f: F,
) where
    D: Dimension,
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    assert_eq!(
        out.dim(),
        input.dim(),
        "invariant: source mask traversal shapes must match"
    );

    match (out.as_slice_mut(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                MASK_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * MASK_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &input[base + lane]);
                    }
                },
            );
        }
        _ => Zip::from(out).and(input).for_each(f),
    }
}
