//! Provider-owned traversal adapters for math kernels.

use leto::{ArrayView, ArrayViewMut};
use leto_ops::zip_mut_with;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

const MATH_CHUNK_SIZE: usize = 4096;

pub(crate) fn zip_mut_ref<T, U, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    input: ArrayView<'_, U, N>,
    f: F,
) where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    assert_eq!(
        out.shape(),
        input.shape(),
        "invariant: math traversal output shape must match input shape"
    );

    match (out.as_mut_slice(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                MATH_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * MATH_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &input[base + lane]);
                    }
                },
            );
        }
        _ => zip_mut_with(&mut out, &input, f).unwrap(),
    }
}
