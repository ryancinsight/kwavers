//! Provider-owned traversal adapters for application arrays.

use moirai_parallel::{
    for_each_chunk_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with, Adaptive,
};
use leto::{
    ArrayView,
    ArrayViewMut,
};

const FIELD_CHUNK_SIZE: usize = 4096;

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
        "invariant: paired traversal output shape must match input shape"
    );

    match (out.as_slice_mut(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &input[base + lane]);
                    }
                },
            );
        }
        _ => Zip::from(out).and(input).for_each(f),
    }
}

pub(crate) fn zip_two_mut_two_refs<T, U, V, W, D, F>(
    mut first_out: ArrayViewMut<'_, T, D>,
    mut second_out: ArrayViewMut<'_, U, D>,
    first: ArrayView<'_, V, D>,
    second: ArrayView<'_, W, D>,
    f: F,
) where
    D: Dimension,
    T: Send,
    U: Send,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &mut U, &V, &W) + Send + Sync,
{
    assert_eq!(
        first_out.dim(),
        second_out.dim(),
        "invariant: paired traversal output shapes must match"
    );
    assert_eq!(
        first_out.dim(),
        first.dim(),
        "invariant: paired traversal first input shape must match output shape"
    );
    assert_eq!(
        first_out.dim(),
        second.dim(),
        "invariant: paired traversal second input shape must match output shape"
    );

    match (
        first_out.as_slice_mut(),
        second_out.as_slice_mut(),
        first.as_slice(),
        second.as_slice(),
    ) {
        (Some(first_out), Some(second_out), Some(first), Some(second)) => {
            let f_ref = &f;
            for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                first_out,
                second_out,
                FIELD_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for lane in 0..first_chunk.len() {
                        f_ref(
                            &mut first_chunk[lane],
                            &mut second_chunk[lane],
                            &first[base + lane],
                            &second[base + lane],
                        );
                    }
                },
            );
        }
        _ => Zip::from(first_out)
            .and(second_out)
            .and(first)
            .and(second)
            .for_each(f),
    }
}
