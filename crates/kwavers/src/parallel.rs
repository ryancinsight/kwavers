//! Provider-owned traversal adapters for application arrays.

use moirai_parallel::{
    for_each_chunk_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with, Adaptive,
};
use leto::{ArrayView, ArrayViewMut};

const FIELD_CHUNK_SIZE: usize = 4096;

/// Row-major odometer increment of a logical multi-index; returns `false` once
/// the index wraps past the final element. Drives the strided fallback walks.
#[inline]
fn next_index<const N: usize>(index: &mut [usize; N], shape: &[usize; N]) -> bool {
    for d in (0..N).rev() {
        index[d] += 1;
        if index[d] < shape[d] {
            return true;
        }
        index[d] = 0;
    }
    false
}

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
        "invariant: paired traversal output shape must match input shape"
    );

    match (out.as_mut_slice(), input.as_slice()) {
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
        _ => {
            let shape = out.shape();
            let mut index = [0usize; N];
            for _ in 0..out.size() {
                let value = out.get_mut(index).expect("invariant: index in bounds");
                f(value, input.get(index).expect("invariant: index in bounds"));
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_two_mut_two_refs<T, U, V, W, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    first: ArrayView<'_, V, N>,
    second: ArrayView<'_, W, N>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &mut U, &V, &W) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: paired traversal output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: paired traversal first input shape must match output shape"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: paired traversal second input shape must match output shape"
    );

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
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
                    for (lane, (first_value, second_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .enumerate()
                    {
                        let index = base + lane;
                        f_ref(first_value, second_value, &first[index], &second[index]);
                    }
                },
            );
        }
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out.get_mut(index).expect("invariant: index in bounds");
                let second_value = second_out.get_mut(index).expect("invariant: index in bounds");
                f(
                    first_value,
                    second_value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}
