//! Parallel zip adapters: one or more immutable views are traversed in lockstep
//! with one or more mutable views, with the closure receiving references.

use leto::{ArrayView3, ArrayViewMut3};
use moirai_parallel::{enumerate_mut_with, for_each_chunk_pair_mut_enumerated_with, Adaptive};

use super::FIELD_CHUNK_SIZE;

/// Apply an unindexed mutation over one mutable and one immutable 3-D view.
#[inline]
pub(crate) fn zip_mut_ref<T, U, F>(mut values: ArrayViewMut3<'_, T>, input: ArrayView3<'_, U>, f: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    assert_eq!(
        values.shape(),
        input.shape(),
        "invariant: physics zip input shape mismatch"
    );

    match (
        values.as_mut_slice_memory_order(),
        input.as_slice_memory_order(),
    ) {
        (Some(values), Some(input)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &input[idx]);
            });
        }
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    f(value, &input[[idx[0], idx[1], idx[2]]]);
                }
            }
        }
    }
}

/// Apply an unindexed mutation over one mutable and two immutable 3-D views.
#[inline]
pub(crate) fn zip_mut_two_refs<T, U, V, F>(
    mut values: ArrayViewMut3<'_, T>,
    first: ArrayView3<'_, U>,
    second: ArrayView3<'_, V>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    F: Fn(&mut T, &U, &V) + Send + Sync,
{
    assert_eq!(
        values.shape(),
        first.shape(),
        "invariant: physics zip first shape mismatch"
    );
    assert_eq!(
        values.shape(),
        second.shape(),
        "invariant: physics zip second shape mismatch"
    );

    match (
        values.as_mut_slice_memory_order(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &first[idx], &second[idx]);
            });
        }
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    f(
                        value,
                        &first[[idx[0], idx[1], idx[2]]],
                        &second[[idx[0], idx[1], idx[2]]],
                    );
                }
            }
        }
    }
}

/// Apply an unindexed mutation over one mutable and three immutable 3-D views.
#[inline]
pub(crate) fn zip_mut_three_refs<T, U, V, W, F>(
    mut values: ArrayViewMut3<'_, T>,
    first: ArrayView3<'_, U>,
    second: ArrayView3<'_, V>,
    third: ArrayView3<'_, W>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &U, &V, &W) + Send + Sync,
{
    assert_eq!(
        values.shape(),
        first.shape(),
        "invariant: physics zip first shape mismatch"
    );
    assert_eq!(
        values.shape(),
        second.shape(),
        "invariant: physics zip second shape mismatch"
    );
    assert_eq!(
        values.shape(),
        third.shape(),
        "invariant: physics zip third shape mismatch"
    );

    match (
        values.as_mut_slice_memory_order(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
        third.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second), Some(third)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &first[idx], &second[idx], &third[idx]);
            });
        }
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    let [i, j, k] = idx;
                    f(
                        value,
                        &first[[i, j, k]],
                        &second[[i, j, k]],
                        &third[[i, j, k]],
                    );
                }
            }
        }
    }
}

/// Apply an unindexed mutation over one mutable and four immutable 3-D views.
#[inline]
pub(crate) fn zip_mut_four_refs<T, U, V, W, X, F>(
    mut values: ArrayViewMut3<'_, T>,
    first: ArrayView3<'_, U>,
    second: ArrayView3<'_, V>,
    third: ArrayView3<'_, W>,
    fourth: ArrayView3<'_, X>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    X: Sync,
    F: Fn(&mut T, &U, &V, &W, &X) + Send + Sync,
{
    assert_eq!(
        values.shape(),
        first.shape(),
        "invariant: physics zip first shape mismatch"
    );
    assert_eq!(
        values.shape(),
        second.shape(),
        "invariant: physics zip second shape mismatch"
    );
    assert_eq!(
        values.shape(),
        third.shape(),
        "invariant: physics zip third shape mismatch"
    );
    assert_eq!(
        values.shape(),
        fourth.shape(),
        "invariant: physics zip fourth shape mismatch"
    );

    match (
        values.as_mut_slice_memory_order(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
        third.as_slice_memory_order(),
        fourth.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second), Some(third), Some(fourth)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &first[idx], &second[idx], &third[idx], &fourth[idx]);
            });
        }
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    let [i, j, k] = idx;
                    f(
                        value,
                        &first[[i, j, k]],
                        &second[[i, j, k]],
                        &third[[i, j, k]],
                        &fourth[[i, j, k]],
                    );
                }
            }
        }
    }
}

/// Apply an unindexed mutation over two mutable and two immutable 3-D views.
#[inline]
pub(crate) fn zip_two_mut_two_refs<T, U, V, W, F>(
    mut first_out: ArrayViewMut3<'_, T>,
    mut second_out: ArrayViewMut3<'_, U>,
    first: ArrayView3<'_, V>,
    second: ArrayView3<'_, W>,
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
        "invariant: physics zip output shape mismatch"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: physics zip first input shape mismatch"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: physics zip second input shape mismatch"
    );

    match (
        first_out.as_mut_slice_memory_order(),
        second_out.as_mut_slice_memory_order(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
    ) {
        (Some(first_out), Some(second_out), Some(first), Some(second)) => {
            for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                first_out,
                second_out,
                FIELD_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk| {
                    let start = chunk_index * FIELD_CHUNK_SIZE;
                    for (offset, (first_value, second_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .enumerate()
                    {
                        let idx = start + offset;
                        f(first_value, second_value, &first[idx], &second[idx]);
                    }
                },
            );
        }
        _ => {
            if let Ok(iter) = first_out.indexed_iter_mut() {
                for (idx, value) in iter {
                    let [i, j, k] = idx;
                    f(
                        value,
                        &mut second_out[[i, j, k]],
                        &first[[i, j, k]],
                        &second[[i, j, k]],
                    );
                }
            }
        }
    }
}

/// Apply an unindexed mutation over two mutable and four immutable 3-D views.
#[inline]
pub(crate) fn zip_two_mut_four_refs<T, U, V, W, X, Y, F>(
    mut first_out: ArrayViewMut3<'_, T>,
    mut second_out: ArrayViewMut3<'_, U>,
    first: ArrayView3<'_, V>,
    second: ArrayView3<'_, W>,
    third: ArrayView3<'_, X>,
    fourth: ArrayView3<'_, Y>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Sync,
    W: Sync,
    X: Sync,
    Y: Sync,
    F: Fn(&mut T, &mut U, &V, &W, &X, &Y) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: physics zip output shape mismatch"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: physics zip first input shape mismatch"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: physics zip second input shape mismatch"
    );
    assert_eq!(
        first_out.shape(),
        third.shape(),
        "invariant: physics zip third input shape mismatch"
    );
    assert_eq!(
        first_out.shape(),
        fourth.shape(),
        "invariant: physics zip fourth input shape mismatch"
    );

    match (
        first_out.as_mut_slice_memory_order(),
        second_out.as_mut_slice_memory_order(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
        third.as_slice_memory_order(),
        fourth.as_slice_memory_order(),
    ) {
        (
            Some(first_out),
            Some(second_out),
            Some(first),
            Some(second),
            Some(third),
            Some(fourth),
        ) => {
            for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                first_out,
                second_out,
                FIELD_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk| {
                    let start = chunk_index * FIELD_CHUNK_SIZE;
                    for (offset, (first_value, second_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .enumerate()
                    {
                        let idx = start + offset;
                        f(
                            first_value,
                            second_value,
                            &first[idx],
                            &second[idx],
                            &third[idx],
                            &fourth[idx],
                        );
                    }
                },
            );
        }
        _ => {
            if let Ok(iter) = first_out.indexed_iter_mut() {
                for (idx, value) in iter {
                    let [i, j, k] = idx;
                    f(
                        value,
                        &mut second_out[[i, j, k]],
                        &first[[i, j, k]],
                        &second[[i, j, k]],
                        &third[[i, j, k]],
                        &fourth[[i, j, k]],
                    );
                }
            }
        }
    }
}
