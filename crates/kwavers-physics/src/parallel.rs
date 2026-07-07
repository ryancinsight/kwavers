//! Atlas parallel-provider adapters for physics field traversal.

use moirai_parallel::{enumerate_mut_with, for_each_chunk_pair_mut_enumerated_with, Adaptive};
use ndarray::{ArrayView3, ArrayViewMut3, Zip};

const FIELD_CHUNK_SIZE: usize = 1024;

/// Apply an indexed mutation over a 3-D view.
#[inline]
pub(crate) fn for_each_indexed_mut<T, F>(mut values: ArrayViewMut3<'_, T>, f: F)
where
    T: Send,
    F: Fn((usize, usize, usize), &mut T) + Send + Sync,
{
    let (_nx, ny, nz) = values.dim();
    if let Some(slice) = values.as_slice_memory_order_mut() {
        let f_ref = &f;
        enumerate_mut_with::<Adaptive, _, _>(slice, |idx, value| {
            let plane = ny * nz;
            let i = idx / plane;
            let rem = idx % plane;
            f_ref((i, rem / nz, rem % nz), value);
        });
    } else {
        Zip::indexed(values).for_each(f);
    }
}

/// Apply an indexed mutation over paired 3-D views.
#[inline]
pub(crate) fn for_each_indexed_pair_mut<T, U, F>(
    mut values: ArrayViewMut3<'_, T>,
    input: ArrayView3<'_, U>,
    f: F,
) where
    T: Send,
    U: Sync,
    F: Fn((usize, usize, usize), &mut T, &U) + Send + Sync,
{
    assert_eq!(
        values.dim(),
        input.dim(),
        "invariant: physics paired traversal shape mismatch"
    );

    let (_nx, ny, nz) = values.dim();
    match (
        values.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
    ) {
        (Some(values), Some(input)) => {
            let f_ref = &f;
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                let plane = ny * nz;
                let i = idx / plane;
                let rem = idx % plane;
                f_ref((i, rem / nz, rem % nz), value, &input[idx]);
            });
        }
        _ => Zip::indexed(values).and(input).for_each(f),
    }
}

/// Apply an unindexed mutation over one mutable and one immutable 3-D view.
#[inline]
pub(crate) fn zip_mut_ref<T, U, F>(mut values: ArrayViewMut3<'_, T>, input: ArrayView3<'_, U>, f: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    assert_eq!(
        values.dim(),
        input.dim(),
        "invariant: physics zip input shape mismatch"
    );

    match (
        values.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
    ) {
        (Some(values), Some(input)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &input[idx]);
            });
        }
        _ => Zip::from(values).and(input).for_each(f),
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
        values.dim(),
        first.dim(),
        "invariant: physics zip first shape mismatch"
    );
    assert_eq!(
        values.dim(),
        second.dim(),
        "invariant: physics zip second shape mismatch"
    );

    match (
        values.as_slice_memory_order_mut(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &first[idx], &second[idx]);
            });
        }
        _ => Zip::from(values).and(first).and(second).for_each(f),
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
        first_out.dim(),
        second_out.dim(),
        "invariant: physics zip output shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        first.dim(),
        "invariant: physics zip first input shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        second.dim(),
        "invariant: physics zip second input shape mismatch"
    );

    match (
        first_out.as_slice_memory_order_mut(),
        second_out.as_slice_memory_order_mut(),
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
        _ => Zip::from(first_out)
            .and(second_out)
            .and(first)
            .and(second)
            .for_each(f),
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
        first_out.dim(),
        second_out.dim(),
        "invariant: physics zip output shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        first.dim(),
        "invariant: physics zip first input shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        second.dim(),
        "invariant: physics zip second input shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        third.dim(),
        "invariant: physics zip third input shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        fourth.dim(),
        "invariant: physics zip fourth input shape mismatch"
    );

    match (
        first_out.as_slice_memory_order_mut(),
        second_out.as_slice_memory_order_mut(),
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
        _ => Zip::from(first_out)
            .and(second_out)
            .and(first)
            .and(second)
            .and(third)
            .and(fourth)
            .for_each(f),
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
        values.dim(),
        first.dim(),
        "invariant: physics zip first shape mismatch"
    );
    assert_eq!(
        values.dim(),
        second.dim(),
        "invariant: physics zip second shape mismatch"
    );
    assert_eq!(
        values.dim(),
        third.dim(),
        "invariant: physics zip third shape mismatch"
    );

    match (
        values.as_slice_memory_order_mut(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
        third.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second), Some(third)) => {
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f(value, &first[idx], &second[idx], &third[idx]);
            });
        }
        _ => Zip::from(values)
            .and(first)
            .and(second)
            .and(third)
            .for_each(f),
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
        values.dim(),
        first.dim(),
        "invariant: physics zip first shape mismatch"
    );
    assert_eq!(
        values.dim(),
        second.dim(),
        "invariant: physics zip second shape mismatch"
    );
    assert_eq!(
        values.dim(),
        third.dim(),
        "invariant: physics zip third shape mismatch"
    );
    assert_eq!(
        values.dim(),
        fourth.dim(),
        "invariant: physics zip fourth shape mismatch"
    );

    match (
        values.as_slice_memory_order_mut(),
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
        _ => Zip::from(values)
            .and(first)
            .and(second)
            .and(third)
            .and(fourth)
            .for_each(f),
    }
}
