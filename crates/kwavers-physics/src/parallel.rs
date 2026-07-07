//! Atlas parallel-provider adapters for physics field traversal.

use moirai_parallel::{
    enumerate_mut_with, for_each_chunk_pair_mut_enumerated_with,
    for_each_chunk_quad_mut_enumerated_with, for_each_chunk_triple_mut_enumerated_with, Adaptive,
};
use ndarray::{ArrayView3, ArrayViewMut3, Zip};

const FIELD_CHUNK_SIZE: usize = 1024;

#[inline]
fn grid_index(idx: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let plane = ny * nz;
    let i = idx / plane;
    let rem = idx % plane;
    (i, rem / nz, rem % nz)
}

#[inline]
fn linear_index(index: (usize, usize, usize), ny: usize, nz: usize) -> usize {
    (index.0 * ny + index.1) * nz + index.2
}

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
            f_ref(grid_index(idx, ny, nz), value);
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
                f_ref(grid_index(idx, ny, nz), value, &input[idx]);
            });
        }
        _ => Zip::indexed(values).and(input).for_each(f),
    }
}

/// Apply an indexed mutation over one mutable and three immutable 3-D views.
#[inline]
pub(crate) fn for_each_indexed_mut_three_refs<T, U, V, W, F>(
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
    F: Fn((usize, usize, usize), &mut T, &U, &V, &W) + Send + Sync,
{
    assert_eq!(
        values.dim(),
        first.dim(),
        "invariant: physics indexed zip first shape mismatch"
    );
    assert_eq!(
        values.dim(),
        second.dim(),
        "invariant: physics indexed zip second shape mismatch"
    );
    assert_eq!(
        values.dim(),
        third.dim(),
        "invariant: physics indexed zip third shape mismatch"
    );

    let (_nx, ny, nz) = values.dim();
    match (
        values.as_slice_memory_order_mut(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
        third.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second), Some(third)) => {
            let f_ref = &f;
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f_ref(
                    grid_index(idx, ny, nz),
                    value,
                    &first[idx],
                    &second[idx],
                    &third[idx],
                );
            });
        }
        _ => Zip::indexed(values)
            .and(first)
            .and(second)
            .and(third)
            .for_each(f),
    }
}

/// Apply an indexed mutation over one mutable and four immutable 3-D views.
#[inline]
pub(crate) fn for_each_indexed_mut_four_refs<T, U, V, W, X, F>(
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
    F: Fn((usize, usize, usize), &mut T, &U, &V, &W, &X) + Send + Sync,
{
    assert_eq!(
        values.dim(),
        first.dim(),
        "invariant: physics indexed zip first shape mismatch"
    );
    assert_eq!(
        values.dim(),
        second.dim(),
        "invariant: physics indexed zip second shape mismatch"
    );
    assert_eq!(
        values.dim(),
        third.dim(),
        "invariant: physics indexed zip third shape mismatch"
    );
    assert_eq!(
        values.dim(),
        fourth.dim(),
        "invariant: physics indexed zip fourth shape mismatch"
    );

    let (_nx, ny, nz) = values.dim();
    match (
        values.as_slice_memory_order_mut(),
        first.as_slice_memory_order(),
        second.as_slice_memory_order(),
        third.as_slice_memory_order(),
        fourth.as_slice_memory_order(),
    ) {
        (Some(values), Some(first), Some(second), Some(third), Some(fourth)) => {
            let f_ref = &f;
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f_ref(
                    grid_index(idx, ny, nz),
                    value,
                    &first[idx],
                    &second[idx],
                    &third[idx],
                    &fourth[idx],
                );
            });
        }
        _ => Zip::indexed(values)
            .and(first)
            .and(second)
            .and(third)
            .and(fourth)
            .for_each(f),
    }
}

/// Apply an indexed mutation over three mutable 3-D views.
#[inline]
pub(crate) fn for_each_indexed_three_mut<T, U, V, F>(
    mut first: ArrayViewMut3<'_, T>,
    mut second: ArrayViewMut3<'_, U>,
    mut third: ArrayViewMut3<'_, V>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    F: Fn(usize, &mut T, &mut U, &mut V) + Send + Sync,
{
    assert_eq!(
        first.dim(),
        second.dim(),
        "invariant: physics indexed triple output second shape mismatch"
    );
    assert_eq!(
        first.dim(),
        third.dim(),
        "invariant: physics indexed triple output third shape mismatch"
    );

    let (_nx, ny, nz) = first.dim();
    match (
        first.as_slice_memory_order_mut(),
        second.as_slice_memory_order_mut(),
        third.as_slice_memory_order_mut(),
    ) {
        (Some(first), Some(second), Some(third)) => {
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                first,
                second,
                third,
                FIELD_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk, third_chunk| {
                    let start = chunk_index * FIELD_CHUNK_SIZE;
                    for (offset, ((first_value, second_value), third_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .zip(third_chunk.iter_mut())
                        .enumerate()
                    {
                        f(start + offset, first_value, second_value, third_value);
                    }
                },
            );
        }
        _ => Zip::indexed(first)
            .and(second)
            .and(third)
            .for_each(|index, first, second, third| {
                f(linear_index(index, ny, nz), first, second, third);
            }),
    }
}

/// Apply an indexed mutation over four mutable 3-D views and one immutable 3-D view.
#[inline]
pub(crate) fn for_each_indexed_four_mut_ref<T, U, V, W, X, F>(
    mut first_out: ArrayViewMut3<'_, T>,
    mut second_out: ArrayViewMut3<'_, U>,
    mut third_out: ArrayViewMut3<'_, V>,
    mut fourth_out: ArrayViewMut3<'_, W>,
    input: ArrayView3<'_, X>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    W: Send,
    X: Sync,
    F: Fn((usize, usize, usize), &mut T, &mut U, &mut V, &mut W, &X) + Send + Sync,
{
    assert_eq!(
        first_out.dim(),
        second_out.dim(),
        "invariant: physics indexed quad output second shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        third_out.dim(),
        "invariant: physics indexed quad output third shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        fourth_out.dim(),
        "invariant: physics indexed quad output fourth shape mismatch"
    );
    assert_eq!(
        first_out.dim(),
        input.dim(),
        "invariant: physics indexed quad input shape mismatch"
    );

    let (_nx, ny, nz) = first_out.dim();
    match (
        first_out.as_slice_memory_order_mut(),
        second_out.as_slice_memory_order_mut(),
        third_out.as_slice_memory_order_mut(),
        fourth_out.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
    ) {
        (Some(first_out), Some(second_out), Some(third_out), Some(fourth_out), Some(input)) => {
            for_each_chunk_quad_mut_enumerated_with::<Adaptive, _, _, _, _, _>(
                first_out,
                second_out,
                third_out,
                fourth_out,
                FIELD_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk, third_chunk, fourth_chunk| {
                    let start = chunk_index * FIELD_CHUNK_SIZE;
                    for (offset, (((first_value, second_value), third_value), fourth_value)) in
                        first_chunk
                            .iter_mut()
                            .zip(second_chunk.iter_mut())
                            .zip(third_chunk.iter_mut())
                            .zip(fourth_chunk.iter_mut())
                            .enumerate()
                    {
                        let idx = start + offset;
                        f(
                            grid_index(idx, ny, nz),
                            first_value,
                            second_value,
                            third_value,
                            fourth_value,
                            &input[idx],
                        );
                    }
                },
            );
        }
        _ => Zip::indexed(first_out)
            .and(second_out)
            .and(third_out)
            .and(fourth_out)
            .and(input)
            .for_each(f),
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
