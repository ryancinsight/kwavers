//! Atlas parallel-provider adapters for physics field traversal.

use leto::{ArrayView3, ArrayViewMut3};
use leto_ops::ZipSources;
use moirai_parallel::{
    enumerate_mut_with, for_each_chunk_pair_mut_enumerated_with,
    for_each_chunk_triple_mut_enumerated_with, Adaptive,
};

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
    let [_nx, ny, nz] = values.shape();
    if let Some(slice) = values.as_mut_slice_memory_order() {
        let f_ref = &f;
        enumerate_mut_with::<Adaptive, _, _>(slice, |idx, value| {
            f_ref(grid_index(idx, ny, nz), value);
        });
    } else if let Ok(iter) = values.indexed_iter_mut() {
        for (idx, value) in iter {
            f((idx[0], idx[1], idx[2]), value);
        }
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
        values.shape(),
        input.shape(),
        "invariant: physics paired traversal shape mismatch"
    );

    let [_nx, ny, nz] = values.shape();
    match (
        values.as_mut_slice_memory_order(),
        input.as_slice_memory_order(),
    ) {
        (Some(values), Some(input)) => {
            let f_ref = &f;
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                f_ref(grid_index(idx, ny, nz), value, &input[idx]);
            });
        }
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    f(
                        (idx[0], idx[1], idx[2]),
                        value,
                        &input[[idx[0], idx[1], idx[2]]],
                    );
                }
            }
        }
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
        values.shape(),
        first.shape(),
        "invariant: physics indexed zip first shape mismatch"
    );
    assert_eq!(
        values.shape(),
        second.shape(),
        "invariant: physics indexed zip second shape mismatch"
    );
    assert_eq!(
        values.shape(),
        third.shape(),
        "invariant: physics indexed zip third shape mismatch"
    );

    let [_nx, ny, nz] = values.shape();
    match (
        values.as_mut_slice_memory_order(),
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
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    let [i, j, k] = idx;
                    f(
                        (i, j, k),
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
        values.shape(),
        first.shape(),
        "invariant: physics indexed zip first shape mismatch"
    );
    assert_eq!(
        values.shape(),
        second.shape(),
        "invariant: physics indexed zip second shape mismatch"
    );
    assert_eq!(
        values.shape(),
        third.shape(),
        "invariant: physics indexed zip third shape mismatch"
    );
    assert_eq!(
        values.shape(),
        fourth.shape(),
        "invariant: physics indexed zip fourth shape mismatch"
    );

    let [_nx, ny, nz] = values.shape();
    match (
        values.as_mut_slice_memory_order(),
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
        _ => {
            if let Ok(iter) = values.indexed_iter_mut() {
                for (idx, value) in iter {
                    let [i, j, k] = idx;
                    f(
                        (i, j, k),
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
        first.shape(),
        second.shape(),
        "invariant: physics indexed triple output second shape mismatch"
    );
    assert_eq!(
        first.shape(),
        third.shape(),
        "invariant: physics indexed triple output third shape mismatch"
    );

    let [_nx, ny, nz] = first.shape();
    match (
        first.as_mut_slice_memory_order(),
        second.as_mut_slice_memory_order(),
        third.as_mut_slice_memory_order(),
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
        _ => {
            if let Ok(iter) = first.indexed_iter_mut() {
                for (idx, value) in iter {
                    let flat = linear_index((idx[0], idx[1], idx[2]), ny, nz);
                    f(
                        flat,
                        value,
                        &mut second[[idx[0], idx[1], idx[2]]],
                        &mut third[[idx[0], idx[1], idx[2]]],
                    );
                }
            }
        }
    }
}

/// Apply an unindexed mutation over one or more mutable 3-D views and one or
/// more statically typed immutable views. Output arity is encoded in the tuple
/// type, so one monomorphized entry point covers one and two mutable outputs.
pub(crate) trait MutableZipOutputs<S, F>: Sized {
    fn zip(outputs: Self, sources: S, f: F);
}

impl<'data, T, S, F> MutableZipOutputs<S, F> for ArrayViewMut3<'data, T>
where
    T: Send,
    S: ZipSources<3>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn(&mut T, S::Values) + Send + Sync,
{
    fn zip(mut values: Self, sources: S, f: F) {
        sources
            .validate(values.shape())
            .expect("invariant: physics zip source shapes and storage must match output");

        match (values.as_mut_slice_memory_order(), sources.contiguous()) {
            (Some(values), Some(source_slices)) => {
                let f_ref = &f;
                enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                    f_ref(value, S::contiguous_values(source_slices, idx));
                });
            }
            _ => {
                if let Ok(iter) = values.indexed_iter_mut() {
                    for (index, value) in iter {
                        let offsets = sources
                            .offsets_at(index)
                            .expect("invariant: validated physics zip offsets are representable");
                        f(value, sources.values(offsets));
                    }
                }
            }
        }
    }
}

impl<'data, T, U, S, F> MutableZipOutputs<S, F>
    for (ArrayViewMut3<'data, T>, ArrayViewMut3<'data, U>)
where
    T: Send,
    U: Send,
    S: ZipSources<3>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn((&mut T, &mut U), S::Values) + Send + Sync,
{
    fn zip(mut outputs: Self, sources: S, f: F) {
        assert_eq!(
            outputs.0.shape(),
            outputs.1.shape(),
            "invariant: physics zip output shape mismatch"
        );
        sources
            .validate(outputs.0.shape())
            .expect("invariant: physics zip source shapes and storage must match outputs");

        match (
            outputs.0.as_mut_slice_memory_order(),
            outputs.1.as_mut_slice_memory_order(),
            sources.contiguous(),
        ) {
            (Some(first), Some(second), Some(source_slices)) => {
                let f_ref = &f;
                for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                    first,
                    second,
                    FIELD_CHUNK_SIZE,
                    |chunk_index, first_chunk, second_chunk| {
                        let start = chunk_index * FIELD_CHUNK_SIZE;
                        for (offset, (first_value, second_value)) in first_chunk
                            .iter_mut()
                            .zip(second_chunk.iter_mut())
                            .enumerate()
                        {
                            f_ref(
                                (first_value, second_value),
                                S::contiguous_values(source_slices, start + offset),
                            );
                        }
                    },
                );
            }
            _ => {
                if let Ok(iter) = outputs.0.indexed_iter_mut() {
                    for (index, value) in iter {
                        let [i, j, k] = index;
                        let offsets = sources
                            .offsets_at(index)
                            .expect("invariant: validated physics zip offsets are representable");
                        f((value, &mut outputs.1[[i, j, k]]), sources.values(offsets));
                    }
                }
            }
        }
    }
}

#[inline]
pub(crate) fn zip_mut_with<O, S, F>(outputs: O, sources: S, f: F)
where
    O: MutableZipOutputs<S, F>,
{
    <O as MutableZipOutputs<S, F>>::zip(outputs, sources, f);
}
