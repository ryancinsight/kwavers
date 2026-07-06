//! Atlas parallel-provider adapters for medium field traversal.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, for_each_chunk_mut_with, Adaptive};
use ndarray::{Array3, ArrayBase, DataMut, Dimension, Zip};

pub(crate) const FIELD_CHUNK_SIZE: usize = 4096;

pub(crate) fn for_each_mut<T, S, D, F>(array: &mut ArrayBase<S, D>, f: F)
where
    S: DataMut<Elem = T>,
    D: Dimension,
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    if let Some(values) = array.as_slice_mut() {
        let f_ref = &f;
        for_each_chunk_mut_with::<Adaptive, _, _>(values, FIELD_CHUNK_SIZE, |chunk| {
            chunk.iter_mut().for_each(f_ref);
        });
    } else {
        array.iter_mut().for_each(f);
    }
}

pub(crate) fn zip_mut_ref<T, U, F>(out: &mut Array3<T>, input: &Array3<U>, f: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    debug_assert_eq!(out.dim(), input.dim());
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

pub(crate) fn zip_mut_two_refs<T, U, V, F>(
    out: &mut Array3<T>,
    first: &Array3<U>,
    second: &Array3<V>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    F: Fn(&mut T, &U, &V) + Send + Sync,
{
    debug_assert_eq!(out.dim(), first.dim());
    debug_assert_eq!(out.dim(), second.dim());
    match (out.as_slice_mut(), first.as_slice(), second.as_slice()) {
        (Some(out), Some(first), Some(second)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &first[base + lane], &second[base + lane]);
                    }
                },
            );
        }
        _ => Zip::from(out).and(first).and(second).for_each(f),
    }
}

pub(crate) fn zip_mut_three_refs<T, U, V, W, F>(
    out: &mut Array3<T>,
    first: &Array3<U>,
    second: &Array3<V>,
    third: &Array3<W>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &U, &V, &W) + Send + Sync,
{
    debug_assert_eq!(out.dim(), first.dim());
    debug_assert_eq!(out.dim(), second.dim());
    debug_assert_eq!(out.dim(), third.dim());
    match (
        out.as_slice_mut(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
    ) {
        (Some(out), Some(first), Some(second), Some(third)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(
                            value,
                            &first[base + lane],
                            &second[base + lane],
                            &third[base + lane],
                        );
                    }
                },
            );
        }
        _ => Zip::from(out).and(first).and(second).and(third).for_each(f),
    }
}
