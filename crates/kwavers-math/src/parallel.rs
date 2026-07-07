//! Provider-owned traversal adapters for math kernels.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{ArrayView, ArrayViewMut, Dimension, Zip};

const MATH_CHUNK_SIZE: usize = 4096;

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
        "invariant: math traversal output shape must match input shape"
    );

    match (out.as_slice_mut(), input.as_slice()) {
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
        _ => Zip::from(out).and(input).for_each(f),
    }
}

pub(crate) fn zip_mut_two_refs<T, U, V, D, F>(
    mut out: ArrayViewMut<'_, T, D>,
    first: ArrayView<'_, U, D>,
    second: ArrayView<'_, V, D>,
    f: F,
) where
    D: Dimension,
    T: Send,
    U: Sync,
    V: Sync,
    F: Fn(&mut T, &U, &V) + Send + Sync,
{
    assert_eq!(
        out.dim(),
        first.dim(),
        "invariant: math traversal output shape must match first input shape"
    );
    assert_eq!(
        out.dim(),
        second.dim(),
        "invariant: math traversal output shape must match second input shape"
    );

    match (out.as_slice_mut(), first.as_slice(), second.as_slice()) {
        (Some(out), Some(first), Some(second)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                MATH_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * MATH_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        let index = base + lane;
                        f_ref(value, &first[index], &second[index]);
                    }
                },
            );
        }
        _ => Zip::from(out).and(first).and(second).for_each(f),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{s, Array3};

    use super::zip_mut_two_refs;

    #[test]
    fn zip_mut_two_refs_updates_dense_arrays() {
        let first = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as i32);
        let second = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (2 * i + j + k) as i32);
        let mut out = Array3::zeros((2, 2, 2));

        zip_mut_two_refs(
            out.view_mut(),
            first.view(),
            second.view(),
            |out, first, second| {
                *out = first + second;
            },
        );

        assert_eq!(
            out,
            Array3::from_shape_fn((2, 2, 2), |(i, j, k)| {
                (i + j + k) as i32 + (2 * i + j + k) as i32
            })
        );
    }

    #[test]
    fn zip_mut_two_refs_updates_strided_views() {
        let first = Array3::from_shape_fn((4, 2, 2), |(i, j, k)| (i + j + k) as i32);
        let second = Array3::from_shape_fn((4, 2, 2), |(i, j, k)| (i * j + k) as i32);
        let mut out = Array3::zeros((4, 2, 2));

        zip_mut_two_refs(
            out.slice_mut(s![..;2, .., ..]),
            first.slice(s![..;2, .., ..]),
            second.slice(s![..;2, .., ..]),
            |out, first, second| {
                *out = first - second;
            },
        );

        assert_eq!(
            out.slice(s![..;2, .., ..]),
            Array3::from_shape_fn((2, 2, 2), |(i, j, k)| {
                let source_i = i * 2;
                (source_i + j + k) as i32 - (source_i * j + k) as i32
            })
        );
        assert_eq!(out.slice(s![1..;2, .., ..]), Array3::zeros((2, 2, 2)));
    }
}
