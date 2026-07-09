//! Provider-owned traversal adapters for math kernels.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use leto::{ArrayView, ArrayViewMut};
use leto_ops::{zip_mut_with, zip2_mut_with};

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

pub(crate) fn zip_mut_two_refs<T, U, V, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    first: ArrayView<'_, U, N>,
    second: ArrayView<'_, V, N>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    F: Fn(&mut T, &U, &V) + Send + Sync,
{
    assert_eq!(
        out.shape(),
        first.shape(),
        "invariant: math traversal output shape must match first input shape"
    );
    assert_eq!(
        out.shape(),
        second.shape(),
        "invariant: math traversal output shape must match second input shape"
    );

    match (out.as_mut_slice(), first.as_slice(), second.as_slice()) {
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
        _ => zip2_mut_with(&mut out, &first, &second, f).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use leto::Array3;

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
}
