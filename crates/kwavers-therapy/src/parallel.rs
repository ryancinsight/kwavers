//! Provider-owned traversal adapters for therapy kernels.

use moirai_parallel::{
    for_each_chunk_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with, Adaptive,
};
use ndarray::{ArrayView, ArrayViewMut, Dimension, Zip};

const THERAPY_CHUNK_SIZE: usize = 4096;

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
        "invariant: therapy traversal output shape must match input shape"
    );

    match (out.as_slice_mut(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, &input[base + lane]);
                    }
                },
            );
        }
        _ => Zip::from(out).and(input).for_each(f),
    }
}

pub(crate) fn zip_mut_four_refs<T, U, V, W, X, D, F>(
    mut out: ArrayViewMut<'_, T, D>,
    first: ArrayView<'_, U, D>,
    second: ArrayView<'_, V, D>,
    third: ArrayView<'_, W, D>,
    fourth: ArrayView<'_, X, D>,
    f: F,
) where
    D: Dimension,
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    X: Sync,
    F: Fn(&mut T, &U, &V, &W, &X) + Send + Sync,
{
    assert_eq!(
        out.dim(),
        first.dim(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        out.dim(),
        second.dim(),
        "invariant: therapy traversal output shape must match second input shape"
    );
    assert_eq!(
        out.dim(),
        third.dim(),
        "invariant: therapy traversal output shape must match third input shape"
    );
    assert_eq!(
        out.dim(),
        fourth.dim(),
        "invariant: therapy traversal output shape must match fourth input shape"
    );

    match (
        out.as_slice_mut(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
        fourth.as_slice(),
    ) {
        (Some(out), Some(first), Some(second), Some(third), Some(fourth)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        let index = base + lane;
                        f_ref(
                            value,
                            &first[index],
                            &second[index],
                            &third[index],
                            &fourth[index],
                        );
                    }
                },
            );
        }
        _ => Zip::from(out)
            .and(first)
            .and(second)
            .and(third)
            .and(fourth)
            .for_each(f),
    }
}

pub(crate) fn zip_two_mut_ref<T, U, V, D, F>(
    mut first_out: ArrayViewMut<'_, T, D>,
    mut second_out: ArrayViewMut<'_, U, D>,
    input: ArrayView<'_, V, D>,
    f: F,
) where
    D: Dimension,
    T: Send,
    U: Send,
    V: Sync,
    F: Fn(&mut T, &mut U, &V) + Send + Sync,
{
    assert_eq!(
        first_out.dim(),
        second_out.dim(),
        "invariant: therapy traversal output shapes must match"
    );
    assert_eq!(
        first_out.dim(),
        input.dim(),
        "invariant: therapy traversal output shape must match input shape"
    );

    match (
        first_out.as_slice_mut(),
        second_out.as_slice_mut(),
        input.as_slice(),
    ) {
        (Some(first_out), Some(second_out), Some(input)) => {
            let f_ref = &f;
            for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                first_out,
                second_out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, (first_value, second_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .enumerate()
                    {
                        f_ref(first_value, second_value, &input[base + lane]);
                    }
                },
            );
        }
        _ => Zip::from(first_out).and(second_out).and(input).for_each(f),
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
        "invariant: therapy traversal output shapes must match"
    );
    assert_eq!(
        first_out.dim(),
        first.dim(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        first_out.dim(),
        second.dim(),
        "invariant: therapy traversal output shape must match second input shape"
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
                THERAPY_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
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
        _ => Zip::from(first_out)
            .and(second_out)
            .and(first)
            .and(second)
            .for_each(f),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{s, Array2};

    use super::{zip_mut_four_refs, zip_mut_ref, zip_two_mut_ref, zip_two_mut_two_refs};

    #[test]
    fn zip_mut_ref_updates_dense_arrays() {
        let input = Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_ref(out.view_mut(), input.view(), |out, input| {
            *out = input * 2;
        });

        assert_eq!(
            out,
            Array2::from_shape_fn((2, 3), |(i, j)| 2 * (i + j) as i32)
        );
    }

    #[test]
    fn zip_mut_four_refs_updates_strided_views() {
        let first = Array2::from_shape_fn((4, 3), |(i, j)| (i + j) as i32);
        let second = Array2::from_shape_fn((4, 3), |(i, j)| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((4, 3), |(i, j)| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((4, 3), |(i, j)| (i * j) as i32);
        let mut out = Array2::zeros((4, 3));

        zip_mut_four_refs(
            out.slice_mut(s![..;2, ..]),
            first.slice(s![..;2, ..]),
            second.slice(s![..;2, ..]),
            third.slice(s![..;2, ..]),
            fourth.slice(s![..;2, ..]),
            |out, first, second, third, fourth| {
                *out = first + second + third + fourth;
            },
        );

        assert_eq!(
            out.slice(s![..;2, ..]),
            Array2::from_shape_fn((2, 3), |(i, j)| {
                let source_i = i * 2;
                (source_i + j) as i32
                    + (source_i * 2 + j) as i32
                    + (source_i + j * 2) as i32
                    + (source_i * j) as i32
            })
        );
        assert_eq!(out.slice(s![1..;2, ..]), Array2::<i32>::zeros((2, 3)));
    }

    #[test]
    fn zip_two_mut_ref_updates_both_outputs() {
        let input = Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));

        zip_two_mut_ref(
            first_out.view_mut(),
            second_out.view_mut(),
            input.view(),
            |first_out, second_out, input| {
                *first_out = *input;
                *second_out = -*input;
            },
        );

        assert_eq!(first_out, input);
        assert_eq!(second_out, input.mapv(|value| -value));
    }

    #[test]
    fn zip_two_mut_two_refs_updates_both_outputs() {
        let first = Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |(i, j)| (i * 2 + j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));

        zip_two_mut_two_refs(
            first_out.view_mut(),
            second_out.view_mut(),
            first.view(),
            second.view(),
            |first_out, second_out, first, second| {
                *first_out = first + second;
                *second_out = first - second;
            },
        );

        assert_eq!(
            first_out,
            Array2::from_shape_fn((2, 3), |(i, j)| { (i + j) as i32 + (i * 2 + j) as i32 })
        );
        assert_eq!(
            second_out,
            Array2::from_shape_fn((2, 3), |(i, j)| { (i + j) as i32 - (i * 2 + j) as i32 })
        );
    }
}
