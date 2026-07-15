//! Provider-owned traversal adapters for therapy kernels.

use leto::{ArrayView, ArrayViewMut};
use moirai_parallel::{
    for_each_chunk_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with,
    for_each_chunk_triple_mut_enumerated_with, Adaptive,
};

const THERAPY_CHUNK_SIZE: usize = 4096;

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
        "invariant: therapy traversal output shape must match input shape"
    );

    match (out.as_mut_slice(), input.as_slice()) {
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

pub(crate) fn zip_mut_four_refs<T, U, V, W, X, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    first: ArrayView<'_, U, N>,
    second: ArrayView<'_, V, N>,
    third: ArrayView<'_, W, N>,
    fourth: ArrayView<'_, X, N>,
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
        out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
    );
    assert_eq!(
        out.shape(),
        third.shape(),
        "invariant: therapy traversal output shape must match third input shape"
    );
    assert_eq!(
        out.shape(),
        fourth.shape(),
        "invariant: therapy traversal output shape must match fourth input shape"
    );

    match (
        out.as_mut_slice(),
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
        _ => {
            let shape = out.shape();
            let mut index = [0usize; N];
            for _ in 0..out.size() {
                let value = out.get_mut(index).expect("invariant: index in bounds");
                f(
                    value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                    third.get(index).expect("invariant: index in bounds"),
                    fourth.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_mut_three_refs<T, U, V, W, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    first: ArrayView<'_, U, N>,
    second: ArrayView<'_, V, N>,
    third: ArrayView<'_, W, N>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &U, &V, &W) + Send + Sync,
{
    assert_eq!(
        out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
    );
    assert_eq!(
        out.shape(),
        third.shape(),
        "invariant: therapy traversal output shape must match third input shape"
    );

    match (
        out.as_mut_slice(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
    ) {
        (Some(out), Some(first), Some(second), Some(third)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        let index = base + lane;
                        f_ref(value, &first[index], &second[index], &third[index]);
                    }
                },
            );
        }
        _ => {
            let shape = out.shape();
            let mut index = [0usize; N];
            for _ in 0..out.size() {
                let value = out.get_mut(index).expect("invariant: index in bounds");
                f(
                    value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                    third.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_mut_five_refs<T, U, V, W, X, Y, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    first: ArrayView<'_, U, N>,
    second: ArrayView<'_, V, N>,
    third: ArrayView<'_, W, N>,
    fourth: ArrayView<'_, X, N>,
    fifth: ArrayView<'_, Y, N>,
    f: F,
) where
    T: Send,
    U: Sync,
    V: Sync,
    W: Sync,
    X: Sync,
    Y: Sync,
    F: Fn(&mut T, &U, &V, &W, &X, &Y) + Send + Sync,
{
    assert_eq!(
        out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
    );
    assert_eq!(
        out.shape(),
        third.shape(),
        "invariant: therapy traversal output shape must match third input shape"
    );
    assert_eq!(
        out.shape(),
        fourth.shape(),
        "invariant: therapy traversal output shape must match fourth input shape"
    );
    assert_eq!(
        out.shape(),
        fifth.shape(),
        "invariant: therapy traversal output shape must match fifth input shape"
    );

    match (
        out.as_mut_slice(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
        fourth.as_slice(),
        fifth.as_slice(),
    ) {
        (Some(out), Some(first), Some(second), Some(third), Some(fourth), Some(fifth)) => {
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
                            &fifth[index],
                        );
                    }
                },
            );
        }
        _ => {
            let shape = out.shape();
            let mut index = [0usize; N];
            for _ in 0..out.size() {
                let value = out.get_mut(index).expect("invariant: index in bounds");
                f(
                    value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                    third.get(index).expect("invariant: index in bounds"),
                    fourth.get(index).expect("invariant: index in bounds"),
                    fifth.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_two_mut_ref<T, U, V, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    input: ArrayView<'_, V, N>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Sync,
    F: Fn(&mut T, &mut U, &V) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: therapy traversal output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        input.shape(),
        "invariant: therapy traversal output shape must match input shape"
    );

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
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
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let second_value = second_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                f(
                    first_value,
                    second_value,
                    input.get(index).expect("invariant: index in bounds"),
                );
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
        "invariant: therapy traversal output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
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
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let second_value = second_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
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

pub(crate) fn zip_three_mut_two_refs<T, U, V, W, X, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    mut third_out: ArrayViewMut<'_, V, N>,
    first: ArrayView<'_, W, N>,
    second: ArrayView<'_, X, N>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    W: Sync,
    X: Sync,
    F: Fn(&mut T, &mut U, &mut V, &W, &X) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: therapy traversal first and second output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        third_out.shape(),
        "invariant: therapy traversal first and third output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
    );

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        third_out.as_mut_slice(),
        first.as_slice(),
        second.as_slice(),
    ) {
        (Some(first_out), Some(second_out), Some(third_out), Some(first), Some(second)) => {
            let f_ref = &f;
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                first_out,
                second_out,
                third_out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk, third_chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, ((first_value, second_value), third_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .zip(third_chunk.iter_mut())
                        .enumerate()
                    {
                        let index = base + lane;
                        f_ref(
                            first_value,
                            second_value,
                            third_value,
                            &first[index],
                            &second[index],
                        );
                    }
                },
            );
        }
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let second_value = second_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let third_value = third_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                f(
                    first_value,
                    second_value,
                    third_value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_three_mut_three_refs<T, U, V, W, X, Y, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    mut third_out: ArrayViewMut<'_, V, N>,
    first: ArrayView<'_, W, N>,
    second: ArrayView<'_, X, N>,
    third: ArrayView<'_, Y, N>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    W: Sync,
    X: Sync,
    Y: Sync,
    F: Fn(&mut T, &mut U, &mut V, &W, &X, &Y) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: therapy traversal first and second output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        third_out.shape(),
        "invariant: therapy traversal first and third output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
    );
    assert_eq!(
        first_out.shape(),
        third.shape(),
        "invariant: therapy traversal output shape must match third input shape"
    );

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        third_out.as_mut_slice(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
    ) {
        (
            Some(first_out),
            Some(second_out),
            Some(third_out),
            Some(first),
            Some(second),
            Some(third),
        ) => {
            let f_ref = &f;
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                first_out,
                second_out,
                third_out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk, third_chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, ((first_value, second_value), third_value)) in first_chunk
                        .iter_mut()
                        .zip(second_chunk.iter_mut())
                        .zip(third_chunk.iter_mut())
                        .enumerate()
                    {
                        let index = base + lane;
                        f_ref(
                            first_value,
                            second_value,
                            third_value,
                            &first[index],
                            &second[index],
                            &third[index],
                        );
                    }
                },
            );
        }
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let second_value = second_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let third_value = third_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                f(
                    first_value,
                    second_value,
                    third_value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                    third.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_two_mut_four_refs<T, U, V, W, X, Y, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    first: ArrayView<'_, V, N>,
    second: ArrayView<'_, W, N>,
    third: ArrayView<'_, X, N>,
    fourth: ArrayView<'_, Y, N>,
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
        "invariant: therapy traversal output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: therapy traversal output shape must match first input shape"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: therapy traversal output shape must match second input shape"
    );
    assert_eq!(
        first_out.shape(),
        third.shape(),
        "invariant: therapy traversal output shape must match third input shape"
    );
    assert_eq!(
        first_out.shape(),
        fourth.shape(),
        "invariant: therapy traversal output shape must match fourth input shape"
    );

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        first.as_slice(),
        second.as_slice(),
        third.as_slice(),
        fourth.as_slice(),
    ) {
        (
            Some(first_out),
            Some(second_out),
            Some(first),
            Some(second),
            Some(third),
            Some(fourth),
        ) => {
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
                        f_ref(
                            first_value,
                            second_value,
                            &first[index],
                            &second[index],
                            &third[index],
                            &fourth[index],
                        );
                    }
                },
            );
        }
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let second_value = second_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                f(
                    first_value,
                    second_value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                    third.get(index).expect("invariant: index in bounds"),
                    fourth.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use leto::{Array2, SliceArg};

    use super::{
        zip_mut_five_refs, zip_mut_four_refs, zip_mut_ref, zip_mut_three_refs,
        zip_three_mut_three_refs, zip_three_mut_two_refs, zip_two_mut_four_refs, zip_two_mut_ref,
        zip_two_mut_two_refs,
    };

    /// `s![..;2, ..]` in leto slice-argument form.
    fn every_other_row() -> [SliceArg; 2] {
        [
            SliceArg::Range {
                start: None,
                end: None,
                step: 2,
            },
            SliceArg::All,
        ]
    }

    /// `s![1..;2, ..]` in leto slice-argument form.
    fn odd_rows() -> [SliceArg; 2] {
        [
            SliceArg::Range {
                start: Some(1),
                end: None,
                step: 2,
            },
            SliceArg::All,
        ]
    }

    #[test]
    fn zip_mut_ref_updates_dense_arrays() {
        let input = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_ref(out.view_mut(), input.view(), |out, input| {
            *out = input * 2;
        });

        assert_eq!(
            out,
            Array2::from_shape_fn((2, 3), |[i, j]| 2 * (i + j) as i32)
        );
    }

    #[test]
    fn zip_mut_four_refs_updates_strided_views() {
        let first = Array2::from_shape_fn((4, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((4, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((4, 3), |[i, j]| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((4, 3), |[i, j]| (i * j) as i32);
        let mut out = Array2::zeros((4, 3));

        zip_mut_four_refs(
            out.slice_with_mut::<2>(&every_other_row()).unwrap(),
            first.slice_with::<2>(&every_other_row()).unwrap(),
            second.slice_with::<2>(&every_other_row()).unwrap(),
            third.slice_with::<2>(&every_other_row()).unwrap(),
            fourth.slice_with::<2>(&every_other_row()).unwrap(),
            |out, first, second, third, fourth| {
                *out = first + second + third + fourth;
            },
        );

        assert_eq!(
            out.slice_with::<2>(&every_other_row())
                .unwrap()
                .to_contiguous(),
            Array2::from_shape_fn((2, 3), |[i, j]| {
                let source_i = i * 2;
                (source_i + j) as i32
                    + (source_i * 2 + j) as i32
                    + (source_i + j * 2) as i32
                    + (source_i * j) as i32
            })
        );
        assert_eq!(
            out.slice_with::<2>(&odd_rows()).unwrap().to_contiguous(),
            Array2::<i32>::zeros((2, 3))
        );
    }

    #[test]
    fn zip_mut_three_refs_updates_dense_arrays() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_three_refs(
            out.view_mut(),
            first.view(),
            second.view(),
            third.view(),
            |out, first, second, third| {
                *out = first + second + third;
            },
        );

        assert_eq!(
            out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32 + (i * 2 + j) as i32 + (i + j * 2) as i32
            })
        );
    }

    #[test]
    fn zip_mut_five_refs_updates_dense_arrays() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((2, 3), |[i, j]| (i * j) as i32);
        let fifth = Array2::from_shape_fn((2, 3), |[i, j]| (i + 3 * j) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_five_refs(
            out.view_mut(),
            first.view(),
            second.view(),
            third.view(),
            fourth.view(),
            fifth.view(),
            |out, first, second, third, fourth, fifth| {
                *out = first + second + third + fourth + fifth;
            },
        );

        assert_eq!(
            out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32
                    + (i * 2 + j) as i32
                    + (i + j * 2) as i32
                    + (i * j) as i32
                    + (i + 3 * j) as i32
            })
        );
    }

    #[test]
    fn zip_two_mut_ref_updates_both_outputs() {
        let input = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
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
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
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
            Array2::from_shape_fn((2, 3), |[i, j]| { (i + j) as i32 + (i * 2 + j) as i32 })
        );
        assert_eq!(
            second_out,
            Array2::from_shape_fn((2, 3), |[i, j]| { (i + j) as i32 - (i * 2 + j) as i32 })
        );
    }

    #[test]
    fn zip_three_mut_two_refs_updates_all_outputs() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));
        let mut third_out = Array2::zeros((2, 3));

        zip_three_mut_two_refs(
            first_out.view_mut(),
            second_out.view_mut(),
            third_out.view_mut(),
            first.view(),
            second.view(),
            |first_out, second_out, third_out, first, second| {
                *first_out = first + second;
                *second_out = first - second;
                *third_out = first * second;
            },
        );

        assert_eq!(
            first_out,
            Array2::from_shape_fn((2, 3), |[i, j]| { (i + j) as i32 + (i * 2 + j) as i32 })
        );
        assert_eq!(
            second_out,
            Array2::from_shape_fn((2, 3), |[i, j]| { (i + j) as i32 - (i * 2 + j) as i32 })
        );
        assert_eq!(
            third_out,
            Array2::from_shape_fn((2, 3), |[i, j]| { (i + j) as i32 * (i * 2 + j) as i32 })
        );
    }

    #[test]
    fn zip_three_mut_three_refs_updates_all_outputs() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));
        let mut third_out = Array2::zeros((2, 3));

        zip_three_mut_three_refs(
            first_out.view_mut(),
            second_out.view_mut(),
            third_out.view_mut(),
            first.view(),
            second.view(),
            third.view(),
            |first_out, second_out, third_out, first, second, third| {
                *first_out = first + second + third;
                *second_out = first - second + third;
                *third_out = first * second - third;
            },
        );

        assert_eq!(
            first_out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32 + (i * 2 + j) as i32 + (i + j * 2) as i32
            })
        );
        assert_eq!(
            second_out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32 - (i * 2 + j) as i32 + (i + j * 2) as i32
            })
        );
        assert_eq!(
            third_out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32 * (i * 2 + j) as i32 - (i + j * 2) as i32
            })
        );
    }

    #[test]
    fn zip_two_mut_four_refs_updates_both_outputs() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((2, 3), |[i, j]| (i * j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));

        zip_two_mut_four_refs(
            first_out.view_mut(),
            second_out.view_mut(),
            first.view(),
            second.view(),
            third.view(),
            fourth.view(),
            |first_out, second_out, first, second, third, fourth| {
                *first_out = first + second + third + fourth;
                *second_out = first - second + third - fourth;
            },
        );

        assert_eq!(
            first_out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32 + (i * 2 + j) as i32 + (i + j * 2) as i32 + (i * j) as i32
            })
        );
        assert_eq!(
            second_out,
            Array2::from_shape_fn((2, 3), |[i, j]| {
                (i + j) as i32 - (i * 2 + j) as i32 + (i + j * 2) as i32 - (i * j) as i32
            })
        );
    }
}
