//! Dual-output traversal adapters: two mutable views plus one or more
//! immutable views, closed over with references.

use leto::{ArrayView, ArrayViewMut};
use moirai_parallel::{for_each_chunk_pair_mut_enumerated_with, Adaptive};

use super::{next_index, THERAPY_CHUNK_SIZE};

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
    use leto::Array2;

    use super::{zip_two_mut_four_refs, zip_two_mut_ref, zip_two_mut_two_refs};

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
