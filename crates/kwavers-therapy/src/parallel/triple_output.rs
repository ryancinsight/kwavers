//! Triple-output traversal adapters: three mutable views plus one or more
//! immutable views, closed over with references.

use leto::{ArrayView, ArrayViewMut};
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

use super::{next_index, THERAPY_CHUNK_SIZE};

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

#[cfg(test)]
mod tests {
    use leto::Array2;

    use super::{zip_three_mut_three_refs, zip_three_mut_two_refs};

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
}
