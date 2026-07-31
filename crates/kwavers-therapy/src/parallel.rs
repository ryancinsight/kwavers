//! Provider-owned traversal adapters for therapy kernels.

use leto::ArrayViewMut;
use leto_ops::ZipSources;
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

/// Mutably map one output with one or more statically typed source views.
///
/// The source tuple is monomorphized through [`ZipSources`], so arity and
/// element types carry no runtime dispatch or allocation. Dense outputs retain
/// the adaptive chunk scheduler; strided layouts use the source provider's
/// logical-index traversal.
pub(crate) fn zip_mut_with<T, S, F, const N: usize>(
    mut out: ArrayViewMut<'_, T, N>,
    sources: S,
    f: F,
) where
    T: Send,
    S: ZipSources<N>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn(&mut T, S::Values) + Send + Sync,
{
    sources
        .validate(out.shape())
        .expect("invariant: therapy zip source shapes and storage must match output");

    match (out.as_mut_slice(), sources.contiguous()) {
        (Some(out), Some(source_slices)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                THERAPY_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * THERAPY_CHUNK_SIZE;
                    for (lane, value) in chunk.iter_mut().enumerate() {
                        f_ref(value, S::contiguous_values(source_slices, base + lane));
                    }
                },
            );
        }
        _ => {
            let shape = out.shape();
            let mut index = [0usize; N];
            for _ in 0..out.size() {
                let value = out.get_mut(index).expect("invariant: index in bounds");
                let offsets = sources
                    .offsets_at(index)
                    .expect("invariant: validated therapy zip offsets are representable");
                f(&mut *value, sources.values(offsets));
                next_index(&mut index, &shape);
            }
        }
    }
}

/// Mutably map two outputs with one or more statically typed source views.
///
/// The two mutable outputs are traversed in one adaptive chunk pass. Source
/// arity and element types are supplied through [`ZipSources`], so each use
/// monomorphizes without allocation or dynamic dispatch.
pub(crate) fn zip_two_mut_with<T, U, S, F, const N: usize>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    sources: S,
    f: F,
) where
    T: Send,
    U: Send,
    S: ZipSources<N>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn(&mut T, &mut U, S::Values) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: therapy traversal output shapes must match"
    );

    sources
        .validate(first_out.shape())
        .expect("invariant: therapy zip source shapes and storage must match outputs");

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        sources.contiguous(),
    ) {
        (Some(first_out), Some(second_out), Some(source_slices)) => {
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
                        f_ref(
                            first_value,
                            second_value,
                            S::contiguous_values(source_slices, base + lane),
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
                let offsets = sources
                    .offsets_at(index)
                    .expect("invariant: validated therapy zip offsets are representable");
                f(first_value, second_value, sources.values(offsets));
                next_index(&mut index, &shape);
            }
        }
    }
}

/// Mutably map three outputs with one or more statically typed source views.
///
/// The three mutable outputs are traversed in one adaptive chunk pass. Source
/// arity and element types are supplied through [`ZipSources`], so each use
/// monomorphizes without allocation or dynamic dispatch.
pub(crate) fn zip_three_mut_with<T, U, V, S, F, const N: usize>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    mut third_out: ArrayViewMut<'_, V, N>,
    sources: S,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    S: ZipSources<N>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn(&mut T, &mut U, &mut V, S::Values) + Send + Sync,
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
    sources
        .validate(first_out.shape())
        .expect("invariant: therapy zip source shapes and storage must match outputs");

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        third_out.as_mut_slice(),
        sources.contiguous(),
    ) {
        (Some(first_out), Some(second_out), Some(third_out), Some(source_slices)) => {
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
                        f_ref(
                            first_value,
                            second_value,
                            third_value,
                            S::contiguous_values(source_slices, base + lane),
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
                let offsets = sources
                    .offsets_at(index)
                    .expect("invariant: validated therapy zip offsets are representable");
                f(
                    first_value,
                    second_value,
                    third_value,
                    sources.values(offsets),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use leto::{Array2, SliceArg};

    use super::{zip_mut_with, zip_three_mut_with, zip_two_mut_with};

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
    fn zip_mut_with_updates_dense_arrays() {
        let input = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_with(out.view_mut(), &input.view(), |out, input| {
            *out = input * 2;
        });

        assert_eq!(
            out,
            Array2::from_shape_fn((2, 3), |[i, j]| 2 * (i + j) as i32)
        );
    }

    #[test]
    fn zip_mut_with_updates_strided_views() {
        let first = Array2::from_shape_fn((4, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((4, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((4, 3), |[i, j]| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((4, 3), |[i, j]| (i * j) as i32);
        let mut out = Array2::zeros((4, 3));

        zip_mut_with(
            out.slice_with_mut::<2>(&every_other_row()).unwrap(),
            (
                &first.slice_with::<2>(&every_other_row()).unwrap(),
                &second.slice_with::<2>(&every_other_row()).unwrap(),
                &third.slice_with::<2>(&every_other_row()).unwrap(),
                &fourth.slice_with::<2>(&every_other_row()).unwrap(),
            ),
            |out, (&first, &second, &third, &fourth)| {
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
    fn zip_mut_with_updates_three_sources() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_with(
            out.view_mut(),
            (&first.view(), &second.view(), &third.view()),
            |out, (&first, &second, &third)| {
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
    fn zip_mut_with_updates_five_sources() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((2, 3), |[i, j]| (i * j) as i32);
        let fifth = Array2::from_shape_fn((2, 3), |[i, j]| (i + 3 * j) as i32);
        let mut out = Array2::zeros((2, 3));

        zip_mut_with(
            out.view_mut(),
            (
                &first.view(),
                &second.view(),
                &third.view(),
                &fourth.view(),
                &fifth.view(),
            ),
            |out, (&first, &second, &third, &fourth, &fifth)| {
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
    fn zip_two_mut_with_updates_both_outputs() {
        let input = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));

        zip_two_mut_with(
            first_out.view_mut(),
            second_out.view_mut(),
            &input.view(),
            |first_out, second_out, input| {
                *first_out = *input;
                *second_out = -*input;
            },
        );

        assert_eq!(first_out, input);
        assert_eq!(second_out, input.mapv(|value| -value));
    }

    #[test]
    fn zip_two_mut_with_updates_two_sources() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));

        zip_two_mut_with(
            first_out.view_mut(),
            second_out.view_mut(),
            (&first.view(), &second.view()),
            |first_out, second_out, (first, second)| {
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
    fn zip_two_mut_with_updates_strided_views() {
        let first = Array2::from_shape_fn((4, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((4, 3), |[i, j]| (i * 2 + j) as i32);
        let mut first_out = Array2::zeros((4, 3));
        let mut second_out = Array2::zeros((4, 3));

        zip_two_mut_with(
            first_out.slice_with_mut::<2>(&every_other_row()).unwrap(),
            second_out.slice_with_mut::<2>(&every_other_row()).unwrap(),
            (
                &first.slice_with::<2>(&every_other_row()).unwrap(),
                &second.slice_with::<2>(&every_other_row()).unwrap(),
            ),
            |first_out, second_out, (first, second)| {
                *first_out = first + second;
                *second_out = first - second;
            },
        );

        assert_eq!(
            first_out
                .slice_with::<2>(&every_other_row())
                .unwrap()
                .to_contiguous(),
            Array2::from_shape_fn((2, 3), |[i, j]| { (i * 2 + j) as i32 + (i * 4 + j) as i32 })
        );
        assert_eq!(
            second_out
                .slice_with::<2>(&every_other_row())
                .unwrap()
                .to_contiguous(),
            Array2::from_shape_fn((2, 3), |[i, j]| { (i * 2 + j) as i32 - (i * 4 + j) as i32 })
        );
    }

    #[test]
    fn zip_three_mut_with_updates_two_sources() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));
        let mut third_out = Array2::zeros((2, 3));

        zip_three_mut_with(
            first_out.view_mut(),
            second_out.view_mut(),
            third_out.view_mut(),
            (&first.view(), &second.view()),
            |first_out, second_out, third_out, (first, second)| {
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
    fn zip_three_mut_with_updates_three_sources() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));
        let mut third_out = Array2::zeros((2, 3));

        zip_three_mut_with(
            first_out.view_mut(),
            second_out.view_mut(),
            third_out.view_mut(),
            (&first.view(), &second.view(), &third.view()),
            |first_out, second_out, third_out, (first, second, third)| {
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
    fn zip_two_mut_with_updates_four_sources() {
        let first = Array2::from_shape_fn((2, 3), |[i, j]| (i + j) as i32);
        let second = Array2::from_shape_fn((2, 3), |[i, j]| (i * 2 + j) as i32);
        let third = Array2::from_shape_fn((2, 3), |[i, j]| (i + j * 2) as i32);
        let fourth = Array2::from_shape_fn((2, 3), |[i, j]| (i * j) as i32);
        let mut first_out = Array2::zeros((2, 3));
        let mut second_out = Array2::zeros((2, 3));

        zip_two_mut_with(
            first_out.view_mut(),
            second_out.view_mut(),
            (&first.view(), &second.view(), &third.view(), &fourth.view()),
            |first_out, second_out, (first, second, third, fourth)| {
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
