//! Provider-owned traversal adapters for application arrays.

use leto::ArrayViewMut;
use leto_ops::ZipSources;
use moirai_parallel::{
    for_each_chunk_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with, Adaptive,
};

const FIELD_CHUNK_SIZE: usize = 4096;

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
        .expect("invariant: field zip source shapes and storage must match output");

    match (out.as_mut_slice(), sources.contiguous()) {
        (Some(out), Some(source_slices)) => {
            let f_ref = &f;
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                out,
                FIELD_CHUNK_SIZE,
                |chunk_index, chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
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
                    .expect("invariant: validated field zip offsets are representable");
                f(value, sources.values(offsets));
                next_index(&mut index, &shape);
            }
        }
    }
}

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
        "invariant: paired traversal output shapes must match"
    );
    sources
        .validate(first_out.shape())
        .expect("invariant: field zip source shapes and storage must match outputs");

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
                FIELD_CHUNK_SIZE,
                |chunk_index, first_chunk, second_chunk| {
                    let base = chunk_index * FIELD_CHUNK_SIZE;
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
                    .expect("invariant: validated field zip offsets are representable");
                f(first_value, second_value, sources.values(offsets));
                next_index(&mut index, &shape);
            }
        }
    }
}
