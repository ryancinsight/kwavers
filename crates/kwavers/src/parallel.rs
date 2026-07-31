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

pub(crate) trait MutableZipOutputs<S, F, const N: usize>: Sized {
    fn zip(outputs: Self, sources: S, f: F);
}

impl<'data, T, S, F, const N: usize> MutableZipOutputs<S, F, N> for ArrayViewMut<'data, T, N>
where
    T: Send,
    S: ZipSources<N>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn(&mut T, S::Values) + Send + Sync,
{
    fn zip(mut output: Self, sources: S, f: F) {
        sources
            .validate(output.shape())
            .expect("invariant: field zip source shapes and storage must match output");

        match (output.as_mut_slice(), sources.contiguous()) {
            (Some(output), Some(source_slices)) => {
                let f_ref = &f;
                for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                    output,
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
                let shape = output.shape();
                let mut index = [0usize; N];
                for _ in 0..output.size() {
                    let value = output.get_mut(index).expect("invariant: index in bounds");
                    let offsets = sources
                        .offsets_at(index)
                        .expect("invariant: validated field zip offsets are representable");
                    f(value, sources.values(offsets));
                    next_index(&mut index, &shape);
                }
            }
        }
    }
}

impl<'data, T, U, S, F, const N: usize> MutableZipOutputs<S, F, N>
    for (ArrayViewMut<'data, T, N>, ArrayViewMut<'data, U, N>)
where
    T: Send,
    U: Send,
    S: ZipSources<N>,
    S::Values: Send,
    S::Contiguous: Sync,
    F: Fn((&mut T, &mut U), S::Values) + Send + Sync,
{
    fn zip(mut outputs: Self, sources: S, f: F) {
        assert_eq!(
            outputs.0.shape(),
            outputs.1.shape(),
            "invariant: paired traversal output shapes must match"
        );
        sources
            .validate(outputs.0.shape())
            .expect("invariant: field zip source shapes and storage must match outputs");

        match (
            outputs.0.as_mut_slice(),
            outputs.1.as_mut_slice(),
            sources.contiguous(),
        ) {
            (Some(first), Some(second), Some(source_slices)) => {
                let f_ref = &f;
                for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                    first,
                    second,
                    FIELD_CHUNK_SIZE,
                    |chunk_index, first_chunk, second_chunk| {
                        let base = chunk_index * FIELD_CHUNK_SIZE;
                        for (lane, (first_value, second_value)) in first_chunk
                            .iter_mut()
                            .zip(second_chunk.iter_mut())
                            .enumerate()
                        {
                            f_ref(
                                (first_value, second_value),
                                S::contiguous_values(source_slices, base + lane),
                            );
                        }
                    },
                );
            }
            _ => {
                let shape = outputs.0.shape();
                let mut index = [0usize; N];
                for _ in 0..outputs.0.size() {
                    let first = outputs
                        .0
                        .get_mut(index)
                        .expect("invariant: index in bounds");
                    let second = outputs
                        .1
                        .get_mut(index)
                        .expect("invariant: index in bounds");
                    let offsets = sources
                        .offsets_at(index)
                        .expect("invariant: validated field zip offsets are representable");
                    f((first, second), sources.values(offsets));
                    next_index(&mut index, &shape);
                }
            }
        }
    }
}

/// Mutably map one or more outputs with one or more statically typed source
/// views. Output arity is encoded in the tuple type, so one monomorphized
/// entry point covers one and two mutable outputs without dynamic dispatch or
/// allocation.
pub(crate) fn zip_mut_with<O, S, F, const N: usize>(outputs: O, sources: S, f: F)
where
    O: MutableZipOutputs<S, F, N>,
{
    <O as MutableZipOutputs<S, F, N>>::zip(outputs, sources, f);
}
