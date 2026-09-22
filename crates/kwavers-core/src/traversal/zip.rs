//! Parallel lockstep traversals writing one, two or three fields.
//!
//! Each output count has one implementation, the indexed form; the plain form
//! wraps it with a closure that drops the index. The index is plain
//! arithmetic with no effect of its own, so once the closure inlines into the
//! unit loop the dropped index is dead code and costs nothing.

use leto::ArrayViewMut;
use moirai_parallel::{
    for_each_unit_task_many_mut_with, for_each_unit_task_mut_with,
    for_each_unit_task_pair_mut_with, for_each_unit_task_triple_mut_with, Adaptive,
};

use super::inputs::ZipInputs;

const ONCE: &str = "invariant: a mutable view addresses each element once";
const RUNS: &str = "invariant: every destination holds the same element count";

/// Apply `f` to every element of `out` beside the matching elements of
/// `inputs`.
///
/// Elements pair by logical row-major position. C-contiguous fields run as
/// moirai unit tasks sized by the bytes one element moves; any other layout
/// runs the logical walk on the calling thread.
///
/// # Panics
///
/// Panics if an input's shape differs from `out`'s.
#[inline]
#[track_caller]
pub fn zip_mut<'a, T, I, const N: usize, F>(out: ArrayViewMut<'_, T, N>, inputs: I, f: F)
where
    T: Send,
    I: ZipInputs<'a, N>,
    F: Fn(&mut T, I::Refs) + Send + Sync,
{
    zip_mut_indexed(out, inputs, |_, value, refs| f(value, refs));
}

/// [`zip_mut`] that also hands `f` each element's logical index.
///
/// # Panics
///
/// Panics if an input's shape differs from `out`'s.
#[track_caller]
pub fn zip_mut_indexed<'a, T, I, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    inputs: I,
    f: F,
) where
    T: Send,
    I: ZipInputs<'a, N>,
    F: Fn([usize; N], &mut T, I::Refs) + Send + Sync,
{
    let shape = out.shape();
    inputs.assert_shape(shape);
    if let (Some(slices), Some(out)) = (inputs.slices(), out.as_mut_slice()) {
        for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
            out,
            1,
            size_of::<T>() + I::UNIT_BYTES,
            || (),
            |(), first, run| {
                let mut index = row_major_index(first, shape);
                for (position, value) in (first..).zip(run) {
                    f(index, value, I::at(slices, position));
                    advance(&mut index, shape);
                }
            },
        );
        return;
    }
    for ((index, value), refs) in out.indexed_iter_mut().expect(ONCE).zip(inputs.logical()) {
        f(index, value, refs);
    }
}

/// [`zip_mut`] writing two fields of one shape.
///
/// # Panics
///
/// Panics if any field's shape differs from `first_out`'s.
#[inline]
#[track_caller]
pub fn zip_mut_pair<'a, T, U, I, const N: usize, F>(
    first_out: ArrayViewMut<'_, T, N>,
    second_out: ArrayViewMut<'_, U, N>,
    inputs: I,
    f: F,
) where
    T: Send,
    U: Send,
    I: ZipInputs<'a, N>,
    F: Fn(&mut T, &mut U, I::Refs) + Send + Sync,
{
    zip_mut_pair_indexed(first_out, second_out, inputs, |_, a, b, refs| f(a, b, refs));
}

/// [`zip_mut_pair`] that also hands `f` each element's logical index.
///
/// # Panics
///
/// Panics if any field's shape differs from `first_out`'s.
#[track_caller]
pub fn zip_mut_pair_indexed<'a, T, U, I, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    inputs: I,
    f: F,
) where
    T: Send,
    U: Send,
    I: ZipInputs<'a, N>,
    F: Fn([usize; N], &mut T, &mut U, I::Refs) + Send + Sync,
{
    let shape = first_out.shape();
    assert_same_shape(second_out.shape(), shape);
    inputs.assert_shape(shape);
    if let (Some(slices), Some(first_out), Some(second_out)) = (
        inputs.slices(),
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
    ) {
        for_each_unit_task_pair_mut_with::<Adaptive, _, _, _, _, _>(
            first_out,
            second_out,
            1,
            size_of::<T>() + size_of::<U>() + I::UNIT_BYTES,
            || (),
            |(), first, first_run, second_run| {
                let mut index = row_major_index(first, shape);
                for (position, (a, b)) in (first..).zip(first_run.iter_mut().zip(second_run)) {
                    f(index, a, b, I::at(slices, position));
                    advance(&mut index, shape);
                }
            },
        );
        return;
    }
    for (((index, a), b), refs) in first_out
        .indexed_iter_mut()
        .expect(ONCE)
        .zip(second_out.try_iter_mut().expect(ONCE))
        .zip(inputs.logical())
    {
        f(index, a, b, refs);
    }
}

/// [`zip_mut`] writing three fields of one shape.
///
/// # Panics
///
/// Panics if any field's shape differs from `first_out`'s.
#[inline]
#[track_caller]
pub fn zip_mut_triple<'a, T, U, V, I, const N: usize, F>(
    first_out: ArrayViewMut<'_, T, N>,
    second_out: ArrayViewMut<'_, U, N>,
    third_out: ArrayViewMut<'_, V, N>,
    inputs: I,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    I: ZipInputs<'a, N>,
    F: Fn(&mut T, &mut U, &mut V, I::Refs) + Send + Sync,
{
    zip_mut_triple_indexed(
        first_out,
        second_out,
        third_out,
        inputs,
        |_, a, b, c, refs| f(a, b, c, refs),
    );
}

/// [`zip_mut_triple`] that also hands `f` each element's logical index.
///
/// # Panics
///
/// Panics if any field's shape differs from `first_out`'s.
#[track_caller]
pub fn zip_mut_triple_indexed<'a, T, U, V, I, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    mut third_out: ArrayViewMut<'_, V, N>,
    inputs: I,
    f: F,
) where
    T: Send,
    U: Send,
    V: Send,
    I: ZipInputs<'a, N>,
    F: Fn([usize; N], &mut T, &mut U, &mut V, I::Refs) + Send + Sync,
{
    let shape = first_out.shape();
    assert_same_shape(second_out.shape(), shape);
    assert_same_shape(third_out.shape(), shape);
    inputs.assert_shape(shape);
    if let (Some(slices), Some(first_out), Some(second_out), Some(third_out)) = (
        inputs.slices(),
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        third_out.as_mut_slice(),
    ) {
        for_each_unit_task_triple_mut_with::<Adaptive, _, _, _, _, _, _>(
            first_out,
            second_out,
            third_out,
            1,
            size_of::<T>() + size_of::<U>() + size_of::<V>() + I::UNIT_BYTES,
            || (),
            |(), first, first_run, second_run, third_run| {
                let mut index = row_major_index(first, shape);
                for (position, ((a, b), c)) in
                    (first..).zip(first_run.iter_mut().zip(second_run).zip(third_run))
                {
                    f(index, a, b, c, I::at(slices, position));
                    advance(&mut index, shape);
                }
            },
        );
        return;
    }
    for ((((index, a), b), c), refs) in first_out
        .indexed_iter_mut()
        .expect(ONCE)
        .zip(second_out.try_iter_mut().expect(ONCE))
        .zip(third_out.try_iter_mut().expect(ONCE))
        .zip(inputs.logical())
    {
        f(index, a, b, c, refs);
    }
}

/// [`zip_mut`] writing `K` fields of one shape and one type.
///
/// The pair and triple forms take their destinations by separate type, which
/// suits two or three fields that differ. A pass writing more than three
/// fields of one type -- the six a velocity-Verlet kick and drift write --
/// takes them as one array instead, and stays one parallel region where a
/// call per destination would be `K` of them.
///
/// # Panics
///
/// Panics if any field's shape differs from the first destination's.
#[inline]
#[track_caller]
pub fn zip_mut_many<'a, T, I, const N: usize, const K: usize, F>(
    outs: [ArrayViewMut<'_, T, N>; K],
    inputs: I,
    f: F,
) where
    T: Send,
    I: ZipInputs<'a, N>,
    F: Fn([&mut T; K], I::Refs) + Send + Sync,
{
    zip_mut_many_indexed(outs, inputs, |_, values, refs| f(values, refs));
}

/// [`zip_mut_many`] that also hands `f` each element's logical index.
///
/// # Panics
///
/// Panics if any field's shape differs from the first destination's.
#[track_caller]
pub fn zip_mut_many_indexed<'a, T, I, const N: usize, const K: usize, F>(
    mut outs: [ArrayViewMut<'_, T, N>; K],
    inputs: I,
    f: F,
) where
    T: Send,
    I: ZipInputs<'a, N>,
    F: Fn([usize; N], [&mut T; K], I::Refs) + Send + Sync,
{
    let Some(first_out) = outs.first() else {
        return;
    };
    let shape = first_out.shape();
    for out in &outs {
        assert_same_shape(out.shape(), shape);
    }
    inputs.assert_shape(shape);
    if outs.iter().all(ArrayViewMut::is_c_dense) {
        let borrowed = outs.each_mut().map(ArrayViewMut::as_mut_slice);
        if let (true, Some(slices)) = (borrowed.iter().all(Option::is_some), inputs.slices()) {
            let runs: [&mut [T]; K] = borrowed.map(|slice| slice.expect(ONCE));
            for_each_unit_task_many_mut_with::<Adaptive, _, _, _, _, K>(
                runs,
                1,
                K * size_of::<T>() + I::UNIT_BYTES,
                || (),
                |(), first, chunk| {
                    let mut index = row_major_index(first, shape);
                    let len = chunk[0].len();
                    // One cursor per run: the runs are distinct buffers, so
                    // walking them in lockstep hands out `K` references that
                    // cannot alias, which indexing one array of runs per
                    // element could not express.
                    let mut cursors = chunk.map(<[T]>::iter_mut);
                    for position in first..first + len {
                        let values: [&mut T; K] =
                            core::array::from_fn(|field| cursors[field].next().expect(RUNS));
                        f(index, values, I::at(slices, position));
                        advance(&mut index, shape);
                    }
                },
            );
            return;
        }
    }
    let mut cursors = outs.map(|out| out.try_iter_mut().expect(ONCE));
    let mut index = row_major_index(0, shape);
    for refs in inputs.logical() {
        let values: [&mut T; K] = core::array::from_fn(|field| cursors[field].next().expect(RUNS));
        f(index, values, refs);
        advance(&mut index, shape);
    }
}

#[inline]
#[track_caller]
fn assert_same_shape<const N: usize>(shape: [usize; N], expected: [usize; N]) {
    assert_eq!(
        shape, expected,
        "invariant: every written field has one shape"
    );
}

/// The logical index of flat row-major position `position` in `shape`.
#[inline]
fn row_major_index<const N: usize>(mut position: usize, shape: [usize; N]) -> [usize; N] {
    let mut index = [0; N];
    for (slot, &extent) in index.iter_mut().zip(&shape).rev() {
        *slot = position % extent;
        position /= extent;
    }
    index
}

/// Step `index` to the next row-major position; the last position wraps to
/// the origin, which no caller reads.
#[inline]
fn advance<const N: usize>(index: &mut [usize; N], shape: [usize; N]) {
    for (slot, &extent) in index.iter_mut().zip(&shape).rev() {
        *slot += 1;
        if *slot < extent {
            return;
        }
        *slot = 0;
    }
}
