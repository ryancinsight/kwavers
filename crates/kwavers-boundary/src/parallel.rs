//! Provider-owned parallel traversal helpers for boundary arrays.

use moirai_parallel::{enumerate_mut_with, for_each_mut_with, Adaptive};
use ndarray::{ArrayView3, ArrayViewMut3, Zip};

/// Apply an indexed mutation over a 3-D view.
#[inline]
pub(crate) fn for_each_indexed_mut<T, F>(mut values: ArrayViewMut3<'_, T>, f: F)
where
    T: Send,
    F: Fn((usize, usize, usize), &mut T) + Send + Sync,
{
    let (_nx, ny, nz) = values.dim();
    if let Some(slice) = values.as_slice_mut() {
        let f_ref = &f;
        enumerate_mut_with::<Adaptive, _, _>(slice, |idx, value| {
            let plane = ny * nz;
            let i = idx / plane;
            let rem = idx % plane;
            f_ref((i, rem / nz, rem % nz), value);
        });
    } else {
        let f_ref = &f;
        Zip::indexed(values).for_each(f_ref);
    }
}

/// Apply an indexed mutation over paired 3-D views.
#[inline]
pub(crate) fn for_each_indexed_pair_mut<T, U, F>(
    mut values: ArrayViewMut3<'_, T>,
    input: ArrayView3<'_, U>,
    f: F,
) where
    T: Send,
    U: Sync,
    F: Fn((usize, usize, usize), &mut T, &U) + Send + Sync,
{
    debug_assert_eq!(
        values.dim(),
        input.dim(),
        "invariant: boundary paired traversal shape mismatch"
    );

    let (_nx, ny, nz) = values.dim();
    match (values.as_slice_mut(), input.as_slice()) {
        (Some(values), Some(input)) => {
            let f_ref = &f;
            enumerate_mut_with::<Adaptive, _, _>(values, |idx, value| {
                let plane = ny * nz;
                let i = idx / plane;
                let rem = idx % plane;
                f_ref((i, rem / nz, rem % nz), value, &input[idx]);
            });
        }
        _ => {
            let f_ref = &f;
            Zip::indexed(values).and(input).for_each(f_ref);
        }
    }
}

/// Apply an unindexed mutation over any 3-D view.
#[inline]
pub(crate) fn for_each_mut<T, F>(mut values: ArrayViewMut3<'_, T>, f: F)
where
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    if let Some(slice) = values.as_slice_mut() {
        for_each_mut_with::<Adaptive, _, _>(slice, f);
    } else {
        values.iter_mut().for_each(f);
    }
}
