//! Provider-owned parallel traversal helpers for boundary arrays.

use leto::{ArrayView3, ArrayViewMut3};

/// Apply an indexed mutation over a 3-D view.
#[inline]
pub(crate) fn for_each_indexed_mut<T, F>(mut values: ArrayViewMut3<'_, T>, f: F)
where
    T: Send,
    F: Fn((usize, usize, usize), &mut T) + Send + Sync,
{
    let [nx, ny, nz] = values.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let value = values
                    .get_mut([i, j, k])
                    .expect("invariant: valid 3-D boundary traversal index");
                f((i, j, k), value);
            }
        }
    }
}

/// Apply an indexed mutation over paired 3-D views.
#[inline]
pub(crate) fn for_each_indexed_pair_mut<T, U, F>(
    mut values: ArrayViewMut3<'_, T>,
    input: ArrayView3<'_, U>,
    f: F,
) where
    F: Fn((usize, usize, usize), &mut T, &U) + Send + Sync,
{
    debug_assert_eq!(
        values.shape(),
        input.shape(),
        "invariant: boundary paired traversal shape mismatch"
    );

    let [nx, ny, nz] = values.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rhs = input
                    .get([i, j, k])
                    .expect("invariant: valid paired boundary traversal input index");
                let lhs = values
                    .get_mut([i, j, k])
                    .expect("invariant: valid paired boundary traversal output index");
                f((i, j, k), lhs, rhs);
            }
        }
    }
}

/// Apply an unindexed mutation over any 3-D view.
#[inline]
pub(crate) fn for_each_mut<T, F>(mut values: ArrayViewMut3<'_, T>, f: F)
where
    F: Fn(&mut T) + Send + Sync,
{
    let [nx, ny, nz] = values.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let value = values
                    .get_mut([i, j, k])
                    .expect("invariant: valid unindexed boundary traversal index");
                f(value);
            }
        }
    }
}
