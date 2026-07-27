//! Compatibility traversal helpers for the Yee staggered half.
//!
//! Pending the staggered half SSOT sweep (ADR 0018 follow-up), this module
//! remains kwavers-side and provides:
//!
//! - [`row_major_index`] — pure-math row-major `index = (i * ny + j) * nz + k`
//!   helper shared by [`crate::numerics::operators::differential::staggered_grid`]
//!   impls.
//! - [`write_standard_layout`] — scalar-loop sequential writer used as the
//!   fast path when the input `ArrayView3<f64>` is C-contiguous and exposes a
//!   raw slice. The previous `try_fill_standard_layout(... ) -> bool` always
//!   returned `false`, defeating the C-contiguous branch entirely; after the
//!   ADR 0018 cleanup this helper is a real, no-allocation write into
//!   `dst`. Gated on `is_c_contiguous && as_slice().is_some()` at the call
//!   site because raw-slice indexing requires the row-major index helper.
//!   Pending the staggered half SSOT sweep, this stays scoped to the
//!   staggered half via `pub(super)`.

use leto::Array3;

/// Pure row-major `index = (i * ny + j) * nz + k`.
///
/// `pub(super)` so only the staggered_grid subdir imports it.
pub(super) const fn row_major_index(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    (i * ny + j) * nz + k
}

/// Sequential scalar-loop write into `dst`.
///
/// `dst.shape()` brackets the loop triple. `value_at(i, j, k)` produces the
/// derivative value to record at `(i, j, k)`. Zero heap allocation; no SIMD;
/// no Moirai join; no parallel scan — this is the kwavers-side C-contiguous
/// fast path for the staggered half. Inlined so LLVM can hoist the bound
/// op-passes and fuse the closure body with the surrounding scalar math.
/// The leto-side `zip2_mut_with` slice-pair path remains as the fallback for
/// the non-C-contiguous case (calls below `is_c_contiguous()` block).
/// When the leto-side generic `FiniteDifference3DScheme::StaggeredForward`
/// / `StaggeredBackward` migrations land, this function retires along with
/// `StaggeredGridOperator`.
///
/// # Panics
/// - Panics if `(i, j, k)` returned by `value_at` is outside `dst`'s shape.
///   Callers must clamp the closure's coordinate ranges to `dst`'s shape
///   (e.g. `(0..nx-1, 0..ny, 0..nz)` for the forward axis, `(0..nx, 0..ny, 0..nz)`
///   for the mixed backward axis).
#[inline]
pub(super) fn write_standard_layout<F>(dst: &mut Array3<f64>, value_at: F)
where
    F: Fn(usize, usize, usize) -> f64 + Copy,
{
    let [nx, ny, nz] = dst.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                dst[[i, j, k]] = value_at(i, j, k);
            }
        }
    }
}
