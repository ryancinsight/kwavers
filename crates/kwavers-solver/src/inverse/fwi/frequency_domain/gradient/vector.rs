//! Scalar field reductions use the kwavers-math numerical contract.

use kwavers_math::linear_algebra::NumericOps;
use leto::Array3;

/// Inner product `⟨a, b⟩` of two fields.
///
/// The reduction is [`NumericOps::dot_product`]'s, the kwavers-math scalar
/// SSOT; this function is only the `Array3` → slice bridge. A non-contiguous
/// view has no borrowed slice, so it keeps the element walk — the same
/// left fold over the same C-order elements.
#[must_use]
pub(in super::super) fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    match (a.as_slice(), b.as_slice()) {
        (Some(a), Some(b)) => {
            f64::dot_product(a, b).expect("invariant: two fields of one shape share a length")
        }
        _ => a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum(),
    }
}

#[must_use]
pub(in super::super) fn max_abs(a: &Array3<f64>) -> f64 {
    a.iter().map(|value| value.abs()).fold(0.0, f64::max)
}
