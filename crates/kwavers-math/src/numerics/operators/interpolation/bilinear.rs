//! Bilinear interpolation in index space — thin wrapper over `leto_ops::bilinear_index_space`.
//!
//! Physical-coordinate callers that carry a grid spacing should divide physical
//! coordinates by the spacing before calling.

use leto::Array2;
use leto_ops::bilinear_index_space as bilinear_ssot;

/// Bilinear interpolation at fractional array indices.
///
/// `x` and `y` are fractional indices into the first and second dimensions of
/// `input`.  Delegates to `leto_ops::bilinear_index_space` (SSOT).
#[must_use]
#[inline]
pub fn bilinear_index_space(input: &Array2<f64>, x: f64, y: f64) -> f64 {
    bilinear_ssot(input.view(), x, y)
}
