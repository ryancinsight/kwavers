//! Bilinear interpolation in index space.
//!
//! Single authoritative implementation used by CT preprocessing and any 2-D
//! resampling path in the codebase. Physical-coordinate callers that carry a
//! grid spacing should divide physical coordinates by the spacing before calling.

use leto::Array2;

/// Bilinear interpolation at fractional array indices.
///
/// `x` and `y` are fractional indices into the first and second dimensions of
/// `input`. Out-of-bounds coordinates are clamped to the nearest boundary
/// sample so the function is defined on the closed domain `[0, nx−1] × [0, ny−1]`.
#[must_use]
pub fn bilinear_index_space(input: &Array2<f64>, x: f64, y: f64) -> f64 {
    let (nx, ny) = input.dim();
    let x0 = x.floor().clamp(0.0, (nx - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (ny - 1) as f64) as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let tx = (x - x0 as f64).clamp(0.0, 1.0);
    let ty = (y - y0 as f64).clamp(0.0, 1.0);
    let row0 = input[[x0, y0]].mul_add(1.0 - tx, input[[x1, y0]] * tx);
    let row1 = input[[x0, y1]].mul_add(1.0 - tx, input[[x1, y1]] * tx);
    row0 * (1.0 - ty) + row1 * ty
}
