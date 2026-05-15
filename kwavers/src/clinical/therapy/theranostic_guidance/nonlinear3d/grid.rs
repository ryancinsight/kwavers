//! Integer grid-index arithmetic for the 3-D nonlinear solver.
//!
//! Extracted from `types` so that geometry concerns are isolated from
//! the configuration and result types that live in that file.  All
//! consumers that previously imported these items via `super::types`
//! continue to do so — `types` re-exports them unchanged.

use super::super::Point3;

/// Integer 3-D grid-cell address on a uniform cubic lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GridIndex {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

/// Row-major flat index: `(x * n + y) * n + z`.
///
/// Inverse: [`grid_index`](crate::clinical::therapy::theranostic_guidance::nonlinear3d::cavitation)
/// in `cavitation.rs`.
#[must_use]
pub(crate) fn flat_index(idx: GridIndex, n: usize) -> usize {
    (idx.x * n + idx.y) * n + idx.z
}

/// Physical centred-lattice position of `idx` in metres.
///
/// The lattice centre is at `((n-1)/2, (n-1)/2, (n-1)/2)` so that the
/// origin of the coordinate system coincides with the geometric centre of
/// the cubic volume — matching the convention used throughout the aperture
/// and waveform modules.
#[must_use]
pub(crate) fn grid_point_m(idx: GridIndex, n: usize, spacing_m: f64) -> Point3 {
    let center = 0.5 * (n - 1) as f64;
    Point3 {
        x_m: (idx.x as f64 - center) * spacing_m,
        y_m: (idx.y as f64 - center) * spacing_m,
        z_m: (idx.z as f64 - center) * spacing_m,
    }
}
