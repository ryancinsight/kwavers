//! Global array-position transform helpers for [`KWaveArray`].
//!
//! These methods compose the optional global affine transform
//! (installed via `set_array_position`) with per-element poses before
//! rasterization, matching k-wave-python's `kWaveArray.set_array_position`
//! semantics.

use super::{math, KWaveArray};

impl KWaveArray {
    /// Apply the installed global array transform to a point in element-local
    /// coordinates, returning world coordinates. When no transform is set this
    /// is the identity.
    pub(super) fn apply_transform_point(&self, p_local: (f64, f64, f64)) -> (f64, f64, f64) {
        match self.array_transform {
            Some(t) => {
                let r = math::euler_xyz_rotation_matrix(t.euler_xyz_deg);
                let rotated = math::apply_matrix(&r, p_local);
                (
                    rotated.0 + t.translation.0,
                    rotated.1 + t.translation.1,
                    rotated.2 + t.translation.2,
                )
            }
            None => p_local,
        }
    }

    /// Compose the global array transform with a per-element Rect pose,
    /// returning the effective `(position, euler_xyz_deg)` to feed into the
    /// rasterizer. k-wave-python applies the global transform to the rectangle
    /// center only and keeps the per-element Euler unchanged.
    pub(super) fn apply_transform_rect(
        &self,
        position: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) -> ((f64, f64, f64), (f64, f64, f64)) {
        match self.array_transform {
            Some(_) => {
                let pos_eff = self.apply_transform_point(position);
                (pos_eff, euler_xyz_deg)
            }
            None => (position, euler_xyz_deg),
        }
    }
}
