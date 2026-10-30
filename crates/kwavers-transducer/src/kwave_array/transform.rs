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
}
