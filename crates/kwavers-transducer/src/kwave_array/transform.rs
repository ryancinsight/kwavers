//! Global array-position transform helpers for [`KWaveArray`].
//!
//! These methods apply the optional global affine transform
//! (installed via `set_array_position`) to element geometry before
//! rasterization. `SurfaceConvention::KWave` elements follow k-wave-python's
//! `kWaveArray.set_array_position` (centre only); `AreaConserving` elements
//! take the full rigid transform (ADR 136).

use super::{math, KWaveArray, SurfaceConvention};
use crate::transducers::physics::PlanarApertureGeometry;

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

    /// Planar-element geometry after the installed array transform.
    ///
    /// The centre always takes the full transform. The orientation follows the
    /// element's [`SurfaceConvention`]: `KWave` keeps the normal and first axis
    /// (k-wave-python applies the affine to a rectangle centre only), and
    /// `AreaConserving` rotates both by the global rotation with no
    /// translation. Without an installed transform the geometry is unchanged.
    pub(super) fn apply_transform_planar(
        &self,
        geometry: PlanarApertureGeometry,
        convention: SurfaceConvention,
    ) -> PlanarApertureGeometry {
        let Some(transform) = self.array_transform else {
            return geometry;
        };
        let [cx, cy, cz] = geometry.center_m();
        let (x, y, z) = self.apply_transform_point((cx, cy, cz));
        let (normal, first_axis) = match convention {
            SurfaceConvention::KWave => (geometry.normal(), geometry.first_axis()),
            SurfaceConvention::AreaConserving => {
                let rotation = math::euler_xyz_rotation_matrix(transform.euler_xyz_deg);
                let rotate = |axis: [f64; 3]| {
                    let (x, y, z) = math::apply_matrix(&rotation, (axis[0], axis[1], axis[2]));
                    [x, y, z]
                };
                (rotate(geometry.normal()), rotate(geometry.first_axis()))
            }
        };
        geometry.with_frame([x, y, z], normal, first_axis)
    }
}
