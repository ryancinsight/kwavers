//! Interpolation operator trait.

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};

/// Trait for all spatial interpolation operators.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to enable parallel computation.
pub trait Interpolator: Send + Sync {
    /// Interpolate 1D data at target points.
    fn interpolate_1d(
        &self,
        data: ArrayView1<f64>,
        target_points: ArrayView1<f64>,
    ) -> KwaversResult<Array1<f64>>;

    /// Interpolate 3D data at target points.
    fn interpolate_3d(
        &self,
        data: ArrayView3<f64>,
        target_x: ArrayView1<f64>,
        target_y: ArrayView1<f64>,
        target_z: ArrayView1<f64>,
    ) -> KwaversResult<Array3<f64>>;

    /// Polynomial order of interpolation.
    fn order(&self) -> usize;

    /// Whether interpolation preserves integral quantities.
    fn is_conservative(&self) -> bool {
        false
    }

    /// Whether interpolation preserves monotonicity of data.
    fn is_monotonic(&self) -> bool {
        false
    }
}
