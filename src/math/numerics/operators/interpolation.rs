//! # Interpolation Operators
//!
//! This module provides spatial interpolation operators for heterogeneous media,
//! sensor data extraction, and grid refinement. All interpolation in kwavers
//! should use these unified implementations.
//!
//! ## Interpolation Methods
//!
//! - **Linear**: 1D linear interpolation (2 points)
//! - **Trilinear**: 3D linear interpolation (8 points)
//! - **Cubic Spline**: Smooth interpolation with continuous derivatives
//! - **Conservative**: Preserves integral quantities (for medium boundaries)
//!
//! ## Mathematical Foundation
//!
//! **Trilinear Interpolation:**
//!
//! For a point (x, y, z) within a grid cell [i, i+1] × [j, j+1] × [k, k+1],
//! the interpolated value is:
//!
//! ```text
//! f(x,y,z) = ∑∑∑ f[i+α, j+β, k+γ] * w_α * w_β * w_γ
//! ```
//!
//! where α, β, γ ∈ {0,1} and w are the linear weights based on distance.
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::{Interpolator, TrilinearInterpolator};
//! use ndarray::{Array1, Array3};
//!
//! let interp = TrilinearInterpolator::new(0.001, 0.001, 0.001);
//! let data = Array3::zeros((100, 100, 100));
//! let target_x = Array1::linspace(0.0, 0.1, 50);
//! let target_y = Array1::linspace(0.0, 0.1, 50);
//! let target_z = Array1::linspace(0.0, 0.1, 50);
//! let result = interp.interpolate_3d(data.view(), target_x.view(), target_y.view(), target_z.view())?;
//! ```
//!
//! ## References
//!
//! - Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*.
//!   Cambridge University Press. Chapter 3: Interpolation and Extrapolation.
//!
//! - de Boor, C. (2001). *A Practical Guide to Splines*. Springer.
//!   DOI: 10.1007/978-1-4612-6333-3

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};

/// Trait for interpolation operators
///
/// This trait defines the interface for all spatial interpolation operators.
/// Implementations must provide methods for 1D and 3D interpolation.
///
/// # Mathematical Properties
///
/// - **Order**: Polynomial order of interpolation
/// - **Continuity**: Degree of smoothness (C⁰, C¹, C², etc.)
/// - **Conservation**: Whether interpolation preserves integrals
/// - **Monotonicity**: Whether interpolation preserves monotonicity
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to enable parallel computation.
pub trait Interpolator: Send + Sync {
    /// Interpolate 1D data at target points
    ///
    /// # Arguments
    ///
    /// * `data` - Source data on regular grid
    /// * `target_points` - Target points for interpolation (physical coordinates)
    ///
    /// # Returns
    ///
    /// Interpolated values at target points
    ///
    /// # Errors
    ///
    /// Returns error if target points are outside data domain
    fn interpolate_1d(
        &self,
        data: ArrayView1<f64>,
        target_points: ArrayView1<f64>,
    ) -> KwaversResult<Array1<f64>>;

    /// Interpolate 3D data at target points
    ///
    /// # Arguments
    ///
    /// * `data` - Source data on regular 3D grid
    /// * `target_x` - Target X coordinates (physical units)
    /// * `target_y` - Target Y coordinates (physical units)
    /// * `target_z` - Target Z coordinates (physical units)
    ///
    /// # Returns
    ///
    /// Interpolated 3D field at target points
    ///
    /// # Errors
    ///
    /// Returns error if dimensions mismatch or points outside domain
    fn interpolate_3d(
        &self,
        data: ArrayView3<f64>,
        target_x: ArrayView1<f64>,
        target_y: ArrayView1<f64>,
        target_z: ArrayView1<f64>,
    ) -> KwaversResult<Array3<f64>>;

    /// Get interpolation order
    ///
    /// # Returns
    ///
    /// Polynomial order (1 for linear, 3 for cubic, etc.)
    fn order(&self) -> usize;

    /// Check if interpolation is conservative
    ///
    /// Conservative interpolation preserves integral quantities
    fn is_conservative(&self) -> bool {
        false
    }

    /// Check if interpolation is monotonic
    ///
    /// Monotonic interpolation preserves monotonicity of data
    fn is_monotonic(&self) -> bool {
        false
    }
}

/// Linear interpolator for 1D data
///
/// Performs piecewise linear interpolation between grid points.
///
/// # Mathematical Form
///
/// For x ∈ [x_i, x_{i+1}]:
/// ```text
/// f(x) = f_i + (f_{i+1} - f_i) * (x - x_i) / (x_{i+1} - x_i)
/// ```
///
/// # Properties
///
/// - Order: 1 (linear)
/// - Continuity: C⁰ (continuous but not smooth)
/// - Monotonicity: Preserved
#[derive(Debug, Clone)]
pub struct LinearInterpolator {
    /// Grid spacing (meters)
    dx: f64,
}

impl LinearInterpolator {
    /// Create a new linear interpolator
    ///
    /// # Arguments
    ///
    /// * `dx` - Grid spacing (meters)
    ///
    /// # Returns
    ///
    /// New interpolator instance
    pub fn new(dx: f64) -> Self {
        Self { dx }
    }
}

impl Interpolator for LinearInterpolator {
    fn interpolate_1d(
        &self,
        data: ArrayView1<f64>,
        target_points: ArrayView1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let n = data.len();
        let mut result = Array1::zeros(target_points.len());

        for (idx, &x) in target_points.iter().enumerate() {
            // Find grid index
            let i_float = x / self.dx;
            let i = i_float.floor() as usize;

            if i >= n - 1 {
                return Err(NumericalError::InterpolationOutOfBounds {
                    point: x,
                    min: 0.0,
                    max: ((n - 1) as f64) * self.dx,
                }
                .into());
            }

            // Linear interpolation weight
            let t = i_float - (i as f64);
            result[idx] = data[i] * (1.0 - t) + data[i + 1] * t;
        }

        Ok(result)
    }

    fn interpolate_3d(
        &self,
        data: ArrayView3<f64>,
        target_x: ArrayView1<f64>,
        target_y: ArrayView1<f64>,
        target_z: ArrayView1<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // For LinearInterpolator, delegate to TrilinearInterpolator
        let trilinear = TrilinearInterpolator::new(self.dx, self.dx, self.dx);
        trilinear.interpolate_3d(data, target_x, target_y, target_z)
    }

    fn order(&self) -> usize {
        1
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

/// Trilinear interpolator for 3D data
///
/// Performs trilinear interpolation within a 3D grid cell using 8 surrounding
/// points. This is the standard method for heterogeneous medium properties.
///
/// # Mathematical Form
///
/// For a point (x, y, z) in cell [i,i+1] × [j,j+1] × [k,k+1]:
///
/// ```text
/// f(x,y,z) = (1-tx)(1-ty)(1-tz) f[i,j,k]
///          + tx(1-ty)(1-tz) f[i+1,j,k]
///          + (1-tx)ty(1-tz) f[i,j+1,k]
///          + ... (8 terms total)
/// ```
///
/// where tx, ty, tz are the normalized distances within the cell.
///
/// # Properties
///
/// - Order: 1 (trilinear = linear in each direction)
/// - Continuity: C⁰
/// - Monotonicity: Preserved
///
/// # References
///
/// - Numerical Recipes, Chapter 3.6: Interpolation in Two or More Dimensions
#[derive(Debug, Clone)]
pub struct TrilinearInterpolator {
    /// Grid spacing in X direction (meters)
    dx: f64,
    /// Grid spacing in Y direction (meters)
    dy: f64,
    /// Grid spacing in Z direction (meters)
    dz: f64,
}

impl TrilinearInterpolator {
    /// Create a new trilinear interpolator
    ///
    /// # Arguments
    ///
    /// * `dx` - Grid spacing in X direction (meters)
    /// * `dy` - Grid spacing in Y direction (meters)
    /// * `dz` - Grid spacing in Z direction (meters)
    ///
    /// # Returns
    ///
    /// New interpolator instance
    pub fn new(dx: f64, dy: f64, dz: f64) -> Self {
        Self { dx, dy, dz }
    }

    /// Interpolate at a single 3D point
    ///
    /// # Arguments
    ///
    /// * `data` - Source 3D field
    /// * `x` - Target X coordinate (meters)
    /// * `y` - Target Y coordinate (meters)
    /// * `z` - Target Z coordinate (meters)
    ///
    /// # Returns
    ///
    /// Interpolated value at (x, y, z)
    pub fn interpolate_point(
        &self,
        data: ArrayView3<f64>,
        x: f64,
        y: f64,
        z: f64,
    ) -> KwaversResult<f64> {
        let (nx, ny, nz) = data.dim();

        // Find grid indices (floating point)
        let i_float = x / self.dx;
        let j_float = y / self.dy;
        let k_float = z / self.dz;

        // Integer indices
        let i = i_float.floor() as usize;
        let j = j_float.floor() as usize;
        let k = k_float.floor() as usize;

        // Check bounds
        if i >= nx - 1 || j >= ny - 1 || k >= nz - 1 {
            return Err(NumericalError::InterpolationOutOfBounds {
                point: (x.powi(2) + y.powi(2) + z.powi(2)).sqrt(),
                min: 0.0,
                max: ((nx - 1) as f64 * self.dx)
                    .max((ny - 1) as f64 * self.dy)
                    .max((nz - 1) as f64 * self.dz),
            }
            .into());
        }

        // Normalized distances within cell [0, 1]
        let tx = i_float - (i as f64);
        let ty = j_float - (j as f64);
        let tz = k_float - (k as f64);

        // Trilinear interpolation weights
        let w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz);
        let w100 = tx * (1.0 - ty) * (1.0 - tz);
        let w010 = (1.0 - tx) * ty * (1.0 - tz);
        let w110 = tx * ty * (1.0 - tz);
        let w001 = (1.0 - tx) * (1.0 - ty) * tz;
        let w101 = tx * (1.0 - ty) * tz;
        let w011 = (1.0 - tx) * ty * tz;
        let w111 = tx * ty * tz;

        // Interpolated value (8-point stencil)
        let value = w000 * data[[i, j, k]]
            + w100 * data[[i + 1, j, k]]
            + w010 * data[[i, j + 1, k]]
            + w110 * data[[i + 1, j + 1, k]]
            + w001 * data[[i, j, k + 1]]
            + w101 * data[[i + 1, j, k + 1]]
            + w011 * data[[i, j + 1, k + 1]]
            + w111 * data[[i + 1, j + 1, k + 1]];

        Ok(value)
    }
}

impl Interpolator for TrilinearInterpolator {
    fn interpolate_1d(
        &self,
        data: ArrayView1<f64>,
        target_points: ArrayView1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let n = data.len();
        let mut result = Array1::zeros(target_points.len());

        for (idx, &x) in target_points.iter().enumerate() {
            let i_float = x / self.dx;
            let i = i_float.floor() as usize;

            if i >= n - 1 {
                return Err(NumericalError::InterpolationOutOfBounds {
                    point: x,
                    min: 0.0,
                    max: ((n - 1) as f64) * self.dx,
                }
                .into());
            }

            let t = i_float - (i as f64);
            result[idx] = data[i] * (1.0 - t) + data[i + 1] * t;
        }

        Ok(result)
    }

    fn interpolate_3d(
        &self,
        data: ArrayView3<f64>,
        target_x: ArrayView1<f64>,
        target_y: ArrayView1<f64>,
        target_z: ArrayView1<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let nx_target = target_x.len();
        let ny_target = target_y.len();
        let nz_target = target_z.len();

        let mut result = Array3::zeros((nx_target, ny_target, nz_target));

        for i in 0..nx_target {
            for j in 0..ny_target {
                for k in 0..nz_target {
                    let x = target_x[i];
                    let y = target_y[j];
                    let z = target_z[k];
                    result[[i, j, k]] = self.interpolate_point(data, x, y, z)?;
                }
            }
        }

        Ok(result)
    }

    fn order(&self) -> usize {
        1
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_interpolator_simple() {
        let dx = 0.1;
        let interp = LinearInterpolator::new(dx);

        // Data: f(x) = 2x at grid points
        let data = Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8]);

        // Interpolate at midpoint
        let target = Array1::from_vec(vec![0.05]); // Midpoint between 0.0 and 0.1
        let result = interp.interpolate_1d(data.view(), target.view()).unwrap();

        // Expected: linear interpolation gives f(0.05) = 0.1
        assert_abs_diff_eq!(result[0], 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_interpolator_exact_at_grid_points() {
        let dx = 1.0;
        let interp = LinearInterpolator::new(dx);

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Interpolate exactly at grid point
        let target = Array1::from_vec(vec![2.0]);
        let result = interp.interpolate_1d(data.view(), target.view()).unwrap();

        assert_abs_diff_eq!(result[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trilinear_constant_field() {
        let dx = 0.1;
        let interp = TrilinearInterpolator::new(dx, dx, dx);

        // Constant field
        let data = Array3::from_elem((5, 5, 5), 10.0);

        // Interpolate anywhere should give constant value
        let x = 0.25;
        let y = 0.15;
        let z = 0.35;
        let result = interp.interpolate_point(data.view(), x, y, z).unwrap();

        assert_abs_diff_eq!(result, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trilinear_linear_function() {
        // Test on linear function: f(x,y,z) = x + 2y + 3z
        // Trilinear should be exact for linear functions
        let dx = 1.0;
        let interp = TrilinearInterpolator::new(dx, dx, dx);

        let mut data = Array3::zeros((4, 4, 4));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    data[[i, j, k]] = (i as f64) + 2.0 * (j as f64) + 3.0 * (k as f64);
                }
            }
        }

        // Test at non-grid point
        let x = 1.5;
        let y = 2.3;
        let z = 1.7;
        let result = interp.interpolate_point(data.view(), x, y, z).unwrap();

        let expected = x + 2.0 * y + 3.0 * z;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_trilinear_at_corner() {
        let dx = 0.1;
        let interp = TrilinearInterpolator::new(dx, dx, dx);

        let mut data = Array3::zeros((3, 3, 3));
        data[[1, 1, 1]] = 5.0;

        // Interpolate exactly at grid point
        let result = interp
            .interpolate_point(data.view(), 0.1, 0.1, 0.1)
            .unwrap();

        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolation_out_of_bounds() {
        let dx = 0.1;
        let interp = TrilinearInterpolator::new(dx, dx, dx);

        let data = Array3::zeros((5, 5, 5));

        // Point outside domain
        let result = interp.interpolate_point(data.view(), 1.0, 0.0, 0.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_trilinear_3d_batch() {
        let dx = 1.0;
        let interp = TrilinearInterpolator::new(dx, dx, dx);

        // Simple 2x2x2 grid
        let data = Array3::from_shape_vec((2, 2, 2), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();

        // Target grid (just one point for simplicity)
        let target_x = Array1::from_vec(vec![0.5]);
        let target_y = Array1::from_vec(vec![0.5]);
        let target_z = Array1::from_vec(vec![0.5]);

        let result = interp
            .interpolate_3d(
                data.view(),
                target_x.view(),
                target_y.view(),
                target_z.view(),
            )
            .unwrap();

        // Center of cube: average of 8 corners
        let expected = (0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0) / 8.0;
        assert_abs_diff_eq!(result[[0, 0, 0]], expected, epsilon = 1e-10);
    }
}
