//! Trilinear interpolator for 3D data.
//!
//! ## Mathematical Form
//!
//! For a point (x, y, z) in cell [i,i+1] × [j,j+1] × [k,k+1]:
//!
//! ```text
//! f(x,y,z) = (1-tx)(1-ty)(1-tz) f[i,j,k]
//!          + tx(1-ty)(1-tz) f[i+1,j,k]
//!          + ... (8 terms total)
//! ```
//!
//! ## References
//!
//! - Numerical Recipes, Chapter 3.6

use super::traits::Interpolator;
use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};

/// Trilinear interpolator (C⁰, order 1, monotonicity-preserving).
#[derive(Debug, Clone)]
pub struct TrilinearInterpolator {
    dx: f64,
    dy: f64,
    dz: f64,
}

impl TrilinearInterpolator {
    /// Create a new trilinear interpolator with grid spacings `dx`, `dy`, `dz` (meters).
    pub fn new(dx: f64, dy: f64, dz: f64) -> Self {
        Self { dx, dy, dz }
    }

    /// Interpolate at a single 3D point (x, y, z) in physical coordinates.
    pub fn interpolate_point(
        &self,
        data: ArrayView3<f64>,
        x: f64,
        y: f64,
        z: f64,
    ) -> KwaversResult<f64> {
        let (nx, ny, nz) = data.dim();

        let i_float = x / self.dx;
        let j_float = y / self.dy;
        let k_float = z / self.dz;

        let i = i_float.floor() as usize;
        let j = j_float.floor() as usize;
        let k = k_float.floor() as usize;

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

        let tx = i_float - (i as f64);
        let ty = j_float - (j as f64);
        let tz = k_float - (k as f64);

        let w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz);
        let w100 = tx * (1.0 - ty) * (1.0 - tz);
        let w010 = (1.0 - tx) * ty * (1.0 - tz);
        let w110 = tx * ty * (1.0 - tz);
        let w001 = (1.0 - tx) * (1.0 - ty) * tz;
        let w101 = tx * (1.0 - ty) * tz;
        let w011 = (1.0 - tx) * ty * tz;
        let w111 = tx * ty * tz;

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
                    result[[i, j, k]] =
                        self.interpolate_point(data, target_x[i], target_y[j], target_z[k])?;
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
