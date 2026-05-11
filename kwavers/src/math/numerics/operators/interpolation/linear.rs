//! Linear interpolator for 1D data.
//!
//! ## Mathematical Form
//!
//! For x ∈ [x_i, x_{i+1}]:
//! ```text
//! f(x) = f_i + (f_{i+1} - f_i) * (x - x_i) / (x_{i+1} - x_i)
//! ```
//!
//! ## References
//!
//! - Press et al. (2007). *Numerical Recipes*. Chapter 3.

use super::traits::Interpolator;
use super::trilinear::TrilinearInterpolator;
use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};

/// Piecewise linear interpolator (C⁰, order 1, monotonicity-preserving).
#[derive(Debug, Clone)]
pub struct LinearInterpolator {
    dx: f64,
}

impl LinearInterpolator {
    /// Create a new linear interpolator with grid spacing `dx` (meters).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
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
            result[idx] = data[i].mul_add(1.0 - t, data[i + 1] * t);
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
        TrilinearInterpolator::new(self.dx, self.dx, self.dx)
            .interpolate_3d(data, target_x, target_y, target_z)
    }

    fn order(&self) -> usize {
        1
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}
