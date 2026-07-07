use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::ComplexLinearAlgebra;
use leto::{Array1 as LetoArray1, Array2 as LetoArray2};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::{validate_real_positive_denominator, MinimumVariance};

impl MinimumVariance {
    /// Compute MVDR beamforming weights.
    ///
    /// Algorithm:
    /// 1. `R_loaded = R + δI`
    /// 2. Solve `R_loaded y = a` (SSOT complex solver)
    /// 3. `denom = a^H y`
    /// 4. `w = y / denom`
    ///
    /// # Errors
    ///
    /// Returns `Err` if covariance is not square, dimensions mismatch, loading
    /// is invalid, solver fails, or denominator is non-finite/non-positive.
    pub fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = steering.len();
        let r_loaded = self.loaded_covariance(covariance, steering.len())?;
        let r_loaded_leto = LetoArray2::from_shape_fn([n, n], |[i, j]| r_loaded[(i, j)]);
        let steering_leto = LetoArray1::from_shape_fn([n], |[i]| steering[i]);
        let y = ComplexLinearAlgebra::solve_linear_system_complex(&r_loaded_leto, &steering_leto)?;

        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * y_i)
            .sum();

        let denom_re = validate_real_positive_denominator(denom, steering.len(), "MVDR weights")?;

        Ok(Array1::from_iter(y.iter().map(|x| *x / denom_re)))
    }
}
