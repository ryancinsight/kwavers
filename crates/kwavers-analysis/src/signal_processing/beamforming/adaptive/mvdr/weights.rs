use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::ComplexLinearAlgebra;
use leto::{
    Array1,
    Array2,
};

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
        let r_loaded = self.loaded_covariance(covariance, steering.len())?;

        let y = ComplexLinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * *y_i)
            .sum();

        let denom_re = validate_real_positive_denominator(denom, steering.len(), "MVDR weights")?;

        Ok(y.mapv(|x| x / denom_re))
    }
}
