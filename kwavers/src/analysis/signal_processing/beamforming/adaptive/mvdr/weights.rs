use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::MinimumVariance;

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

        let y = LinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * y_i)
            .sum();

        let denom_re = denom.re;
        if !denom_re.is_finite() {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(
                    "MVDR: non-finite denominator a^H R^{-1} a (covariance may be ill-conditioned)"
                        .to_string(),
                ),
            ));
        }
        if denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(format!(
                    "MVDR: non-positive denominator a^H R^{{-1}} a = {} (covariance may not be Hermitian PD)",
                    denom_re
                )),
            ));
        }

        Ok(y.mapv(|x| x / denom_re))
    }
}
