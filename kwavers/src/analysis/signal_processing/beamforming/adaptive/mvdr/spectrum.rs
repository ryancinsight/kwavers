use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::MinimumVariance;

impl MinimumVariance {
    /// Compute MVDR pseudospectrum: `P_MVDR(a) = 1 / (a^H R^{-1} a)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if covariance is not square, dimensions mismatch,
    /// solver fails, or denominator is invalid.
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64> {
        let n = covariance.nrows();

        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MVDR pseudospectrum: covariance must be square; got {}×{}",
                n,
                covariance.ncols()
            )));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(
                "MVDR pseudospectrum: steering length mismatch".to_owned(),
            ));
        }

        let r_loaded = self.loaded_covariance(covariance, steering.len())?;

        let y = LinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * y_i)
            .sum();

        let denom_re = denom.re;
        if !denom_re.is_finite() || denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(
                    "MVDR pseudospectrum: invalid denominator".to_owned(),
                ),
            ));
        }

        Ok(1.0 / denom_re)
    }
}
