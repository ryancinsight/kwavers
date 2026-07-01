use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::linear_algebra::ComplexLinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::{validate_real_positive_denominator, MinimumVariance};

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

        let y = ComplexLinearAlgebra::solve_linear_system_complex(&r_loaded, steering)?;

        let denom: Complex64 = steering
            .iter()
            .zip(y.iter())
            .map(|(a, y_i)| a.conj() * y_i)
            .sum();

        let denom_re =
            validate_real_positive_denominator(denom, steering.len(), "MVDR pseudospectrum")?;

        Ok(1.0 / denom_re)
    }
}
