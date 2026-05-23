//! MUSIC (Multiple Signal Classification) algorithm.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::math::linear_algebra::EigenDecomposition;
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

/// MUSIC (Multiple Signal Classification) Algorithm
///
/// Exploits eigenstructure of the spatial covariance matrix for super-resolution
/// direction-of-arrival (DOA) estimation.
///
/// # Mathematical Definition
///
/// Given covariance matrix **R** and M sources:
///
/// 1. Eigendecomposition: **R = E Λ E^H**
/// 2. Noise subspace: **E_n** (last N-M eigenvectors)
/// 3. MUSIC pseudospectrum: **P(θ) = 1 / ||E_n^H a(θ)||²**
///
/// # References
///
/// - Schmidt (1986), "Multiple emitter location and signal parameter estimation"
#[derive(Debug, Clone, Copy)]
pub struct MUSIC {
    /// Number of signal sources (signal subspace dimension)
    pub num_sources: usize,
}

impl MUSIC {
    /// Create MUSIC algorithm with specified number of sources.
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self { num_sources }
    }

    /// Compute MUSIC pseudospectrum for a given steering vector.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// P_MUSIC(θ) = 1 / ||E_n^H a(θ)||²
    /// ```
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64> {
        let n = covariance.nrows();

        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(
                "MUSIC::pseudospectrum: covariance must be non-empty square matrix".to_owned(),
            ));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MUSIC::pseudospectrum: steering vector length {} does not match covariance dimension {}",
                steering.len(),
                n
            )));
        }
        if self.num_sources >= n {
            return Err(KwaversError::InvalidInput(format!(
                "MUSIC::pseudospectrum: num_sources {} must be < N {}",
                self.num_sources, n
            )));
        }

        for &val in steering {
            if !val.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::NaN {
                    operation: "MUSIC::pseudospectrum".to_owned(),
                    inputs: "steering vector contains non-finite values".to_owned(),
                }));
            }
        }

        let cov_for_eig = covariance.mapv(|z| num_complex::Complex::new(z.re, z.im));
        let (eigenvalues, eigenvectors) =
            EigenDecomposition::hermitian_eigendecomposition_complex(&cov_for_eig)?;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .total_cmp(&eigenvalues[i])
        });

        let noise_start = self.num_sources;

        let mut en_h_a_norm_sq = 0.0;
        for &idx in indices.iter().skip(noise_start) {
            let eigenvec = eigenvectors.slice(s![.., idx]);

            let mut e_h_a = Complex64::zero();
            for j in 0..n {
                let e_val = eigenvec[j];
                let e_c64 = Complex64::new(e_val.re, e_val.im);
                e_h_a += e_c64.conj() * steering[j];
            }

            en_h_a_norm_sq += e_h_a.norm_sqr();
        }

        if en_h_a_norm_sq < 1e-30 {
            return Ok(1e30);
        }

        let pseudospectrum = 1.0 / en_h_a_norm_sq;

        if !pseudospectrum.is_finite() || pseudospectrum < 0.0 {
            return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                format!(
                    "MUSIC::pseudospectrum: invalid output value {}",
                    pseudospectrum
                ),
            )));
        }

        Ok(pseudospectrum)
    }
}
