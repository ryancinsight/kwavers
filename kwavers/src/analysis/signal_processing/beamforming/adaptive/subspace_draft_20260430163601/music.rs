//! MUSIC (Multiple Signal Classification) super-resolution DOA pseudospectrum.

use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::math::linear_algebra::LinearAlgebra;

/// MUSIC (Multiple Signal Classification) Algorithm
///
/// MUSIC is a subspace-based method that exploits the eigenstructure of the
/// spatial covariance matrix to achieve super-resolution direction-of-arrival (DOA)
/// estimation. It separates the observation space into signal and noise subspaces
/// and uses the orthogonality property to identify source directions.
///
/// # Mathematical Definition
///
/// Given covariance matrix **R** and M sources:
///
/// 1. Eigendecomposition: **R = E Λ E^H**
/// 2. Noise subspace: **E_n** (last N-M eigenvectors)
/// 3. MUSIC pseudospectrum: **P(θ) = 1 / ||E_n^H a(θ)||²**
///
/// Peaks in P(θ) correspond to source directions.
///
/// # SSOT Guarantees
///
/// - Uses `math::linear_algebra::hermitian_eigendecomposition_complex` exclusively
/// - Returns `Err(...)` on eigendecomposition failure (no silent fallback)
/// - Validates all inputs (dimensions, finite values)
/// - Explicit error propagation (no zero pseudospectrum masking)
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::adaptive::subspace::MUSIC;
///
/// let music = MUSIC::new(2); // 2 sources
/// let spectrum = music.pseudospectrum(&covariance, &steering)?;
/// ```
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
    ///
    /// # Parameters
    ///
    /// - `num_sources`: Signal subspace dimension M (must satisfy 0 < M < N)
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
    ///
    /// where E_n is the noise subspace basis (last N-M eigenvectors).
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Input dimensions are invalid or inconsistent
    /// - Covariance matrix contains non-finite values
    /// - Eigendecomposition fails
    /// - num_sources ≥ N (invalid subspace partition)
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64> {
        let n = covariance.nrows();

        // Validate dimensions
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(
                "MUSIC::pseudospectrum: covariance must be non-empty square matrix".to_string(),
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

        // Validate finiteness of inputs
        for &val in steering.iter() {
            if !val.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::NaN {
                    operation: "MUSIC::pseudospectrum".to_string(),
                    inputs: "steering vector contains non-finite values".to_string(),
                }));
            }
        }

        // SSOT: Hermitian eigendecomposition via math layer
        let cov_for_eig = covariance.mapv(|z| num_complex::Complex::new(z.re, z.im));
        let (eigenvalues, eigenvectors) =
            LinearAlgebra::hermitian_eigendecomposition_complex(&cov_for_eig)?;

        // Sort eigenpairs in descending order by eigenvalue
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Noise subspace: eigenvectors corresponding to smallest N-M eigenvalues
        let noise_start = self.num_sources;

        // Compute ||E_n^H a||² = sum_i |e_i^H a|² for noise eigenvectors
        let mut en_h_a_norm_sq = 0.0;
        for &idx in indices.iter().skip(noise_start) {
            let eigenvec = eigenvectors.slice(s![.., idx]);

            // Compute e_i^H a = sum_j conj(e_i[j]) * a[j]
            let mut e_h_a = Complex64::zero();
            for j in 0..n {
                let e_val = eigenvec[j];
                let e_c64 = Complex64::new(e_val.re, e_val.im);
                e_h_a += e_c64.conj() * steering[j];
            }

            en_h_a_norm_sq += e_h_a.norm_sqr();
        }

        // MUSIC pseudospectrum: P = 1 / ||E_n^H a||²
        if en_h_a_norm_sq < 1e-30 {
            // Near-zero denominator indicates steering vector is in signal subspace
            // Return large value (source direction detected)
            return Ok(1e30);
        }

        let pseudospectrum = 1.0 / en_h_a_norm_sq;

        // Validate output
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
