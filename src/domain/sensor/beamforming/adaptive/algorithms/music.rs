//! MUSIC (Multiple Signal Classification) algorithm

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::BeamformingAlgorithm;

/// MUSIC (Multiple Signal Classification) algorithm
///
/// MUSIC is a subspace-based method that exploits the eigenstructure of
/// the covariance matrix to estimate directions of arrival (DOA).
///
/// The MUSIC pseudospectrum is: P_MUSIC(θ) = 1 / (a^H P_N P_N^H a)
///
/// where:
/// - P_N is the projection onto the noise subspace
/// - a is the steering vector
///
/// # References
/// - Schmidt (1986), "Multiple emitter location and signal parameter estimation",
///   IEEE Transactions on Antennas and Propagation, 34(3), 276-280
/// - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5), 720-741
#[derive(Debug)]
pub struct MUSIC {
    /// Number of sources (signals)
    pub num_sources: usize,
}

impl MUSIC {
    /// Create MUSIC algorithm with specified number of sources
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self { num_sources }
    }

    /// Compute MUSIC pseudospectrum value for given steering vector.
    ///
    /// # Mathematical definition
    /// Let `R` be the (Hermitian) covariance matrix. Let `E_n` be the noise-subspace basis
    /// (eigenvectors associated with the smallest `M - d` eigenvalues, where `d` is the number of sources).
    ///
    /// The MUSIC pseudospectrum is:
    /// `P_MUSIC = 1 / (aᴴ E_n E_nᴴ a)`
    ///
    /// # SSOT / correctness policy
    /// This function **does not** silently fall back to dummy results. All failures are surfaced
    /// as errors to prevent error masking.
    ///
    /// # Errors
    /// Returns an error if:
    /// - input dimensions are inconsistent
    /// - invariants are violated (non-finite values)
    /// - complex eigendecomposition is not available in SSOT
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64> {
        let n = covariance.nrows();
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(
                "MUSIC::pseudospectrum requires a non-empty square covariance matrix".to_string(),
            ));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "MUSIC::pseudospectrum steering length ({}) must match covariance dimension ({n})",
                steering.len()
            )));
        }
        if self.num_sources > n {
            return Err(KwaversError::InvalidInput(format!(
                "MUSIC::pseudospectrum num_sources ({}) must be <= n ({n})",
                self.num_sources
            )));
        }

        if self.num_sources == n {
            return Err(KwaversError::InvalidInput(
                "MUSIC::pseudospectrum requires at least one noise eigenvector: num_sources must be < n"
                    .to_string(),
            ));
        }

        // SSOT complex Hermitian eigendecomposition.
        // Convert to SSOT Complex<f64> signature.
        let cov_ssot = covariance.mapv(|z| num_complex::Complex::new(z.re, z.im));
        let (eigs, vecs) = LinearAlgebra::hermitian_eigendecomposition_complex(&cov_ssot)?;

        // Sort eigenpairs by descending eigenvalue (signal subspace first).
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| {
            eigs[j]
                .partial_cmp(&eigs[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Noise subspace dimension = n - d
        let d = self.num_sources;
        let noise_dim = n - d;

        // Compute denom = aᴴ E_n E_nᴴ a = ||E_nᴴ a||²
        // This is non-negative real by construction.
        let mut denom = 0.0f64;
        for k in 0..noise_dim {
            let col = idx[d + k];
            let mut proj = Complex64::new(0.0, 0.0);
            for i in 0..n {
                let e = vecs[(i, col)];
                let e = Complex64::new(e.re, e.im);
                proj += e.conj() * steering[i];
            }
            denom += proj.norm_sqr();
        }

        if !denom.is_finite() || denom <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(
                    "MUSIC::pseudospectrum: non-positive or non-finite denominator a^H Pn a"
                        .to_string(),
                ),
            ));
        }

        Ok(1.0 / denom)
    }
}

impl BeamformingAlgorithm for MUSIC {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        // MUSIC doesn't directly provide beamforming weights; it provides a pseudospectrum.
        // We return the steering vector as a conventional placeholder *without* claiming MVDR-optimality.
        // This is a deterministic, explicit behavior (no numerical fallbacks are hidden here).
        Ok(steering.clone())
    }
}
