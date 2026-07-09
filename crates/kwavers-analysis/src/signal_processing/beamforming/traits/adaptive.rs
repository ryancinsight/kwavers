//! Adaptive beamforming trait — data-driven weight optimization with
//! diagonal loading and pseudospectrum.

use leto::{
    Array1,
    Array2,
};
use num_complex::Complex64;

use kwavers_core::error::KwaversResult;

use super::FrequencyDomainBeamformer;

/// Adaptive beamforming algorithms (narrowband frequency-domain specialization).
///
/// Adaptive beamformers compute data-dependent weights that optimize performance
/// criteria such as signal-to-interference-plus-noise ratio (SINR). They automatically
/// form nulls in interference directions.
///
/// # Mathematical Framework
///
/// Given:
/// - Covariance matrix **R** (N×N Hermitian)
/// - Steering vector **a** (N×1 complex)
///
/// Compute optimal weights **w** via algorithm-specific optimization:
///
/// ## MVDR (Minimum Variance Distortionless Response)
/// ```text
/// minimize   w^H R w
/// subject to w^H a = 1
/// Solution:  w = R^{-1} a / (a^H R^{-1} a)
/// ```
///
/// ## Maximum SINR
/// ```text
/// maximize   w^H R_s w / w^H R_n w
/// Solution:  w ∝ R_n^{-1} R_s w (generalized eigenproblem)
/// ```
///
/// # Characteristics
///
/// - **Adaptivity**: Weights computed from data covariance
/// - **Interference Rejection**: Automatic null formation
/// - **Complexity**: O(N³) for matrix inversion
/// - **SNR Requirement**: Moderate to high SNR needed for stability
///
/// # Implementations
///
/// - `MinimumVariance`: MVDR/Capon with diagonal loading
/// - `EigenspaceMV`: Subspace-based MVDR
/// - `MUSIC`: Pseudospectrum via noise subspace projection
pub trait AdaptiveFrequencyBeamformer: FrequencyDomainBeamformer {
    /// Compute optimal beamforming weights from covariance and steering vector.
    ///
    /// This is the core operation for adaptive beamformers. Different algorithms
    /// implement different optimization criteria (MVDR, MUSIC, etc.).
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N Hermitian)
    /// - `steering`: Steering vector **a** (N×1 complex)
    ///
    /// # Returns
    ///
    /// Optimal weight vector **w** (N×1 complex) according to algorithm's criterion.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if (no silent fallbacks):
    /// - Dimensions inconsistent (covariance not square, steering length mismatch)
    /// - Covariance matrix singular or ill-conditioned
    /// - Numerical operation fails (solve, eigendecomposition, etc.)
    /// - Algorithm-specific constraints violated
    ///
    /// **MUST NOT**:
    /// - Return steering vector as fallback on failure
    /// - Return zero weights to mask errors
    /// - Silently skip numerical issues
    ///
    /// # Mathematical Guarantees
    ///
    /// Implementations must ensure:
    /// - Unit gain constraint (if applicable): `w^H a = 1`
    /// - Numerical stability (e.g., via diagonal loading)
    /// - Hermitian covariance matrix preserved
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(N²) to O(N³) depending on algorithm
    /// - **Space Complexity**: O(N²) for covariance matrix
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>>;

    /// Get diagonal loading factor for numerical stability.
    ///
    /// Diagonal loading adds a small value to covariance matrix diagonal to
    /// prevent singular matrices and improve robustness.
    ///
    /// # Mathematical Effect
    ///
    /// ```text
    /// R_loaded = R + ε·I
    /// ```
    ///
    /// Typical values: ε ∈ [1e-6, 1e-2] depending on SNR and condition number.
    fn diagonal_loading(&self) -> f64;

    /// Compute adaptive beamforming pseudospectrum (for DOA estimation).
    ///
    /// The pseudospectrum is a spatial function used for direction-of-arrival
    /// estimation. It peaks at source locations.
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N)
    /// - `steering`: Steering vector **a** (N×1) for scan direction
    ///
    /// # Returns
    ///
    /// Pseudospectrum value (non-negative real scalar).
    ///
    /// # Mathematical Definitions
    ///
    /// **MVDR pseudospectrum:**
    /// ```text
    /// P_MVDR(θ) = 1 / (a^H(θ) R^{-1} a(θ))
    /// ```
    ///
    /// **MUSIC pseudospectrum:**
    /// ```text
    /// P_MUSIC(θ) = 1 / (a^H(θ) E_n E_n^H a(θ))
    /// ```
    /// where E_n is the noise subspace eigenvector matrix.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Covariance matrix singular or ill-conditioned
    /// - Numerical operations fail
    ///
    /// **MUST NOT** return 0.0 as error masking.
    fn compute_pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64>;
}
