//! Subspace-Based Adaptive Beamforming Algorithms
//!
//! # Overview
//!
//! Subspace methods exploit the eigenstructure of the spatial covariance matrix
//! to achieve super-resolution direction-of-arrival (DOA) estimation and robust
//! beamforming. By decomposing the observation space into signal and noise subspaces,
//! these algorithms can resolve closely-spaced sources and suppress interference.
//!
//! # Mathematical Foundation
//!
//! ## Eigendecomposition of Covariance Matrix
//!
//! Given N-element array covariance matrix **R** (N×N Hermitian):
//!
//! ```text
//! R = E Λ E^H
//! ```
//!
//! where:
//! - **E**: Matrix of eigenvectors (N×N unitary)
//! - **Λ**: Diagonal matrix of eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ ≥ 0
//!
//! ## Signal and Noise Subspace Partition
//!
//! Assuming M sources (M < N):
//!
//! - **Signal subspace E_s**: First M eigenvectors (large eigenvalues)
//! - **Noise subspace E_n**: Last (N-M) eigenvectors (small eigenvalues)
//!
//! Key property: **a^H E_n = 0** for any source direction **a**
//!
//! ## MUSIC (Multiple Signal Classification)
//!
//! MUSIC exploits orthogonality between steering vectors and noise subspace:
//!
//! ```text
//! P_MUSIC(θ) = 1 / ||E_n^H a(θ)||²
//! ```
//!
//! Peaks in P_MUSIC correspond to source directions.
//!
//! ## Eigenspace Minimum Variance (ESMV)
//!
//! ESMV constrains beamforming to the signal subspace for robustness:
//!
//! ```text
//! w = P_s R^{-1} a / (a^H R^{-1} P_s a)
//! ```
//!
//! where **P_s = E_s E_s^H** is the signal subspace projector.
//!
//! # Module layout
//!
//! - [`music`]: `MUSIC` super-resolution DOA pseudospectrum.
//! - [`eigenspace_mv`]: `EigenspaceMV` signal-subspace-projected MVDR
//!   beamformer with diagonal loading.
//!
//! # Mathematical Correctness Verification
//!
//! Implementations include property-based tests to verify:
//!
//! 1. **Orthogonality**: E_n^H a ≈ 0 for source directions
//! 2. **Unit Gain**: w^H a = 1 (MVDR constraint for ESMV)
//! 3. **Positivity**: P_MUSIC(θ) ≥ 0 for all θ
//! 4. **Eigenvalue Ordering**: λ₁ ≥ λ₂ ≥ ... ≥ λₙ
//! 5. **Reconstruction**: R ≈ E Λ E^H (eigendecomposition sanity)
//!
//! # Literature References
//!
//! ## Foundational Papers
//!
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation."
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.
//!   DOI: 10.1109/TAP.1986.1143830
//!
//! - Barabell, A. (1983). "Improving the resolution performance of eigenstructure-based
//!   direction-finding algorithms." *IEEE ICASSP*, 8, 336-339.
//!
//! ## Advanced Topics
//!
//! - Stoica, P., & Nehorai, A. (1990). "MUSIC, maximum likelihood, and Cramer-Rao bound."
//!   *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 38(5), 720-741.
//!   DOI: 10.1109/29.56027
//!
//! - Gershman, A. B., et al. (1999). "Adaptive beamforming algorithms with robustness
//!   against jammer motion." *IEEE Transactions on Signal Processing*, 47(7), 1878-1885.
//!   DOI: 10.1109/78.771038

mod eigenspace_mv;
mod music;

#[cfg(test)]
mod tests;

pub use eigenspace_mv::EigenspaceMV;
pub use music::MUSIC;
