//! Subspace-Based Adaptive Beamforming Algorithms
//!
//! Subspace methods exploit the eigenstructure of the spatial covariance matrix
//! to achieve super-resolution direction-of-arrival (DOA) estimation and robust
//! beamforming. By decomposing the observation space into signal and noise subspaces,
//! these algorithms can resolve closely-spaced sources and suppress interference.
//!
//! # Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::adaptive::subspace (Layer 7)
//!   ↓ imports from
//! math::linear_algebra::LinearAlgebra (Layer 1) - SSOT eigendecomposition
//! core::error (Layer 0) - error handling
//! ```
//!
//! # Algorithms
//!
//! - `MUSIC`: Multiple Signal Classification — super-resolution DOA estimation
//! - `EigenspaceMV`: Eigenspace Minimum Variance — robust adaptive beamforming
//!
//! # References
//!
//! - Schmidt (1986), "Multiple emitter location and signal parameter estimation"
//! - Gershman et al. (1999), "Adaptive beamforming algorithms with robustness
//!   against jammer motion"

mod esmv;
mod music;
#[cfg(test)]
mod tests;

pub use esmv::EigenspaceMV;
pub use music::MUSIC;
