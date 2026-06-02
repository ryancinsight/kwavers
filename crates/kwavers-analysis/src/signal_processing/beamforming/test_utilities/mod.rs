//! Shared Test Utilities for Beamforming Algorithms
//!
//! Provides common test data generators (covariance matrices, steering vectors,
//! angle utilities) used across all beamforming test modules.

pub mod angle;
pub mod covariance;
pub mod steering;
#[cfg(test)]
mod tests;

pub use covariance::{
    create_diagonal_dominant_covariance, create_identity_covariance,
    create_rank_deficient_covariance, create_test_covariance, TestCovarianceBuilder,
};
pub use steering::create_steering_vector;
