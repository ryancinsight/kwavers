//! Regularization Methods for Ill-Posed Inverse Problems
//!
//! Provides Tikhonov (L2), Total Variation, Smoothness (Laplacian), L1 (Lasso),
//! and depth-weighting regularization for 1D, 2D, and 3D model arrays.
//!
//! **References**:
//! - Tikhonov & Arsenin (1977): "Solutions of Ill-posed Problems"
//! - Rudin, Osher, Fatemi (1992): "Nonlinear total variation based noise removal"
//! - Hastie, Tibshirani, Wainwright (2015): "Statistical Learning with Sparsity"

pub mod config;
pub mod regularizer_1d;
pub mod regularizer_2d;
pub mod regularizer_3d;

#[cfg(test)]
mod tests;

pub use config::RegularizationConfig;
pub use regularizer_1d::ModelRegularizer1D;
pub use regularizer_2d::ModelRegularizer2D;
pub use regularizer_3d::ModelRegularizer3D;
