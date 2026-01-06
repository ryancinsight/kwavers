//! Differential operators module for discretized grids
//!
//! This module provides modular differential operators:
//! - Information Expert: Each operator type owns its implementation
//! - Single Responsibility: Separate modules for each operator category
//! - High Cohesion: Related operations grouped together

pub mod coefficients;
pub mod curl;
pub mod divergence;
pub mod gradient;
pub mod laplacian;

// Re-export main types
pub use coefficients::{FDCoefficients, SpatialOrder};
pub use curl::curl;
pub use divergence::divergence;
pub use gradient::gradient;
pub use laplacian::laplacian;
