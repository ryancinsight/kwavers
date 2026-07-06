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
pub mod gradient_optimized;
pub mod laplacian;

// Re-export main types
pub use coefficients::{FDCoefficients, FdAccuracyOrder};
pub use curl::{curl, curl_leto};
pub use divergence::{divergence, divergence_leto};
pub use gradient::{gradient, gradient_leto};
pub use gradient_optimized::{GradientCache, GradientOperator, GradientOperatorBuilder};
pub use laplacian::{laplacian, laplacian_leto};
