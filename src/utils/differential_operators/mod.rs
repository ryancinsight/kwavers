//! Differential operators module with GRASP compliance
//!
//! This module provides modular differential operators following GRASP principles:
//! - Information Expert: Each operator type owns its implementation
//! - Single Responsibility: Separate modules for each operator category
//! - High Cohesion: Related operations grouped together
//!
//! References:
//! - GRASP Principles: Larman, "Applying UML and Patterns"
//! - Rust Book Ch.7: Module organization
//! - IEEE TSE 2022: Memory safety practices

pub mod coefficients;
pub mod gradient;
pub mod divergence;
pub mod curl;
pub mod laplacian;

// Re-export main types
pub use coefficients::{FDCoefficients, SpatialOrder};
pub use gradient::gradient;
pub use divergence::divergence;
pub use curl::curl;
pub use laplacian::laplacian;