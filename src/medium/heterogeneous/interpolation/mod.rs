//! Interpolation utilities for heterogeneous media
//!
//! Modular interpolation system supporting multiple algorithms
//! with zero-cost abstractions per Rust performance guidelines.

pub mod trilinear;

// Re-export primary interpolator
pub use trilinear::TrilinearInterpolator;
