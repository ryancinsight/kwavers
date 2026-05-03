//! Optimized gradient operations with caching and parallelization.
//!
//! # Theorem: Accuracy of Centered Finite-Difference Gradient Stencils
//!
//! For a smooth function f(x) sampled at spacing Δx, the centered difference
//! approximations achieve the following accuracy orders:
//!
//! ## 2nd-Order Stencil (standard)
//! ```text
//!   ∂f/∂x |_i = (f_{i+1} − f_{i−1}) / (2Δx) + O(Δx²)
//! ```
//! ## 4th-Order Stencil (high accuracy)
//! ```text
//!   ∂f/∂x |_i = (−f_{i+2} + 8f_{i+1} − 8f_{i−1} + f_{i+2}) / (12Δx) + O(Δx⁴)
//! ```
//!
//! # References
//! - Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
//!   spaced grids. Math. Comput. 51(184), 699–706.

pub mod cache;
pub mod functions;
pub mod operator;

pub use cache::GradientCache;
pub use functions::{gradient_optimized, gradient_with_boundaries};
pub use operator::{BoundaryStrategy, GradientOperator, GradientOperatorBuilder};
