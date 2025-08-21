//! Functional Programming Utilities for Physics Calculations
//!
//! This module provides a comprehensive set of functional programming tools
//! for physics field transformations, operations, and computations.
//! Organized into focused submodules for maintainability.

pub mod iter; // Lazy iterators and evaluation
pub mod ops; // Field operations and combinators
pub mod result;
pub mod transform; // Field transformation pipelines // Monadic operations for Result types

// Re-export commonly used types
pub use iter::LazyFieldIterator;
pub use ops::{apply_kernel, compose_operations, FieldOps};
pub use result::ResultOps;
pub use transform::FieldTransform;
