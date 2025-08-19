//! Functional Programming Utilities for Physics Calculations
//!
//! This module provides a comprehensive set of functional programming tools
//! for physics field transformations, operations, and computations.
//! Organized into focused submodules for maintainability.

pub mod transform;  // Field transformation pipelines
pub mod ops;        // Field operations and combinators
pub mod iter;       // Lazy iterators and evaluation
pub mod result;     // Monadic operations for Result types

// Re-export commonly used types
pub use transform::FieldTransform;
pub use ops::{FieldOps, apply_kernel, compose_operations};
pub use iter::LazyFieldIterator;
pub use result::ResultOps;