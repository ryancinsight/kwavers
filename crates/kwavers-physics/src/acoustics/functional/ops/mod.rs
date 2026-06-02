//! Field Operations and Combinators
//!
//! Provides enhanced field operations with iterator support,
//! better ergonomics, and optimizations for sparse operations.

pub mod field_ops;
pub mod kernel;
pub mod reduction;

pub use field_ops::FieldOps;
pub use kernel::{apply_kernel, apply_kernel_parallel, compose_operations, windowed_operation};
pub use reduction::FieldReduction;

#[cfg(test)]
mod tests;
