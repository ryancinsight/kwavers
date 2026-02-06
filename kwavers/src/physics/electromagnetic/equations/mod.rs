//! Electromagnetic wave equations and physics implementations
//!
//! This module provides the core electromagnetic physics implementations,
//! including Maxwell's equations, material properties, field calculations,
//! and source definitions.

pub mod fields;
pub mod materials;
pub mod traits;
pub mod types;

// Re-export for convenient access
pub use fields::*;
pub use materials::*;
pub use traits::*;
pub use types::*;

// Re-export key items from parent module for backward compatibility
