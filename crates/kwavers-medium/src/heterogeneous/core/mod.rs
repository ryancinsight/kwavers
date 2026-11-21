//! Core module for heterogeneous medium structure
//!
//! Following deep hierarchical organization per Rust Book Ch.7

pub mod structure;

// Re-export core type
pub use structure::HeterogeneousMedium;

mod absorption_field;
pub use absorption_field::AbsorptionField;
