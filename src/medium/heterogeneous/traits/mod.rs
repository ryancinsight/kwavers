//! Consolidated trait implementations for heterogeneous media
//!
//! **Architecture**: Deep hierarchical organization per Rust Book Ch.7
//! **Principle**: Separation of Concerns following Clean Architecture
//! 
//! **Note**: Only essential acoustic traits implemented initially
//! Additional traits to be added incrementally per SOLID principles

pub mod acoustic;

// Re-export core acoustic traits
pub use acoustic::*;