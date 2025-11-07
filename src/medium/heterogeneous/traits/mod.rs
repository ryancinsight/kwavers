//! Consolidated trait implementations for heterogeneous media
//!
//! **Architecture**: Deep hierarchical organization per Rust Book Ch.7
//! **Principle**: Separation of Concerns following Clean Architecture
//!
//! **Note**: Only essential acoustic traits implemented initially
//! Additional traits to be added incrementally per SOLID principles

pub mod acoustic;
pub mod elastic;
pub mod viscous;
pub mod thermal;
pub mod optical;
pub mod bubble;

// Re-export trait implementations for external use
pub use acoustic::*;
pub use elastic::*;
pub use viscous::*;
pub use thermal::*;
pub use optical::*;
pub use bubble::*;
