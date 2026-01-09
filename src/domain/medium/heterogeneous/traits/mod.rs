//! Consolidated trait implementations for heterogeneous media
//!
//! **Architecture**: Deep hierarchical organization per Rust Book Ch.7
//! **Principle**: Separation of Concerns following Clean Architecture
//!
//! **Note**: Only essential acoustic traits implemented initially
//! Additional traits to be added incrementally per SOLID principles

pub mod acoustic;
pub mod bubble;
pub mod elastic;
pub mod optical;
pub mod thermal;
pub mod viscous;
