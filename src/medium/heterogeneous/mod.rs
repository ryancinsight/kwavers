//! Heterogeneous medium with spatially-varying properties
//!
//! **Deep Hierarchical Architecture**: Following Rust Book Ch.7 per senior engineering standards
//! **Evidence-Based Refactoring**: 479-line monolith → focused submodules per GRASP principles

// Core structure definition
pub mod core;

// Specialized trait implementations  
pub mod traits;

// Interpolation utilities
pub mod interpolation;

// Factory patterns for medium creation
pub mod factory;

// Domain-specific modules
pub mod constants;
pub mod tissue;

pub use core::HeterogeneousMedium;
pub use factory::TissueFactory;
