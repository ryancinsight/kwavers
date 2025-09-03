//! Heterogeneous tissue medium implementation
//!
//! Modular architecture separating concerns per SOLID principles

mod implementation;
mod properties;
mod region;

pub use implementation::HeterogeneousTissueMedium;
pub use properties::TissuePropertyCache;
pub use region::{TissueMap, TissueRegion};

// Re-export tissue types from absorption module
pub use crate::medium::absorption::TissueType;
