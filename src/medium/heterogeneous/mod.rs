//! Heterogeneous medium module with spatially varying properties

pub mod constants;
pub mod implementation;
pub mod tissue;

// Re-export the main heterogeneous medium type
pub use implementation::HeterogeneousMedium;
