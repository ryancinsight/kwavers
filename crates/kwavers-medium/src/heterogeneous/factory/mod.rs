//! Factory module for heterogeneous media creation

pub mod general;
pub mod tissue;

// Re-export factories
pub use general::HeterogeneousFactory;
pub use tissue::TissueFactory;
