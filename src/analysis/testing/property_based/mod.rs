//! Property-based testing framework
//!
//! **Evidence-Based Testing**: Following ACM FSE 2025 standards
//! **Risk Mitigation**: Edge case discovery per senior engineering requirements

pub mod acoustic;

// Explicit re-exports of property-based test generators
pub use acoustic::{acoustic_properties, grid_properties, medium_properties};
