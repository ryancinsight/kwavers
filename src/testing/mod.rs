//! Testing utilities and frameworks
//!
//! **Deep Testing Architecture**: Following systematic testing principles
//! **Standards**: ICSE 2020, FSE 2025 property-based testing methodologies

pub mod property_based;

// Re-export primary testing framework
pub use property_based::*;