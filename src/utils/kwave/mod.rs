//! k-Wave Compatible Utility Functions Module
//!
//! This module provides utility functions compatible with k-Wave toolbox,
//! organized into focused submodules following SOLID principles.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for each module
//! - **DRY**: Reusable implementations across the codebase
//! - **Zero-Copy**: Uses iterators and efficient data structures
//! - **KISS**: Clear, well-documented interfaces

pub mod angular_spectrum;
pub mod beam_patterns;
pub mod hounsfield;
pub mod numerical;
pub mod time_reversal;
pub mod water_properties;

// Re-export main types for convenience
pub use angular_spectrum::AngularSpectrum;
pub use beam_patterns::BeamPatterns;
pub use hounsfield::HounsfieldUnits;
pub use numerical::NumericalUtils;
pub use time_reversal::TimeReversalUtils;
pub use water_properties::WaterProperties;
