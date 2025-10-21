//! Transducer Physics Module
//!
//! This module implements ultrasound transducer field calculations and optimizations.
//!
//! ## Submodules
//!
//! - `fast_nearfield`: Fast Nearfield Method (FNM) for O(n) transducer field calculation

pub mod fast_nearfield;

pub use fast_nearfield::{FastNearfieldMethod, FnmConfiguration};
