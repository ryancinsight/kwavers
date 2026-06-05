//! Sonoluminescence sensor module
//!
//! This module contains the sonoluminescence detector and related functionality.

pub mod detector;

// Explicit re-exports of sonoluminescence detector types
pub use detector::{
    DetectorConfig, SonoluminescenceDetector, SonoluminescenceEvent, SonoluminescenceStatistics,
};
