//! Domain-layer time-domain beamforming interface.
//!
//! This module re-exports time-domain beamforming types from the analysis layer
//! for convenience when working with sensor-specific code.
//!
//! The canonical implementations are in:
//! - [`crate::analysis::signal_processing::beamforming::time_domain`]

// Re-export the canonical DelayReference from the analysis layer
pub use crate::analysis::signal_processing::beamforming::time_domain::DelayReference;

/// Default delay reference (sensor index 0)
pub const DEFAULT_DELAY_REFERENCE: DelayReference = DelayReference::SensorIndex(0);
