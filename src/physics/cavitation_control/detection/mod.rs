//! Modular cavitation detection components
//!
//! This module provides various detection algorithms for cavitation monitoring,
//! split into focused, composable components following CUPID principles.

pub mod broadband;
pub mod constants;
pub mod spectral;
pub mod subharmonic;
pub mod traits;
pub mod types;

// Re-export main types for convenience
pub use broadband::BroadbandDetector;
pub use crate::physics::constants::*;
pub use spectral::SpectralDetector;
pub use subharmonic::SubharmonicDetector;
pub use traits::{CavitationDetector, DetectorParameters};
pub use types::{CavitationMetrics, CavitationState, DetectionMethod, HistoryBuffer};
