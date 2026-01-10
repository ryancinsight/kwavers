//! Cavitation detection components
//!
//! Provides time-domain and spectral-feature detectors for cavitation monitoring.
//! The primary entry points are [`BroadbandDetector`], [`SpectralDetector`],
//! and [`SubharmonicDetector`], all implementing [`CavitationDetector`].
//!
//! For end-user code, these detector types are also re-exported from
//! [`crate::physics::cavitation_control`].
//!
//! # Example
//!
//! ```rust
//! use kwavers::physics::cavitation_control::{CavitationDetector, SpectralDetector};
//! use ndarray::Array1;
//!
//! let mut detector = SpectralDetector::new(1e6, 10e6);
//! let signal = Array1::<f64>::zeros(1024);
//! let metrics = detector.detect(&signal.view());
//! let _ = metrics;
//! ```

pub mod broadband;
pub mod constants;
pub mod spectral;
pub mod subharmonic;
pub mod traits;
pub mod types;

// Re-export main types for convenience.
pub use crate::domain::core::constants::*;
pub use broadband::BroadbandDetector;
pub use spectral::SpectralDetector;
pub use subharmonic::SubharmonicDetector;
pub use traits::{CavitationDetector, DetectorParameters};
pub use types::{CavitationMetrics, CavitationState, DetectionMethod, HistoryBuffer};
