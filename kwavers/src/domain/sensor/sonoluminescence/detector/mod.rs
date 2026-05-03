//! Sonoluminescence Detection and Analysis
//!
//! Provides complete detection and analysis of sonoluminescence events
//! based on physical criteria from established literature.
//!
//! References:
//! - Brenner et al. (2002) "Single-bubble sonoluminescence"
//! - Yasui (1997) "Alternative model of single-bubble sonoluminescence"
//! - Gaitan et al. (1992) "Sonoluminescence and bubble dynamics"

pub mod constants;
pub mod core;
pub mod types;

pub use core::SonoluminescenceDetector;
pub use types::{DetectorConfig, SonoluminescenceEvent, SonoluminescenceStatistics};
