//! Signal Processing Domain Layer
//!
//! **Deprecation Notice (Phase 2 Migration)**: This module provides domain-level abstractions
//! for signal processing that are being migrated to more appropriate layers:
//!
//! **Migration Guide:**
//! - Beamforming interfaces → Analysis layer (`analysis::signal_processing::beamforming`)
//! - Beamforming configs → Domain sensor module (`domain::sensor::beamforming`)
//! - Filter processors → Analysis layer (`analysis::signal_processing::clutter_filter`)
//! - Localization processors → Analysis layer (`analysis::signal_processing::localization`)
//! - PAM processors → Analysis layer (`analysis::signal_processing::pam`)
//!
//! **Architecture Principle**:
//! - Domain layer: Type definitions and core abstractions only
//! - Analysis layer: Algorithm implementations and post-processing
//! - Physics layer: Physical law computations (no dependencies on analysis)
//! - Clinical layer: Application-specific workflows
//!
//! This module is kept for backward compatibility but new code should import from
//! the analysis layer directly.

pub mod beamforming;
pub mod filtering;
pub mod localization;
pub mod pam;

pub use beamforming::{BeamPattern, BeamformingConfig, BeamformingProcessor, BeamformingResult};
pub use filtering::FilterProcessor;
pub use localization::LocalizationProcessor;
pub use pam::PAMProcessor;
