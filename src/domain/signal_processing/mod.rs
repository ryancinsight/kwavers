//! Signal Processing Domain Layer
//!
//! Provides domain-level abstractions for signal processing operations that are
//! independent of implementation details (physics-based, neural network, etc.).
//!
//! This ensures clean architecture where:
//! - Physics layer computes theoretical values
//! - Analysis layer implements algorithms
//! - Clinical layer applies results
//! - All layers depend on domain abstractions, not on each other

pub mod beamforming;
pub mod filtering;
pub mod localization;
pub mod pam;

pub use beamforming::{BeamPattern, BeamformingConfig, BeamformingProcessor, BeamformingResult};
pub use filtering::FilterProcessor;
pub use localization::LocalizationProcessor;
pub use pam::PAMProcessor;
