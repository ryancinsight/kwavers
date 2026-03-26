//! Nonlinear acoustic wave propagation module
//!
//! This module provides implementations for nonlinear acoustic wave propagation,
//! including finite-amplitude effects and harmonic generation.

pub mod multi_frequency;
pub mod numerical_methods;
pub mod trait_impl;
pub mod wave_model;

pub use multi_frequency::MultiFrequencyConfig;
pub use wave_model::NonlinearWave;

// Re-export commonly used items
pub use crate::physics::traits::AcousticWaveModel;
