//! Nonlinear acoustic wave propagation module
//!
//! This module provides implementations for nonlinear acoustic wave propagation,
//! including finite-amplitude effects and harmonic generation.

mod wave_model;
mod multi_frequency;
mod numerical_methods;
mod trait_impl;

pub use wave_model::NonlinearWave;
pub use multi_frequency::MultiFrequencyConfig;

// Re-export commonly used items
pub use crate::physics::traits::AcousticWaveModel;