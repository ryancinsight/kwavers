//! Nonlinear acoustic wave propagation module
//!
//! This module provides implementations for nonlinear acoustic wave propagation,
//! including finite-amplitude effects and harmonic generation.

mod multi_frequency;
mod numerical_methods;
mod trait_impl;
mod wave_model;

pub use multi_frequency::MultiFrequencyConfig;
pub use wave_model::NonlinearWave;

// Re-export commonly used items
pub use crate::physics::traits::AcousticWaveModel;
