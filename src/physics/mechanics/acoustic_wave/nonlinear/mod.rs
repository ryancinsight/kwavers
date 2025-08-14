// src/physics/mechanics/acoustic_wave/nonlinear/mod.rs

// This module now organizes the different components of the nonlinear wave model.

// Core simulation logic (implementation of the AcousticWaveModel trait)
pub mod core;
// Simulation state module temporarily disabled - double buffering implemented in-place

// Re-export the main struct and multi-frequency configuration
pub use core::{NonlinearWave, MultiFrequencyConfig};
