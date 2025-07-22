// src/physics/mechanics/acoustic_wave/nonlinear/mod.rs

// This module now organizes the different components of the nonlinear wave model.

// Core simulation logic (implementation of the AcousticWaveModel trait)
pub mod core;
pub mod optimized;

// Re-export the main structs
pub use core::NonlinearWave;
pub use optimized::OptimizedNonlinearWave;
