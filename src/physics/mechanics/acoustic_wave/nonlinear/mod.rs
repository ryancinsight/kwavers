// src/physics/mechanics/acoustic_wave/nonlinear/mod.rs

// This module now organizes the different components of the nonlinear wave model.

// Configuration and struct definition
pub mod config;

// Core simulation logic (implementation of the AcousticWaveModel trait)
pub mod core;

// Helper functions (e.g., for calculating phase factors)
pub mod helpers;

// Performance monitoring and reporting
pub mod performance;

// Stability checks and enforcement (e.g., CFL condition, clamping)
pub mod stability;

// Trait implementations
pub mod trait_impls;
