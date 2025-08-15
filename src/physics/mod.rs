// src/physics/mod.rs
pub mod bubble_dynamics;
pub mod cavitation_control;
pub mod chemistry;
pub mod phase_modulation;
// composable module removed - use plugin system instead
pub mod field_indices;  // Unified field indices (SSOT)
pub mod field_mapping;  // NEW: Unified field mapping system
pub mod heterogeneity;
pub mod mechanics;
// migration module removed - composable has been fully removed
pub mod optics;
pub mod plugin; // Plugin architecture for extensible physics
pub mod state;
pub mod thermal; // Unified thermal physics module
pub mod therapy; // Unified therapy physics module
pub mod imaging; // Unified imaging physics module
pub mod traits;
pub mod sonoluminescence_detector;
pub mod wave_propagation; // NEW: Wave propagation with reflection and refraction

#[cfg(test)]
pub mod analytical_tests;

#[cfg(test)]
pub mod validation_tests;

// Re-export commonly used types
pub use bubble_dynamics::{BubbleField, BubbleState, BubbleParameters};
// Removed composable exports - use plugin system instead
pub use field_mapping::{UnifiedFieldType, FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut};
pub use state::PhysicsState;
pub use traits::*;
pub use plugin::{PhysicsPlugin, PluginManager, PluginMetadata, PluginContext}; // NEW: Plugin exports
pub use optics::sonoluminescence::{SonoluminescenceEmission, EmissionParameters};
pub use wave_propagation::{WavePropagationCalculator, WaveMode, Polarization, PropagationCoefficients};

/// Enhanced functional programming utilities for physics calculations
/// 
/// This module provides a comprehensive set of functional programming tools
/// organized into focused submodules for better maintainability and performance.
pub mod functional;