// src/physics/mod.rs
pub mod bubble_dynamics;
pub mod chemistry;
// composable module removed - use plugin system instead
pub mod field_indices;  // Unified field indices (SSOT)
pub mod field_mapping;  // NEW: Unified field mapping system
pub mod heterogeneity;
pub mod mechanics;
// migration module removed - composable has been fully removed
pub mod optics;
pub mod plugin; // NEW: Plugin architecture for extensible physics
pub mod scattering;
pub mod state;
pub mod thermodynamics;
pub mod traits;
pub mod sonoluminescence_detector;

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

/// Enhanced functional programming utilities for physics calculations
/// 
/// This module provides a comprehensive set of functional programming tools
/// organized into focused submodules for better maintainability and performance.
pub mod functional;