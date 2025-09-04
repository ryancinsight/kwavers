//! Physics simulation modules organized according to SSOT and SOLID principles
//!
//! This module provides the core physics implementations structured as:
//! - mod.rs: Trait definitions with invariants (design-by-contract)
//! - solvers.rs: Numerical methods consolidated
//! - data.rs: In-memory structures optimized for zero-copy operations
//! - gpu.rs: wgpu-rs integration for cross-platform GPU acceleration  
//! - constants.rs: Single source of truth for all physical constants

// Core physics modules as required by architecture
pub mod solvers;    // Numerical methods (FDTD, PSTD, spectral-DG)
pub mod data;       // In-memory data structures with SIMD optimization
pub mod gpu;        // wgpu-rs GPU acceleration
pub mod constants;  // Physical constants (SSOT)

// Legacy modules for backward compatibility (to be refactored)
pub mod bubble_dynamics;
pub mod cavitation_control;
pub mod chemistry;
pub mod phase_modulation;
pub mod field_indices; // Unified field indices (SSOT)
pub mod field_mapping; // Unified field mapping system
pub mod heterogeneity;
pub mod mechanics;
pub mod imaging; // Unified imaging physics module
pub mod optics;
pub mod plugin; // Plugin architecture for extensible physics
pub mod sonoluminescence_detector;
pub mod state;
pub mod therapy; // Unified therapy physics module
pub mod thermal; // Unified thermal physics module
pub mod traits;
pub mod validation;
pub mod wave_propagation; // Wave propagation with reflection and refraction

pub mod analytical;
pub mod conservation;

#[cfg(test)]
pub mod validation_tests; // Literature-based validation tests

/// Functional programming utilities for physics calculations
///
/// This module provides a comprehensive set of functional programming tools
/// organized into focused submodules for maintainability and performance.
pub mod functional;

// Re-export core physics components
pub use solvers::{PhysicsSolver, FdtdSolver, PstdSolver, AdaptiveSolver, AdaptiveConfig};
pub use data::{
    AlignedField, AcousticFields, ThermalFields, BubbleFields, PhysicsData, 
    MemorySummary, FieldProcessor
};
pub use constants::*;

// Re-export GPU components when feature is enabled
#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuFdtdSolver, GpuMemoryManager, SimulationUniforms};

// Legacy exports for backward compatibility
pub use bubble_dynamics::{BubbleField, BubbleParameters, BubbleState};
pub use field_mapping::{
    FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut, UnifiedFieldType,
};
pub use optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
pub use plugin::{PhysicsPlugin, PluginContext, PluginManager, PluginMetadata};
pub use state::PhysicsState;
pub use traits::*;
pub use wave_propagation::{
    AttenuationCalculator, Polarization, PropagationCoefficients, WaveMode,
    WavePropagationCalculator,
};
